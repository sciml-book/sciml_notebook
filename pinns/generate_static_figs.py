"""
Generate PDF figures for the Poisson equation section (Chapter 6 PINNs).

Usage:
    python generate_static_figs.py

Produces:
    - poisson-result-nn.pdf
    - poisson-result-pinn.pdf
    - poisson-extrapolation.pdf
    - poisson-spectral-bias.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ── Style and paths ──────────────────────────────────────────────────────────
STYLE_PATH = os.path.join(os.path.dirname(__file__),
                          '..', 'sciml_style.mplstyle')
plt.style.use(STYLE_PATH)

FIGS_DIR = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'sciml-book', 'chapters', '06-pinns', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)


# ── Problem definition ───────────────────────────────────────────────────────
class PoissonProblem:
    """One-dimensional Poisson equation: -u'' = f on (0,1), u(0)=u(1)=0."""

    def __init__(self, source_freq=1.0):
        self.freq = source_freq

    def exact_solution(self, x):
        return np.sin(self.freq * np.pi * x)

    def source_term(self, x):
        return (self.freq * np.pi)**2 * np.sin(self.freq * np.pi * x)

    def generate_data(self, num_domain_points=500, num_data_points=5,
                      data_x_min=0.25, data_x_max=0.75):
        x_domain = np.linspace(0, 1, num_domain_points).reshape(-1, 1)
        u_exact = self.exact_solution(x_domain).reshape(-1, 1)
        x_data = np.linspace(data_x_min, data_x_max, num_data_points).reshape(-1, 1)
        u_data = self.exact_solution(x_data).reshape(-1, 1)
        return x_domain, u_exact, x_data, u_data


# ── Neural network ───────────────────────────────────────────────────────────
class NeuralNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── PINN loss ────────────────────────────────────────────────────────────────
def pinn_loss(model, x_data, u_data, x_colloc, x_bc, source_fn,
              lambda_phys=1.0, lambda_bc=10.0):
    # Data loss
    if x_data.numel() > 0:
        u_pred_data = model(x_data)
        loss_data = nn.MSELoss()(u_pred_data, u_data)
    else:
        loss_data = torch.tensor(0.0)

    # Physics loss: -u'' - f = 0
    x_c = x_colloc.clone().detach().requires_grad_(True)
    u_c = model(x_c)
    du_dx = torch.autograd.grad(u_c, x_c,
                                grad_outputs=torch.ones_like(u_c),
                                create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dx, x_c,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True)[0]
    f_val = source_fn(x_c)
    residual = -d2u_dx2 - f_val
    loss_physics = torch.mean(residual ** 2)

    # Boundary loss
    u_left = model(x_bc[0:1])
    u_right = model(x_bc[1:2])
    loss_bc = (u_left**2 + u_right**2).squeeze()

    total = loss_data + lambda_phys * loss_physics + lambda_bc * loss_bc
    return total, loss_data.item(), loss_physics.item(), loss_bc.item()


# ── Training functions ───────────────────────────────────────────────────────
def train_standard_nn(problem, num_domain_points=500, num_data_points=5):
    print('Training Standard NN (data only)...')
    x_domain_np, u_exact_np, x_data_np, u_data_np = problem.generate_data(
        num_domain_points=num_domain_points,
        num_data_points=num_data_points)

    layer_sizes = [1, 32, 32, 32, 1]
    model = NeuralNetwork(layer_sizes)

    x_data_t = torch.tensor(x_data_np, dtype=torch.float32)
    u_data_t = torch.tensor(u_data_np, dtype=torch.float32)
    x_domain_t = torch.tensor(x_domain_np, dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for step in tqdm(range(20000), desc='Standard NN'):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(x_data_t), u_data_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, x_domain_t, x_domain_np, u_exact_np, x_data_np, u_data_np


def train_pinn(problem, num_domain_points=500, num_data_points=5,
               num_colloc_points=50, lambda_phys=1.0, lambda_bc=10.0):
    print('Training PINN (data + physics + BC)...')
    x_domain_np, u_exact_np, x_data_np, u_data_np = problem.generate_data(
        num_domain_points=num_domain_points,
        num_data_points=num_data_points)

    x_colloc_np = np.linspace(0.01, 0.99, num_colloc_points).reshape(-1, 1)

    layer_sizes = [1, 32, 32, 32, 1]
    model = NeuralNetwork(layer_sizes)

    x_data_t = torch.tensor(x_data_np, dtype=torch.float32)
    u_data_t = torch.tensor(u_data_np, dtype=torch.float32)
    x_domain_t = torch.tensor(x_domain_np, dtype=torch.float32)
    x_colloc_t = torch.tensor(x_colloc_np, dtype=torch.float32)
    x_bc_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    freq = problem.freq
    source_fn = lambda x: (freq * np.pi)**2 * torch.sin(freq * np.pi * x)

    # Phase 1: Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    for step in tqdm(range(15000), desc='PINN Adam'):
        model.train()
        optimizer_adam.zero_grad()
        loss, _, _, _ = pinn_loss(model, x_data_t, u_data_t, x_colloc_t,
                                  x_bc_t, source_fn, lambda_phys, lambda_bc)
        loss.backward()
        optimizer_adam.step()

    # Phase 2: L-BFGS
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.5,
                                   max_iter=20, history_size=50,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
    for step in tqdm(range(200), desc='PINN L-BFGS'):
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _, _, _ = pinn_loss(model, x_data_t, u_data_t, x_colloc_t,
                                      x_bc_t, source_fn, lambda_phys, lambda_bc)
            loss.backward()
            return loss
        optimizer_lbfgs.step(closure)

    model.eval()
    return (model, x_domain_t, x_domain_np, u_exact_np,
            x_data_np, u_data_np, x_colloc_np)


# ── Figure: result plots (NN and PINN) ──────────────────────────────────────
def save_result_figure(model, x_domain_t, x_domain_np, u_exact_np,
                       filename, title, x_data_np=None, u_data_np=None,
                       x_colloc_np=None, pred_label='NN prediction'):
    """Save a publication-quality figure of model prediction vs exact.

    Parameters
    ----------
    x_data_np, u_data_np : array or None
        If None, training data points are omitted (e.g. Poisson PINN figure).
    x_colloc_np : array or None
        If provided, collocation points are shown along the bottom.
    """
    model.eval()
    with torch.no_grad():
        u_pred = model(x_domain_t).numpy()

    fig, ax = plt.subplots()

    # NN prediction (underneath)
    ax.plot(x_domain_np, u_pred, color='tab:blue', linewidth=2.5,
            label=pred_label)
    # Exact solution (on top, dashed)
    ax.plot(x_domain_np, u_exact_np, color='#333333', linestyle='--',
            linewidth=2, label='Exact solution', zorder=5)

    # Optional: training data
    if x_data_np is not None and u_data_np is not None:
        ax.scatter(x_data_np, u_data_np, color='tab:red', s=80, zorder=6,
                   label='Training data')

    # Optional: collocation points
    if x_colloc_np is not None:
        ax.scatter(x_colloc_np, np.zeros_like(x_colloc_np) - 0.05,
                   color='#B8860B', s=30, marker='^', alpha=0.8,
                   zorder=4, label='Collocation points')

    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$u(x)$')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([-1.3, 1.3])

    fig.savefig(filename)
    print(f'Saved: {filename}')
    plt.close(fig)


# ── Figure: extrapolation ────────────────────────────────────────────────────
def plot_extrapolation(model_nn, model_pinn, problem, save_path):
    x_ext = np.linspace(-0.5, 1.5, 1000).reshape(-1, 1)
    u_ext_exact = problem.exact_solution(x_ext).reshape(-1, 1)
    x_ext_t = torch.tensor(x_ext, dtype=torch.float32)

    _, _, x_data_np, u_data_np = problem.generate_data()

    model_nn.eval()
    model_pinn.eval()
    with torch.no_grad():
        u_nn_ext = model_nn(x_ext_t).numpy()
        u_pinn_ext = model_pinn(x_ext_t).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.75))

    for i, (u_pred, label, color) in enumerate([
        (u_nn_ext, 'Standard NN', 'tab:blue'),
        (u_pinn_ext, 'PINN', 'tab:blue')
    ]):
        ax = axes[i]
        ax.axvspan(0, 1, alpha=0.08, color='green',
                   label=r'Training domain $[0, 1]$')
        ax.axvspan(0.25, 0.75, alpha=0.12, color='orange',
                   label=r'Data region $[0.25, 0.75]$')

        ax.plot(x_ext, u_ext_exact, color='#333333', linestyle='--',
                linewidth=2, label='Exact', zorder=5)
        ax.plot(x_ext, u_pred, color=color, linewidth=2.5, label=label)
        ax.scatter(x_data_np, u_data_np, color='tab:red', s=60, zorder=6,
                   label='Data')

        mask_in = (x_ext.ravel() >= 0) & (x_ext.ravel() <= 1)
        mask_out = ~mask_in
        l2_in = np.sqrt(np.mean(
            (u_pred[mask_in].ravel() - u_ext_exact[mask_in].ravel())**2))
        l2_out = np.sqrt(np.mean(
            (u_pred[mask_out].ravel() - u_ext_exact[mask_out].ravel())**2))

        ax.set_title(
            f'{label}\n$L_2$ in $[0,1]$: {l2_in:.4f}'
            f' | $L_2$ outside: {l2_out:.4f}')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$u(x)$')
        ax.set_xlim([-0.5, 1.5])
        ax.set_ylim([-1.5, 2.0])
        ax.legend(fontsize=8)

    fig.savefig(save_path)
    print(f'Saved: {save_path}')
    plt.close(fig)


# ── Figure: spectral bias ───────────────────────────────────────────────────
def train_pinn_for_spectral_bias(source_freq,
                                  num_adam_steps=15000, num_lbfgs_steps=200):
    problem = PoissonProblem(source_freq=source_freq)
    x_domain_np, u_exact_np, x_data_np, u_data_np = problem.generate_data()
    x_colloc_np = np.linspace(0.01, 0.99, 50).reshape(-1, 1)

    torch.manual_seed(42)
    model = NeuralNetwork([1, 32, 32, 32, 1])

    x_data_t = torch.tensor(x_data_np, dtype=torch.float32)
    u_data_t = torch.tensor(u_data_np, dtype=torch.float32)
    x_domain_t = torch.tensor(x_domain_np, dtype=torch.float32)
    x_colloc_t = torch.tensor(x_colloc_np, dtype=torch.float32)
    x_bc_t = torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    freq = problem.freq
    source_fn = lambda x: (freq * np.pi)**2 * torch.sin(freq * np.pi * x)

    # Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=1e-3)
    for step in tqdm(range(num_adam_steps),
                     desc=f'Spectral Adam (freq={source_freq})'):
        model.train()
        optimizer_adam.zero_grad()
        loss, _, _, _ = pinn_loss(model, x_data_t, u_data_t, x_colloc_t,
                                  x_bc_t, source_fn, 1.0, 10.0)
        loss.backward()
        optimizer_adam.step()

    # L-BFGS
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.5,
                                   max_iter=20, history_size=50,
                                   tolerance_grad=1e-9, tolerance_change=1e-11)
    for step in tqdm(range(num_lbfgs_steps),
                     desc=f'Spectral L-BFGS (freq={source_freq})'):
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _, _, _ = pinn_loss(model, x_data_t, u_data_t, x_colloc_t,
                                      x_bc_t, source_fn, 1.0, 10.0)
            loss.backward()
            return loss
        optimizer_lbfgs.step(closure)

    model.eval()
    with torch.no_grad():
        u_pred = model(x_domain_t).numpy()

    l2_err = np.sqrt(np.mean((u_pred.ravel() - u_exact_np.ravel())**2))
    return model, x_domain_np, u_exact_np, u_pred, x_data_np, u_data_np, l2_err


def plot_spectral_bias(save_path):
    model_f2, x_f2, ue_f2, up_f2, xd_f2, ud_f2, l2_f2 = \
        train_pinn_for_spectral_bias(source_freq=2.0)
    model_f10, x_f10, ue_f10, up_f10, xd_f10, ud_f10, l2_f10 = \
        train_pinn_for_spectral_bias(source_freq=10.0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.75))

    # Left: freq=2 (success)
    axes[0].plot(x_f2, ue_f2, color='#333333', linestyle='--',
                 linewidth=2, label='Exact', zorder=5)
    axes[0].plot(x_f2, up_f2, color='tab:blue', linewidth=2.5, label='PINN')
    axes[0].set_title(
        r'$u(x) = \sin(2\pi x)$' + f'\n$L_2$ error = {l2_f2:.2e}')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$u(x)$')
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([-1.3, 1.3])
    axes[0].legend()

    # Right: freq=10 (failure)
    axes[1].plot(x_f10, ue_f10, color='#333333', linestyle='--',
                 linewidth=2, label='Exact', zorder=5)
    axes[1].plot(x_f10, up_f10, color='tab:blue', linewidth=2.5, label='PINN')
    axes[1].set_title(
        r'$u(x) = \sin(10\pi x)$' + f'\n$L_2$ error = {l2_f10:.2e}')
    axes[1].set_xlabel(r'$x$')
    axes[1].set_ylabel(r'$u(x)$')
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([-1.3, 1.3])
    axes[1].legend()

    fig.savefig(save_path)
    print(f'Saved: {save_path}')
    plt.close(fig)

    print(f'  Freq=2  L2 error: {l2_f2:.2e}')
    print(f'  Freq=10 L2 error: {l2_f10:.2e}')


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    problem = PoissonProblem(source_freq=2.0)

    # Train both models
    (model_nn, x_domain_t, x_domain_np, u_exact_np,
     x_data_np, u_data_np) = train_standard_nn(problem)

    torch.manual_seed(42)  # reset seed for PINN
    (model_pinn, _, _, _, _, _, x_colloc_np) = train_pinn(problem)

    # ── Figure 1: Standard NN result (with training data) ────────────────
    save_result_figure(
        model_nn, x_domain_t, x_domain_np, u_exact_np,
        filename=os.path.join(FIGS_DIR, 'poisson-result-nn.pdf'),
        title='Standard NN: data only',
        x_data_np=x_data_np, u_data_np=u_data_np)

    # ── Figure 2: PINN result (NO training data; show collocation) ───────
    save_result_figure(
        model_pinn, x_domain_t, x_domain_np, u_exact_np,
        filename=os.path.join(FIGS_DIR, 'poisson-result-pinn.pdf'),
        title='',
        x_colloc_np=x_colloc_np,
        pred_label='PINN')

    # ── Figure 3: Extrapolation ──────────────────────────────────────────
    plot_extrapolation(
        model_nn, model_pinn, problem,
        save_path=os.path.join(FIGS_DIR, 'poisson-extrapolation.pdf'))

    # ── Figure 4: Spectral bias ──────────────────────────────────────────
    plot_spectral_bias(
        save_path=os.path.join(FIGS_DIR, 'poisson-spectral-bias.pdf'))

    print('\nAll figures saved to:', FIGS_DIR)
