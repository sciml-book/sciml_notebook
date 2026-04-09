"""
Generate oscillator figures for Chapter 6 (PINNs).

Produces PDF figures using the SciML book style:
  - harmonic-oscillator.pdf      (analytical displacement curve)
  - oscillator-result-nn.pdf     (standard NN prediction vs exact)
  - oscillator-result-pinn.pdf   (PINN prediction vs exact + collocation)
  - oscillator-extrapolation.pdf (interpolation vs extrapolation comparison)

Usage:
    python generate_oscillator_figs.py
"""

import sys
from pathlib import Path

# Add parent directory so we can import sciml_plots
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from sciml_plots import setup_style, plot_exact, plot_prediction, plot_data, savefig

# ── Setup ────────────────────────────────────────────────────────────────────
setup_style()

SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "sciml-book" / "chapters" / "06-pinns" / "figs"

torch.manual_seed(42)
np.random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
OVERALL_T_MIN = 0.0
OVERALL_T_MAX = 1.0
NUM_OVERALL_POINTS = 500

DATA_T_MIN = 0.0
DATA_T_MAX = 0.3607
NUM_DATA_POINTS = 10

PHYSICS_T_MIN = 0.0
PHYSICS_T_MAX = 1.0
NUM_PHYSICS_POINTS = 30

DAMPING_COEFFICIENT = 2
NATURAL_FREQUENCY = 20


# ── Physics ──────────────────────────────────────────────────────────────────
class HarmonicOscillator:
    def __init__(self, d, w0):
        self.d = d
        self.w0 = w0
        assert d < w0
        self.w = np.sqrt(w0**2 - d**2)
        self.phi = np.arctan(-d / self.w)
        self.A = 1 / (2 * np.cos(self.phi))

    def solution(self, t):
        return np.exp(-self.d * t) * 2 * self.A * np.cos(self.phi + self.w * t)

    def derivative(self, t):
        u = self.solution(t)
        return -self.d * u - np.exp(-self.d * t) * 2 * self.A * self.w * np.sin(self.phi + self.w * t)


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


def pinn_loss(model, t_data, u_data, t_physics, mu, k):
    """Combined data + physics loss."""
    # Data loss
    if t_data.numel() > 0:
        loss_data = nn.MSELoss()(model(t_data), u_data)
    else:
        loss_data = torch.tensor(0.0)

    # Physics loss
    if t_physics.numel() > 0:
        t_p = t_physics.clone().detach().requires_grad_(True)
        u_p = model(t_p)
        du = torch.autograd.grad(u_p, t_p, torch.ones_like(u_p), create_graph=True)[0]
        d2u = torch.autograd.grad(du, t_p, torch.ones_like(du), create_graph=True)[0]
        residual = d2u + mu * du + k * u_p
        loss_physics = 1e-4 * torch.mean(residual**2)
    else:
        loss_physics = torch.tensor(0.0)

    return loss_data + loss_physics


def train(ho, t_domain_np, t_data_np, u_data_np, t_physics_np=None, num_steps=25000):
    """Train a standard NN (t_physics_np=None) or PINN."""
    layer_sizes = [1, 32, 32, 32, 1]
    model = NeuralNetwork(layer_sizes)

    t_data_t = torch.tensor(t_data_np, dtype=torch.float32)
    u_data_t = torch.tensor(u_data_np, dtype=torch.float32)
    t_domain_t = torch.tensor(t_domain_np, dtype=torch.float32)

    is_pinn = t_physics_np is not None
    if is_pinn:
        t_physics_t = torch.tensor(t_physics_np, dtype=torch.float32)
        mu_t = torch.tensor(2 * ho.d, dtype=torch.float32)
        k_t = torch.tensor(ho.w0**2, dtype=torch.float32)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        label = "PINN"
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        label = "Standard NN"

    criterion = nn.MSELoss()

    for step in tqdm(range(num_steps), desc=f"Training {label}"):
        model.train()
        optimizer.zero_grad()
        if is_pinn:
            loss = pinn_loss(model, t_data_t, u_data_t, t_physics_t, mu_t, k_t)
        else:
            loss = criterion(model(t_data_t), u_data_t)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, t_domain_t


# ── Generate data ────────────────────────────────────────────────────────────
ho = HarmonicOscillator(DAMPING_COEFFICIENT, NATURAL_FREQUENCY)

t_domain = np.linspace(OVERALL_T_MIN, OVERALL_T_MAX, NUM_OVERALL_POINTS).reshape(-1, 1)
u_exact = ho.solution(t_domain).reshape(-1, 1)

t_data = np.linspace(DATA_T_MIN, DATA_T_MAX, NUM_DATA_POINTS).reshape(-1, 1)
u_data = ho.solution(t_data).reshape(-1, 1)

t_physics = np.linspace(PHYSICS_T_MIN, PHYSICS_T_MAX, NUM_PHYSICS_POINTS).reshape(-1, 1)


# ── Figure 1: Harmonic oscillator displacement ──────────────────────────────
print("Generating harmonic-oscillator.pdf ...")
fig, ax = plt.subplots()
ax.plot(t_domain.ravel(), u_exact.ravel(), color='#333333', linewidth=2.0)
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Displacement $u(t)$")
ax.set_xlim([OVERALL_T_MIN, OVERALL_T_MAX])
savefig(fig, "harmonic-oscillator", path=SAVE_DIR)
plt.close(fig)


# ── Train models ─────────────────────────────────────────────────────────────
print("\n--- Training Standard NN ---")
model_nn, t_tensor = train(ho, t_domain, t_data, u_data, t_physics_np=None)

print("\n--- Training PINN ---")
model_pinn, _ = train(ho, t_domain, t_data, u_data, t_physics_np=t_physics)


# ── Figure 2: Standard NN result ────────────────────────────────────────────
print("\nGenerating oscillator-result-nn.pdf ...")
with torch.no_grad():
    u_pred_nn = model_nn(t_tensor).numpy()

fig, ax = plt.subplots()
plot_prediction(ax, t_domain.ravel(), u_pred_nn.ravel(), label=r"NN prediction")
plot_exact(ax, t_domain.ravel(), u_exact.ravel(), label=r"Exact solution")
plot_data(ax, t_data.ravel(), u_data.ravel(), label=r"Training data")
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Displacement $u(t)$")
ax.set_xlim([OVERALL_T_MIN, OVERALL_T_MAX])
ax.set_ylim([-1.2, 1.2])
ax.legend()
savefig(fig, "oscillator-result-nn", path=SAVE_DIR)
plt.close(fig)


# ── Figure 3: PINN result ───────────────────────────────────────────────────
print("Generating oscillator-result-pinn.pdf ...")
with torch.no_grad():
    u_pred_pinn = model_pinn(t_tensor).numpy()

fig, ax = plt.subplots()
plot_prediction(ax, t_domain.ravel(), u_pred_pinn.ravel(), label=r"PINN prediction")
plot_exact(ax, t_domain.ravel(), u_exact.ravel(), label=r"Exact solution")
plot_data(ax, t_data.ravel(), u_data.ravel(), label=r"Training data")
ax.scatter(t_physics.ravel(), np.zeros(len(t_physics)),
           color='#B8860B', s=30, marker='^', alpha=0.8, zorder=4,
           label=r"Collocation points")
ax.set_xlabel(r"Time $t$")
ax.set_ylabel(r"Displacement $u(t)$")
ax.set_xlim([OVERALL_T_MIN, OVERALL_T_MAX])
ax.set_ylim([-1.2, 1.2])
ax.legend()
savefig(fig, "oscillator-result-pinn", path=SAVE_DIR)
plt.close(fig)


# ── Figure 4: Extrapolation comparison ──────────────────────────────────────
print("Generating oscillator-extrapolation.pdf ...")
t_train_max = 1.0
t_extrap_max = 2.0
num_ext = 1000

t_ext_np = np.linspace(0.0, t_extrap_max, num_ext).reshape(-1, 1)
t_ext_tensor = torch.tensor(t_ext_np, dtype=torch.float32)
u_exact_ext = ho.solution(t_ext_np).ravel()

with torch.no_grad():
    u_nn_ext = model_nn(t_ext_tensor).numpy().ravel()
    u_pinn_ext = model_pinn(t_ext_tensor).numpy().ravel()

t_plot = t_ext_np.ravel()

# L2 errors
mask_train = t_plot <= t_train_max
mask_extrap = t_plot > t_train_max
l2_nn_train = np.sqrt(np.mean((u_nn_ext[mask_train] - u_exact_ext[mask_train])**2))
l2_nn_extrap = np.sqrt(np.mean((u_nn_ext[mask_extrap] - u_exact_ext[mask_extrap])**2))
l2_pinn_train = np.sqrt(np.mean((u_pinn_ext[mask_train] - u_exact_ext[mask_train])**2))
l2_pinn_extrap = np.sqrt(np.mean((u_pinn_ext[mask_extrap] - u_exact_ext[mask_extrap])**2))

print(f"  L2 (train):  NN={l2_nn_train:.4e}  PINN={l2_pinn_train:.4e}")
print(f"  L2 (extrap): NN={l2_nn_extrap:.4e}  PINN={l2_pinn_extrap:.4e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.75), sharey=True)

for ax, u_pred, title, l2_tr, l2_ex in [
    (ax1, u_nn_ext, "Standard Neural Network", l2_nn_train, l2_nn_extrap),
    (ax2, u_pinn_ext, "Physics-Informed Neural Network", l2_pinn_train, l2_pinn_extrap),
]:
    # Shaded regions
    ax.axvspan(0, t_train_max, alpha=0.06, color='green')
    ax.axvspan(t_train_max, t_extrap_max, alpha=0.06, color='orange')
    ax.axvline(t_train_max, color='gray', linestyle=':', linewidth=1)

    # Curves
    plot_prediction(ax, t_plot, u_pred, label=r"NN prediction")
    plot_exact(ax, t_plot, u_exact_ext, label=r"Exact solution")
    plot_data(ax, t_data.ravel(), u_data.ravel(), label=r"Training data")

    ax.set_xlabel(r"Time $t$")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim([0, t_extrap_max])

    # Error annotation
    ax.text(0.5, -0.85,
            rf"$L_2$ train: {l2_tr:.2e}" + "\n" + rf"$L_2$ extrap: {l2_ex:.2e}",
            fontsize=9, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

ax1.set_ylabel(r"Displacement $u(t)$")

savefig(fig, "oscillator-extrapolation", path=SAVE_DIR)
plt.close(fig)

print(f"\nAll figures saved to {SAVE_DIR}")
