"""
Generate the oscillator extrapolation figure for Chapter 6.

Trains a PINN on t in [0,1] for the damped harmonic oscillator:
    m*u'' + c*u' + k*u = 0,  u(0)=0, u'(0)=2*pi
with m=1, c=0.5, k=(2*pi)^2.

Evaluates on [0,2] to show extrapolation failure beyond the training domain.

Saves figure to:
    sciml-book/chapters/06-pinns/figs/oscillator-extrapolation.png
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

# ── Physical parameters ──────────────────────────────────────────────
m = 1.0
c = 0.5
k = (2 * np.pi) ** 2
w0_sq = k / m          # (2*pi)^2
mu = c / m              # 0.5  (damping ratio coefficient: u'' + mu*u' + w0^2*u = 0)
d = mu / 2              # 0.25 (decay rate)
wd = np.sqrt(w0_sq - d**2)  # damped frequency


def exact_solution(t):
    """u(t) = (2*pi / wd) * exp(-d*t) * sin(wd*t)."""
    return (2 * np.pi / wd) * np.exp(-d * t) * np.sin(wd * t)


def exact_derivative(t):
    """u'(t) via product rule."""
    A = 2 * np.pi / wd
    return A * np.exp(-d * t) * (-d * np.sin(wd * t) + wd * np.cos(wd * t))


# Quick sanity checks
assert abs(exact_solution(0.0)) < 1e-12, "u(0) should be 0"
assert abs(exact_derivative(0.0) - 2 * np.pi) < 1e-10, "u'(0) should be 2*pi"


# ── Neural network ───────────────────────────────────────────────────
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 hidden layers, 32 neurons, tanh
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, t):
        return self.net(t)


# ── Loss function ────────────────────────────────────────────────────
def pinn_loss(model, t_ic, u_ic, du_ic, t_phys, w_phys=1.0):
    """
    Combined loss:
      - IC on u(0) and u'(0)
      - Physics residual: u'' + mu*u' + w0^2*u = 0
    """
    # ── Initial conditions ──
    t_ic_g = t_ic.clone().requires_grad_(True)
    u_pred_ic = model(t_ic_g)
    du_pred_ic = torch.autograd.grad(
        u_pred_ic, t_ic_g,
        grad_outputs=torch.ones_like(u_pred_ic),
        create_graph=True,
    )[0]
    loss_ic_u = torch.mean((u_pred_ic - u_ic) ** 2)
    loss_ic_du = torch.mean((du_pred_ic - du_ic) ** 2)

    # ── Physics residual ──
    t_p = t_phys.clone().requires_grad_(True)
    u_p = model(t_p)
    du_p = torch.autograd.grad(
        u_p, t_p,
        grad_outputs=torch.ones_like(u_p),
        create_graph=True,
    )[0]
    d2u_p = torch.autograd.grad(
        du_p, t_p,
        grad_outputs=torch.ones_like(du_p),
        create_graph=True,
    )[0]
    residual = d2u_p + mu * du_p + w0_sq * u_p
    loss_phys = torch.mean(residual ** 2)

    return loss_ic_u + loss_ic_du + w_phys * loss_phys


# ── Training data ────────────────────────────────────────────────────
t_train_max = 1.0

# IC point
t_ic = torch.tensor([[0.0]], dtype=torch.float32)
u_ic = torch.tensor([[0.0]], dtype=torch.float32)
du_ic = torch.tensor([[2 * np.pi]], dtype=torch.float32)

# Collocation points in [0, 1]
N_phys = 50
t_phys = torch.linspace(0, t_train_max, N_phys).reshape(-1, 1).requires_grad_(False)


# ── Train ────────────────────────────────────────────────────────────
model = PINN()

# Phase 1: Adam
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
n_adam = 10000

print("Phase 1: Adam optimisation")
for step in range(n_adam):
    optimizer_adam.zero_grad()
    loss = pinn_loss(model, t_ic, u_ic, du_ic, t_phys)
    loss.backward()
    optimizer_adam.step()
    if (step + 1) % 2000 == 0:
        print(f"  step {step+1:>5d}  loss = {loss.item():.4e}")

# Phase 2: L-BFGS for fine-tuning
print("Phase 2: L-BFGS optimisation")
optimizer_lbfgs = torch.optim.LBFGS(
    model.parameters(), lr=1.0, max_iter=20,
    history_size=50, line_search_fn="strong_wolfe",
)

for step in range(200):
    def closure():
        optimizer_lbfgs.zero_grad()
        loss = pinn_loss(model, t_ic, u_ic, du_ic, t_phys)
        loss.backward()
        return loss

    loss = optimizer_lbfgs.step(closure)
    if (step + 1) % 50 == 0:
        print(f"  step {step+1:>3d}  loss = {loss.item():.4e}")


# ── Evaluate on [0, 2] ──────────────────────────────────────────────
model.eval()
t_eval = np.linspace(0, 2.0, 1000).reshape(-1, 1)
u_exact = exact_solution(t_eval).ravel()

t_eval_tensor = torch.tensor(t_eval, dtype=torch.float32)
with torch.no_grad():
    u_pred = model(t_eval_tensor).numpy().ravel()

t_plot = t_eval.ravel()


# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))

# Training domain shading
ax.axvspan(0, t_train_max, alpha=0.10, color='gray', zorder=0)

# Vertical boundary line
ax.axvline(t_train_max, color='gray', linestyle='--', linewidth=1.0, zorder=1)

# PINN prediction (underneath)
ax.plot(t_plot, u_pred, color='tab:blue', linewidth=2.5, label='PINN prediction', zorder=3)

# Exact solution (on top)
ax.plot(t_plot, u_exact, color='#333333', linestyle='--', linewidth=2,
        label='Exact solution', zorder=5)

# Labels
ax.set_xlabel(r'$t$', fontsize=12)
ax.set_ylabel(r'$u(t)$', fontsize=12)
ax.set_xlim([0, 2.0])

# Annotate regions
y_top = ax.get_ylim()[1]
ax.text(0.5, y_top * 0.88, 'Training domain', ha='center', fontsize=9, color='0.35')
ax.text(1.5, y_top * 0.88, 'Extrapolation', ha='center', fontsize=9, color='0.35')

ax.legend(loc='lower left', fontsize=9, frameon=True, facecolor='white', edgecolor='0.7')
ax.tick_params(labelsize=10)

fig.tight_layout()

save_path = '/Users/krishna/courses/CE397-Scientific-MachineLearning/book/sciml-book/chapters/06-pinns/figs/oscillator-extrapolation.png'
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {save_path}")
plt.close(fig)
