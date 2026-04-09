#!/usr/bin/env python3
"""Build the mixed-signal Poisson PINN notebook and chapter figures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import torch
import torch.nn as nn


STYLE_PATH = ROOT / "sciml_notebook" / "docs" / "sciml_style.mplstyle"
plt.style.use(STYLE_PATH)
torch.set_num_threads(1)

NOTEBOOK_PATH = Path(__file__).with_name("ntk-spectral-bias.ipynb")
PINN_MD_PATH = Path(__file__).with_name("pinns.md")
LOCAL_FIG_DIR = Path(__file__).parent
CHAPTER_FIG_DIR = ROOT / "sciml-book" / "chapters" / "06-pinns" / "figs"
PINN_FIG_BASENAME = "mixed-frequency-poisson"
AMPLITUDE_FIG_BASENAME = "mixed-frequency-amplitudes"
NTK_SPECTRA_FIG_BASENAME = "mixed-frequency-ntk-spectra"
NTK_BLOCKS_FIG_BASENAME = "mixed-frequency-ntk-blocks"
NTK_DRIFT_FIG_BASENAME = "mixed-frequency-ntk-drift"


@dataclass(frozen=True)
class MixedPoissonProblem:
    """Poisson problem whose exact solution is a mixed signal."""

    low_amp: float = 1.0
    low_freq: int = 1
    high_amp: float = 0.5
    high_freq: int = 3
    xmin: float = 0.0
    xmax: float = 1.0

    def exact(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return self.low_amp * np.sin(self.low_freq * np.pi * x) + self.high_amp * np.sin(
            self.high_freq * np.pi * x
        )

    def low_component(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return self.low_amp * np.sin(self.low_freq * np.pi * x)

    def high_component(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        return self.high_amp * np.sin(self.high_freq * np.pi * x)

    def forcing(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        low = self.low_amp * (self.low_freq * np.pi) ** 2 * np.sin(self.low_freq * np.pi * x)
        high = self.high_amp * (self.high_freq * np.pi) ** 2 * np.sin(
            self.high_freq * np.pi * x
        )
        return low + high


class HardBCMLP(nn.Module):
    """MLP with an output transform that enforces Dirichlet data exactly."""

    def __init__(self, xmin: float, xmax: float, width: int = 64, depth: int = 3):
        super().__init__()
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        layers: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = (x - self.xmin) / (self.xmax - self.xmin)
        return s * (1.0 - s) * self.net(s)


def project_amplitude(u: np.ndarray, x: np.ndarray, freq: int) -> float:
    basis = np.sin(freq * np.pi * x)
    num = np.trapezoid(u * basis, x)
    den = np.trapezoid(basis * basis, x)
    return float(num / den)


def evaluate_residual(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    xg = x.clone().detach().requires_grad_(True)
    u = model(xg)
    du = torch.autograd.grad(u, xg, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, xg, torch.ones_like(du), create_graph=True)[0]
    return -d2u


def evaluate_solution(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def evaluate_training_state(
    model: nn.Module,
    problem: MixedPoissonProblem,
    x_collocation: torch.Tensor,
    f_collocation: torch.Tensor,
    x_test_tensor: torch.Tensor,
    x_test: np.ndarray,
    u_exact: np.ndarray,
) -> dict[str, float]:
    residual = evaluate_residual(model, x_collocation)
    loss = (residual - f_collocation).pow(2).mean()
    with torch.no_grad():
        pred = model(x_test_tensor).cpu().numpy().ravel()
    a_low = project_amplitude(pred, x_test, problem.low_freq)
    a_high = project_amplitude(pred, x_test, problem.high_freq)
    return {
        "loss": float(loss.item()),
        "l2": float(np.sqrt(np.mean((pred - u_exact) ** 2))),
        "a_low": a_low,
        "a_high": a_high,
        "low_ratio": a_low / problem.low_amp,
        "high_ratio": a_high / problem.high_amp,
    }


def train_mixed_poisson(
    problem: MixedPoissonProblem,
    *,
    num_steps: int = 1000,
    lr: float = 5e-4,
    n_collocation: int = 160,
    width: int = 64,
    depth: int = 3,
    seed: int = 42,
    log_every: int = 10,
    verbose: bool = False,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = HardBCMLP(problem.xmin, problem.xmax, width=width, depth=depth)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    x_collocation = torch.linspace(
        problem.xmin + 1e-3, problem.xmax - 1e-3, n_collocation
    ).reshape(-1, 1)
    f_collocation = torch.tensor(problem.forcing(x_collocation.numpy()), dtype=torch.float32)
    x_test = np.linspace(problem.xmin, problem.xmax, 600)
    x_test_tensor = torch.tensor(x_test.reshape(-1, 1), dtype=torch.float32)
    u_exact = problem.exact(x_test)

    history: list[dict[str, float]] = []
    snapshots: dict[int, np.ndarray] = {}
    state_snapshots: dict[int, dict[str, torch.Tensor]] = {}
    snapshot_steps = {0, 100, 500, 1000}

    for step in range(num_steps + 1):
        if step % log_every == 0 or step in snapshot_steps:
            row = evaluate_training_state(
                model,
                problem,
                x_collocation,
                f_collocation,
                x_test_tensor,
                x_test,
                u_exact,
            )
            row["step"] = float(step)
            history.append(row)
            snapshots[step] = model(x_test_tensor).detach().cpu().numpy().ravel()
            if step in snapshot_steps:
                state_snapshots[step] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            if verbose:
                print(
                    f"step={step:5d} loss={row['loss']:.3e} l2={row['l2']:.3e} "
                    f"low={row['low_ratio']:.3f} high={row['high_ratio']:.3f}",
                    flush=True,
                )

        if step == num_steps:
            break

        optimizer.zero_grad()
        residual = evaluate_residual(model, x_collocation)
        loss = (residual - f_collocation).pow(2).mean()
        loss.backward()
        optimizer.step()

    return {
        "model": model,
        "history": history,
        "snapshots": snapshots,
        "state_snapshots": state_snapshots,
        "x_test": x_test,
        "u_exact": u_exact,
        "problem": problem,
        "width": width,
        "depth": depth,
    }


def state_vector(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in state.values()])


def jacobian_matrix(model: nn.Module, x: torch.Tensor, operator: str) -> np.ndarray:
    params = list(model.parameters())
    rows: list[torch.Tensor] = []
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        xi = x[i : i + 1].clone().detach().requires_grad_(True)
        if operator == "solution":
            scalar = evaluate_solution(model, xi).reshape(())
        elif operator == "residual":
            scalar = evaluate_residual(model, xi).reshape(())
        else:
            raise ValueError(f"unknown operator: {operator}")
        scalar.backward()
        grads: list[torch.Tensor] = []
        for param in params:
            if param.grad is None:
                grads.append(torch.zeros_like(param).reshape(-1))
            else:
                grads.append(param.grad.reshape(-1))
        rows.append(torch.cat(grads).detach().cpu())
    return torch.stack(rows, dim=0).numpy()


def analyze_ntk(
    result: dict[str, object],
    *,
    ntk_points: int = 40,
) -> dict[str, object]:
    problem: MixedPoissonProblem = result["problem"]  # type: ignore[assignment]
    width = int(result["width"])
    depth = int(result["depth"])
    state_snapshots: dict[int, dict[str, torch.Tensor]] = result["state_snapshots"]  # type: ignore[assignment]
    snapshot_steps = sorted(state_snapshots)
    x_ntk = torch.linspace(problem.xmin + 1e-3, problem.xmax - 1e-3, ntk_points).reshape(-1, 1)

    blocks: dict[int, dict[str, np.ndarray]] = {}
    spectra: dict[int, dict[str, np.ndarray]] = {}
    ntk_change: dict[int, float] = {}
    weight_change: dict[int, float] = {}
    theta0 = state_vector(state_snapshots[snapshot_steps[0]])
    K0: np.ndarray | None = None

    for step in snapshot_steps:
        model = HardBCMLP(problem.xmin, problem.xmax, width=width, depth=depth)
        model.load_state_dict(state_snapshots[step])
        J_u = jacobian_matrix(model, x_ntk, "solution")
        J_r = jacobian_matrix(model, x_ntk, "residual")
        K_uu = J_u @ J_u.T
        K_ur = J_u @ J_r.T
        K_rr = J_r @ J_r.T
        K = np.block([[K_uu, K_ur], [K_ur.T, K_rr]])
        blocks[step] = {"K_uu": K_uu, "K_ur": K_ur, "K_rr": K_rr, "K": K}
        spectra[step] = {
            "K": np.sort(np.linalg.eigvalsh(K))[::-1],
            "K_uu": np.sort(np.linalg.eigvalsh(K_uu))[::-1],
            "K_rr": np.sort(np.linalg.eigvalsh(K_rr))[::-1],
        }
        if K0 is None:
            K0 = K
        ntk_change[step] = float(np.linalg.norm(K - K0) / np.linalg.norm(K0))
        theta = state_vector(state_snapshots[step])
        weight_change[step] = float(torch.linalg.norm(theta - theta0) / torch.linalg.norm(theta0))

    return {
        "steps": snapshot_steps,
        "x_ntk": x_ntk.cpu().numpy().ravel(),
        "blocks": blocks,
        "spectra": spectra,
        "ntk_change": ntk_change,
        "weight_change": weight_change,
    }


def plot_solution_snapshots(
    result: dict[str, object],
    panel_steps: tuple[int, int, int] = (100, 500, 1000),
) -> plt.Figure:
    problem: MixedPoissonProblem = result["problem"]  # type: ignore[assignment]
    snapshots: dict[int, np.ndarray] = result["snapshots"]  # type: ignore[assignment]
    x: np.ndarray = result["x_test"]  # type: ignore[assignment]
    u_exact: np.ndarray = result["u_exact"]  # type: ignore[assignment]
    low = problem.low_component(x)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    for ax, step in zip(axes, panel_steps):
        pred = snapshots[step]
        ax.plot(
            x,
            u_exact,
            color="tab:green",
            lw=1.9,
            label=rf"$u(x)=\sin(\pi x)+0.5\sin({problem.high_freq}\pi x)$",
        )
        ax.plot(x, low, color="black", ls="--", lw=1.4, label=r"$\sin(\pi x)$")
        ax.plot(
            x,
            pred,
            color="tab:red",
            lw=1.4,
            marker="o",
            ms=2.0,
            markevery=18,
            label="PINN",
        )
        ax.set_title(f"Step {step}")
        ax.set_xlabel("$x$")
        ax.set_xlim(problem.xmin, problem.xmax)

    axes[0].set_ylabel("$u(x)$")
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles[:3], labels[:3], fontsize=8, loc="upper right")
    return fig


def plot_amplitude_history(result: dict[str, object]) -> plt.Figure:
    history: list[dict[str, float]] = result["history"]  # type: ignore[assignment]
    steps = [row["step"] for row in history]
    low = [row["low_ratio"] for row in history]
    high = [row["high_ratio"] for row in history]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(steps, low, lw=1.8, label="Low-mode amplitude")
    ax.plot(steps, high, lw=1.8, label="High-mode amplitude")
    ax.axhline(1.0, color="black", ls="--", lw=1.0)
    ax.set_xlabel("Gradient descent steps")
    ax.set_ylabel("Recovered amplitude / exact amplitude")
    ax.set_ylim(0.0, max(2.2, max(low) * 1.05))
    ax.legend(loc="upper right")
    return fig


def plot_ntk_spectra(analysis: dict[str, object]) -> plt.Figure:
    steps: list[int] = analysis["steps"]  # type: ignore[assignment]
    spectra: dict[int, dict[str, np.ndarray]] = analysis["spectra"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2), constrained_layout=True)

    for step in steps:
        axes[0].semilogy(spectra[step]["K"], lw=1.2, label=f"Step {step}")
        axes[1].semilogy(spectra[step]["K_uu"], lw=1.2, label=f"Step {step}")
        axes[2].semilogy(spectra[step]["K_rr"], lw=1.2, label=f"Step {step}")

    axes[0].set_title(r"Full block kernel $K$")
    axes[1].set_title(r"Solution block $K_{uu}$")
    axes[2].set_title(r"Residual block $K_{rr}$")
    for ax in axes:
        ax.set_xlabel("Mode index")
        ax.set_ylabel("Eigenvalue")
    axes[0].legend(loc="upper right", fontsize=8)
    return fig


def plot_ntk_blocks(analysis: dict[str, object], step: int = 100) -> plt.Figure:
    blocks: dict[int, dict[str, np.ndarray]] = analysis["blocks"]  # type: ignore[assignment]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.0), constrained_layout=True)
    labels = [("K_uu", r"$K_{uu}$"), ("K_ur", r"$K_{ur}$"), ("K_rr", r"$K_{rr}$")]
    for ax, (key, title) in zip(axes, labels):
        image = ax.imshow(blocks[step][key], cmap="magma", aspect="auto")
        ax.set_title(f"{title} at step {step}")
        ax.set_xlabel("Point index")
        ax.set_ylabel("Point index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig


def plot_ntk_drift(analysis: dict[str, object]) -> plt.Figure:
    steps: list[int] = analysis["steps"]  # type: ignore[assignment]
    ntk_change: dict[int, float] = analysis["ntk_change"]  # type: ignore[assignment]
    weight_change: dict[int, float] = analysis["weight_change"]  # type: ignore[assignment]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.1), constrained_layout=True)
    axes[0].plot(steps, [ntk_change[step] for step in steps], lw=1.8)
    axes[0].set_title(r"$\|K(t)-K(0)\|/\|K(0)\|$")
    axes[0].set_xlabel("Gradient descent steps")
    axes[0].set_ylabel("Relative NTK change")

    axes[1].plot(steps, [weight_change[step] for step in steps], lw=1.8)
    axes[1].set_title(r"$\|\theta(t)-\theta(0)\|/\|\theta(0)\|$")
    axes[1].set_xlabel("Gradient descent steps")
    axes[1].set_ylabel("Relative parameter change")
    return fig


def save_figure(fig: plt.Figure, basename: str, *, copy_to_chapter: bool = False) -> Path:
    LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHAPTER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = LOCAL_FIG_DIR / f"{basename}.png"
    pdf = LOCAL_FIG_DIR / f"{basename}.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if copy_to_chapter:
        fig.savefig(CHAPTER_FIG_DIR / f"{basename}.png", dpi=220, bbox_inches="tight")
        fig.savefig(CHAPTER_FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


def history_lookup(result: dict[str, object], step: int) -> dict[str, float]:
    history: list[dict[str, float]] = result["history"]  # type: ignore[assignment]
    return next(row for row in history if int(row["step"]) == step)


def build_notebook(result: dict[str, object], analysis: dict[str, object]) -> nbf.NotebookNode:
    step_100 = history_lookup(result, 100)
    step_500 = history_lookup(result, 500)
    step_1000 = history_lookup(result, 1000)

    cells: list[nbf.NotebookNode] = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# NTK Analysis and Spectral Bias in PINNs\n\n"
            "This notebook studies one Poisson problem from beginning to end.\n"
            "The exact solution is\n"
            "$u(x)=\\sin(\\pi x)+0.5\\sin(3\\pi x)$ on $[0,1]$.\n"
            "The boundary values vanish at both endpoints.\n"
            "Differentiating twice gives\n"
            "$-u''(x)=\\pi^2\\sin(\\pi x)+4.5\\pi^2\\sin(3\\pi x)$.\n"
            "The signal has one broad component and one oscillatory correction.\n"
            "That makes it a clean test for spectral bias in a true PINN solve."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. A mixed signal as a Poisson problem\n\n"
            "The figure below shows three snapshots from full-batch gradient descent.\n"
            "At step 100, the low mode already has order-one amplitude while the high mode is still small.\n"
            f"The recovered high-mode amplitude is only {step_100['high_ratio']:.2f} of its exact value.\n"
            f"By step 500 it reaches {step_500['high_ratio']:.2f}.\n"
            f"By step 1000 both modes are essentially correct.\n\n"
            f"![Mixed-signal Poisson PINN]({PINN_FIG_BASENAME}.png)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "from rewrite_ntk_spectral_bias import (\n"
            "    MixedPoissonProblem,\n"
            "    analyze_ntk,\n"
            "    plot_amplitude_history,\n"
            "    plot_ntk_blocks,\n"
            "    plot_ntk_drift,\n"
            "    plot_ntk_spectra,\n"
            "    plot_solution_snapshots,\n"
            "    train_mixed_poisson,\n"
            ")\n"
            "\n"
            "problem = MixedPoissonProblem()\n"
            "result = train_mixed_poisson(problem, verbose=False)\n"
            "analysis = analyze_ntk(result)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Amplitude history\n\n"
            "A direct way to see spectral bias is to track the Fourier amplitudes during training.\n"
            "The low mode reaches order one first.\n"
            "The high mode lags behind.\n\n"
            f"![Amplitude history]({AMPLITUDE_FIG_BASENAME}.png)"
        )
    )

    cells.append(nbf.v4.new_code_cell("plot_amplitude_history(result);"))

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. NTK intuition\n\n"
            "At the training points, a parameter update changes the solution values through the Jacobian $J_u$.\n"
            "The residual values have their own Jacobian $J_r$.\n"
            "These produce the block kernels\n"
            "$K_{uu}=J_uJ_u^\\top$,\n"
            "$K_{ur}=J_uJ_r^\\top$,\n"
            "and $K_{rr}=J_rJ_r^\\top$.\n"
            "The spectra below show how the kernel acts on the solution side and on the residual side.\n\n"
            f"![NTK spectra]({NTK_SPECTRA_FIG_BASENAME}.png)\n\n"
            f"![NTK blocks]({NTK_BLOCKS_FIG_BASENAME}.png)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "plot_ntk_spectra(analysis)\n"
            "plot_ntk_blocks(analysis, step=100);"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. How far does the kernel move\n\n"
            "The Poisson notebook in `PINNsNTK` also tracks how far the kernel and the parameters move from initialization.\n"
            "That tells us whether the frozen-kernel picture is plausible over the time window we are studying.\n\n"
            f"![NTK drift]({NTK_DRIFT_FIG_BASENAME}.png)"
        )
    )

    cells.append(nbf.v4.new_code_cell("plot_ntk_drift(analysis);"))

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. Reproduce the main figure\n\n"
            "The call below redraws the three solution snapshots used in the chapter.\n"
        )
    )

    cells.append(nbf.v4.new_code_cell("plot_solution_snapshots(result);"))

    return nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )


def update_pinns_md() -> None:
    text = PINN_MD_PATH.read_text()
    replacements = {
        "### NTK, spectral bias, and mixed-signal intuition\n- [Mixed-Signal Regression and PINN](ntk-spectral-bias.ipynb)\n": (
            "### NTK and spectral bias in PINNs\n"
            "- [Mixed-Signal Poisson PINN](ntk-spectral-bias.ipynb)\n"
        ),
        "### NTK, spectral bias, and mixed-frequency Poisson\n- [NTK and Spectral Bias](ntk-spectral-bias.ipynb)\n": (
            "### NTK and spectral bias in PINNs\n"
            "- [Mixed-Signal Poisson PINN](ntk-spectral-bias.ipynb)\n"
        ),
        "### NTK, spectral bias, and mixed-frequency Poisson\n- [Mixed-Frequency Poisson](ntk-spectral-bias.ipynb)\n": (
            "### NTK and spectral bias in PINNs\n"
            "- [Mixed-Signal Poisson PINN](ntk-spectral-bias.ipynb)\n"
        ),
    }
    for old, new in replacements.items():
        if old in text:
            text = text.replace(old, new)
    PINN_MD_PATH.write_text(text)


def main() -> None:
    problem = MixedPoissonProblem()
    print("training mixed-signal Poisson PINN", flush=True)
    result = train_mixed_poisson(problem, verbose=True)
    print("analyzing block NTKs", flush=True)
    analysis = analyze_ntk(result)

    print("saving figures", flush=True)
    save_figure(plot_solution_snapshots(result), PINN_FIG_BASENAME, copy_to_chapter=True)
    save_figure(plot_amplitude_history(result), AMPLITUDE_FIG_BASENAME)
    save_figure(plot_ntk_spectra(analysis), NTK_SPECTRA_FIG_BASENAME, copy_to_chapter=True)
    save_figure(plot_ntk_blocks(analysis, step=100), NTK_BLOCKS_FIG_BASENAME)
    save_figure(plot_ntk_drift(analysis), NTK_DRIFT_FIG_BASENAME)

    print("writing notebook", flush=True)
    NOTEBOOK_PATH.write_text(nbf.writes(build_notebook(result, analysis)))
    print("updating notebook index", flush=True)
    update_pinns_md()

    step_100 = history_lookup(result, 100)
    step_1000 = history_lookup(result, 1000)
    print(
        "step 100 ratios",
        f"low={step_100['low_ratio']:.3f}",
        f"high={step_100['high_ratio']:.3f}",
        f"l2={step_100['l2']:.4f}",
    )
    print(
        "step 1000 ratios",
        f"low={step_1000['low_ratio']:.3f}",
        f"high={step_1000['high_ratio']:.3f}",
        f"l2={step_1000['l2']:.4f}",
    )
    print(f"wrote notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
