#!/usr/bin/env python3
"""Build an optimizer benchmark for the continuous-time Burgers PINN."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = ROOT / ".matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import scipy.io
from scipy.optimize import line_search
from scipy.stats import qmc
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


STYLE_PATH = ROOT / "sciml_notebook" / "sciml_style.mplstyle"
plt.style.use(STYLE_PATH)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

NOTEBOOK_PATH = Path(__file__).with_name("burgers-optimizers.ipynb")
PINN_MD_PATH = Path(__file__).with_name("pinns.md")
LOCAL_FIG_DIR = Path(__file__).parent
CHAPTER_FIG_DIR = ROOT / "sciml-book" / "chapters" / "06-pinns" / "figs"
HISTORY_FIG_BASENAME = "burgers-optimizer-history"
SLICE_FIG_BASENAME = "burgers-optimizer-slices"


@dataclass(frozen=True)
class BurgersConfig:
    """Configuration for the optimizer benchmark."""

    data_path: Path = ROOT / "sciml_notebook" / "pinns" / "discrete-time-burgers" / "burgers_shock.mat"
    nu: float = 0.01 / np.pi
    n_u: int = 100
    n_f: int = 2000
    warmup_steps: int = 400
    second_stage_steps: int = 80
    adam_lr: float = 1e-3
    lbfgs_memory: int = 20
    seed: int = 7


class BurgersPINN(nn.Module):
    """Continuous-time Burgers PINN with feature scaling."""

    def __init__(self, lb: np.ndarray, ub: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("lb", torch.tensor(lb.reshape(1, -1), dtype=torch.get_default_dtype()))
        self.register_buffer("ub", torch.tensor(ub.reshape(1, -1), dtype=torch.get_default_dtype()))
        self.layers = nn.ModuleList(
            [
                nn.Linear(2, 20),
                nn.Linear(20, 20),
                nn.Linear(20, 20),
                nn.Linear(20, 20),
                nn.Linear(20, 1),
            ]
        )
        self.activation = nn.Tanh()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        x = scaled
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def residual(self, inputs: torch.Tensor, nu: float) -> torch.Tensor:
        xt = inputs.clone().detach().requires_grad_(True)
        u = self.forward(xt)
        grads = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, :1]
        u_t = grads[:, 1:]
        u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x), create_graph=True)[0][:, :1]
        return u_t + u * u_x - nu * u_xx


class BurgersBenchmark:
    """Full-batch Burgers benchmark for optimizer comparison."""

    def __init__(self, config: BurgersConfig) -> None:
        self.config = config
        self._set_seed(config.seed)
        self._load_data()
        self.model = BurgersPINN(self.lb, self.ub)

    @staticmethod
    def _set_seed(seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _load_data(self) -> None:
        data = scipy.io.loadmat(self.config.data_path)
        x = data["x"].astype(np.float64)
        t = data["t"].astype(np.float64)
        usol = np.real(data["usol"]).astype(np.float64)
        X, T = np.meshgrid(x.squeeze(), t.squeeze())

        initial_xt = np.hstack((X[0:1, :].T, T[0:1, :].T))
        initial_u = usol[:, 0:1]
        left_xt = np.hstack((X[:, 0:1], T[:, 0:1]))
        left_u = usol[0:1, :].T
        right_xt = np.hstack((X[:, -1:], T[:, -1:]))
        right_u = usol[-1:, :].T

        train_xt = np.vstack((initial_xt, left_xt, right_xt))
        train_u = np.vstack((initial_u, left_u, right_u))
        idx = np.random.choice(train_xt.shape[0], self.config.n_u, replace=False)
        self.X_u_train_np = train_xt[idx]
        self.u_train_np = train_u[idx]

        self.lb = np.array([x.min(), t.min()], dtype=np.float64).reshape(-1)
        self.ub = np.array([x.max(), t.max()], dtype=np.float64).reshape(-1)
        sampler = qmc.LatinHypercube(d=2, seed=self.config.seed)
        colloc = qmc.scale(sampler.random(self.config.n_f), self.lb, self.ub)
        self.X_f_train_np = np.vstack((colloc, self.X_u_train_np))

        self.X_star_np = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
        self.u_star_np = usol.T.flatten()[:, None]
        self.exact_grid = usol.T
        self.x = x.squeeze()
        self.t = t.squeeze()

        self.X_u_train = torch.tensor(self.X_u_train_np)
        self.u_train = torch.tensor(self.u_train_np)
        self.X_f_train = torch.tensor(self.X_f_train_np)
        self.X_star = torch.tensor(self.X_star_np)

    def clone_model(self) -> BurgersPINN:
        clone = BurgersPINN(self.lb, self.ub)
        clone.load_state_dict(self.model.state_dict())
        return clone

    def loss_terms(self, model: BurgersPINN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_u = model(self.X_u_train)
        loss_u = torch.mean((pred_u - self.u_train) ** 2)
        residual = model.residual(self.X_f_train, self.config.nu)
        loss_f = torch.mean(residual**2)
        return loss_u + loss_f, loss_u, loss_f

    def evaluate_metrics(self, model: BurgersPINN) -> dict[str, float]:
        total, loss_u, loss_f = self.loss_terms(model)
        with torch.no_grad():
            pred = model(self.X_star).detach().cpu().numpy()
        rel_l2 = float(np.linalg.norm(pred - self.u_star_np, 2) / np.linalg.norm(self.u_star_np, 2))
        return {
            "loss": float(total.item()),
            "loss_u": float(loss_u.item()),
            "loss_f": float(loss_f.item()),
            "rel_l2": rel_l2,
        }

    def predict_grid(self, model: BurgersPINN) -> np.ndarray:
        with torch.no_grad():
            pred = model(self.X_star).detach().cpu().numpy()
        return pred.reshape(len(self.t), len(self.x))


def flatten_gradients(model: nn.Module) -> np.ndarray:
    grads = []
    for param in model.parameters():
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.reshape(-1))
    return torch.cat(grads).detach().cpu().numpy()


def set_model_parameters(model: nn.Module, vector: np.ndarray) -> None:
    tensor = torch.tensor(vector, dtype=torch.get_default_dtype())
    vector_to_parameters(tensor, model.parameters())


def model_parameters(model: nn.Module) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().cpu().numpy()


class Objective:
    """Flattened objective for line-search quasi-Newton methods."""

    def __init__(self, benchmark: BurgersBenchmark, model: BurgersPINN) -> None:
        self.benchmark = benchmark
        self.model = model
        self._cache_x: np.ndarray | None = None
        self._cache_value: float | None = None
        self._cache_grad: np.ndarray | None = None

    def value_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self._cache_x is not None and np.array_equal(x, self._cache_x):
            return self._cache_value, self._cache_grad  # type: ignore[return-value]

        set_model_parameters(self.model, x)
        self.model.zero_grad(set_to_none=True)
        loss, _, _ = self.benchmark.loss_terms(self.model)
        loss.backward()
        grad = flatten_gradients(self.model)

        self._cache_x = np.array(x, copy=True)
        self._cache_value = float(loss.item())
        self._cache_grad = grad
        return self._cache_value, self._cache_grad

    def value(self, x: np.ndarray) -> float:
        return self.value_and_grad(x)[0]

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.value_and_grad(x)[1]


def backtracking_line_search(
    objective: Objective,
    x: np.ndarray,
    value: float,
    grad: np.ndarray,
    direction: np.ndarray,
    *,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    shrink: float = 0.5,
    min_alpha: float = 1e-8,
) -> float:
    alpha = alpha0
    directional = float(np.dot(grad, direction))
    while alpha > min_alpha:
        new_value = objective.value(x + alpha * direction)
        if new_value <= value + c1 * alpha * directional:
            return alpha
        alpha *= shrink
    return min_alpha


def line_search_wolfe(objective: Objective, x: np.ndarray, value: float, grad: np.ndarray, direction: np.ndarray) -> float:
    alpha, *_ = line_search(objective.value, objective.grad, x, direction, grad, value, c2=0.9)
    if alpha is None or not np.isfinite(alpha):
        alpha = backtracking_line_search(objective, x, value, grad, direction)
    return float(alpha)


def log_entry(stage_iter: int, total_iter: int, benchmark: BurgersBenchmark, model: BurgersPINN, optimizer: str) -> dict[str, float | str]:
    metrics = benchmark.evaluate_metrics(model)
    return {
        "optimizer": optimizer,
        "stage_iter": float(stage_iter),
        "total_iter": float(total_iter),
        **metrics,
    }


def run_adam_warmstart(benchmark: BurgersBenchmark) -> tuple[BurgersPINN, list[dict[str, float | str]]]:
    model = benchmark.clone_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=benchmark.config.adam_lr)
    history: list[dict[str, float | str]] = [log_entry(0, 0, benchmark, model, "Adam warmup")]
    for step in range(1, benchmark.config.warmup_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = benchmark.loss_terms(model)
        loss.backward()
        optimizer.step()
        if step % 20 == 0 or step == benchmark.config.warmup_steps:
            history.append(log_entry(step, step, benchmark, model, "Adam warmup"))
    return model, history


def run_adam_continuation(benchmark: BurgersBenchmark, warm_model: BurgersPINN) -> list[dict[str, float | str]]:
    model = benchmark.clone_model()
    model.load_state_dict(warm_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=benchmark.config.adam_lr)
    history: list[dict[str, float | str]] = [log_entry(0, benchmark.config.warmup_steps, benchmark, model, "Adam")]
    for step in range(1, benchmark.config.second_stage_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = benchmark.loss_terms(model)
        loss.backward()
        optimizer.step()
        history.append(log_entry(step, benchmark.config.warmup_steps + step, benchmark, model, "Adam"))
    return history


def run_bfgs_family(
    benchmark: BurgersBenchmark,
    warm_model: BurgersPINN,
    *,
    optimizer_name: str,
    direction_builder: Callable[[dict[str, np.ndarray]], np.ndarray],
    state_updater: Callable[[dict[str, np.ndarray | list[np.ndarray] | int | float]], None],
) -> tuple[list[dict[str, float | str]], BurgersPINN]:
    model = benchmark.clone_model()
    model.load_state_dict(warm_model.state_dict())
    objective = Objective(benchmark, model)

    x = model_parameters(model)
    value, grad = objective.value_and_grad(x)
    dim = x.size
    state: dict[str, np.ndarray | list[np.ndarray] | int | float] = {
        "x": x,
        "value": np.array([value]),
        "grad": grad,
        "H": np.eye(dim),
        "s_list": [],
        "y_list": [],
        "rho_list": [],
        "alpha": 1.0,
    }

    history: list[dict[str, float | str]] = [log_entry(0, benchmark.config.warmup_steps, benchmark, model, optimizer_name)]
    for step in range(1, benchmark.config.second_stage_steps + 1):
        direction = direction_builder(state)
        grad_vec = state["grad"]  # type: ignore[assignment]
        if float(np.dot(direction, grad_vec)) >= 0.0:
            if "H" in state:
                state["H"] = np.eye(dim)
                direction = -grad_vec
            else:
                direction = -grad_vec

        value = float(state["value"][0])  # type: ignore[index]
        x = state["x"]  # type: ignore[assignment]
        alpha = line_search_wolfe(objective, x, value, grad_vec, direction)
        x_new = x + alpha * direction
        value_new, grad_new = objective.value_and_grad(x_new)
        s = x_new - x
        y = grad_new - grad_vec
        ys = float(np.dot(y, s))

        state["x"] = x_new
        state["value"] = np.array([value_new])
        state["grad"] = grad_new
        state["s"] = s
        state["y"] = y
        state["ys"] = ys
        state["alpha"] = alpha
        state["old_grad"] = grad_vec

        if ys > 1e-12:
            state_updater(state)
        else:
            if "H" in state:
                state["H"] = np.eye(dim)
            if "s_list" in state:
                state["s_list"] = []
                state["y_list"] = []
                state["rho_list"] = []

        set_model_parameters(model, x_new)
        history.append(log_entry(step, benchmark.config.warmup_steps + step, benchmark, model, optimizer_name))
    return history, model


def bfgs_direction(state: dict[str, np.ndarray]) -> np.ndarray:
    return -state["H"] @ state["grad"]


def bfgs_update(state: dict[str, np.ndarray | float]) -> None:
    H = state["H"]  # type: ignore[assignment]
    s = state["s"]  # type: ignore[assignment]
    y = state["y"]  # type: ignore[assignment]
    ys = float(state["ys"])  # type: ignore[arg-type]
    rho = 1.0 / ys
    identity = np.eye(H.shape[0])
    V = identity - rho * np.outer(s, y)
    H_new = V @ H @ V.T + rho * np.outer(s, s)
    state["H"] = 0.5 * (H_new + H_new.T)


def ssbfgs_update(state: dict[str, np.ndarray | float]) -> None:
    H = state["H"]  # type: ignore[assignment]
    s = state["s"]  # type: ignore[assignment]
    y = state["y"]  # type: ignore[assignment]
    ys = float(state["ys"])  # type: ignore[arg-type]
    grad = state["old_grad"]  # type: ignore[assignment]
    alpha = float(state["alpha"])  # type: ignore[arg-type]

    rho = 1.0 / ys
    identity = np.eye(H.shape[0])
    V = identity - rho * np.outer(s, y)
    H_bfgs = V @ H @ V.T + rho * np.outer(s, s)

    b_k_num = -alpha * float(np.dot(s, grad))
    b_k = b_k_num / max(ys, 1e-12)
    if b_k <= 0.0 or not np.isfinite(b_k):
        tau = 1.0
    else:
        tau = min(1.0, 1.0 / b_k)

    H_new = H_bfgs / tau
    state["H"] = 0.5 * (H_new + H_new.T)


def lbfgs_direction(state: dict[str, np.ndarray | list[np.ndarray]]) -> np.ndarray:
    grad = state["grad"]  # type: ignore[assignment]
    s_list = state["s_list"]  # type: ignore[assignment]
    y_list = state["y_list"]  # type: ignore[assignment]
    rho_list = state["rho_list"]  # type: ignore[assignment]

    q = grad.copy()
    alphas: list[float] = []
    for s, y, rho in zip(reversed(s_list), reversed(y_list), reversed(rho_list)):
        alpha = rho * float(np.dot(s, q))
        alphas.append(alpha)
        q = q - alpha * y

    if s_list:
        s_last = s_list[-1]
        y_last = y_list[-1]
        gamma = float(np.dot(s_last, y_last) / np.dot(y_last, y_last))
    else:
        gamma = 1.0
    r = gamma * q

    for s, y, rho, alpha in zip(s_list, y_list, rho_list, reversed(alphas)):
        beta = rho * float(np.dot(y, r))
        r = r + s * (alpha - beta)
    return -r


def lbfgs_update(state: dict[str, np.ndarray | list[np.ndarray] | int]) -> None:
    s = state["s"]  # type: ignore[assignment]
    y = state["y"]  # type: ignore[assignment]
    ys = float(state["ys"])  # type: ignore[arg-type]
    rho = 1.0 / ys
    s_list = state["s_list"]  # type: ignore[assignment]
    y_list = state["y_list"]  # type: ignore[assignment]
    rho_list = state["rho_list"]  # type: ignore[assignment]
    memory = 20
    s_list.append(s.copy())
    y_list.append(y.copy())
    rho_list.append(rho)
    if len(s_list) > memory:
        s_list.pop(0)
        y_list.pop(0)
        rho_list.pop(0)


def plot_optimizer_history(histories: dict[str, list[dict[str, float | str]]]) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), constrained_layout=True)
    colors = {
        "Adam": "#4C78A8",
        "L-BFGS": "#F58518",
        "BFGS": "#54A24B",
        "SSBFGS": "#E45756",
    }
    for name, history in histories.items():
        x = [row["stage_iter"] for row in history]
        loss = [row["loss"] for row in history]
        err = [row["rel_l2"] for row in history]
        axes[0].plot(x, loss, label=name, lw=1.8, color=colors[name])
        axes[1].plot(x, err, label=name, lw=1.8, color=colors[name])

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Second-stage iterations")
    axes[1].set_xlabel("Second-stage iterations")
    axes[0].set_ylabel("Training loss")
    axes[1].set_ylabel(r"Relative $L^2$ error")
    axes[0].set_title("Loss decay after a shared Adam warm start")
    axes[1].set_title("Solution error after a shared Adam warm start")
    axes[1].legend(loc="best")
    return fig


def plot_optimizer_slices(
    benchmark: BurgersBenchmark,
    predictions: dict[str, np.ndarray],
) -> plt.Figure:
    times = [0.25, 0.50, 0.75]
    idx = [int(np.argmin(np.abs(benchmark.t - val))) for val in times]
    exact = benchmark.exact_grid

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    styles = {
        "Exact": ("#222222", "--"),
        "L-BFGS": ("#F58518", "-"),
        "BFGS": ("#54A24B", "-"),
        "SSBFGS": ("#E45756", "-"),
    }
    for ax, column, time_value in zip(axes, idx, times):
        ax.plot(benchmark.x, exact[column], color=styles["Exact"][0], linestyle=styles["Exact"][1], lw=2.0, label="Exact")
        for name in ("L-BFGS", "BFGS", "SSBFGS"):
            ax.plot(
                benchmark.x,
                predictions[name][column],
                color=styles[name][0],
                linestyle=styles[name][1],
                lw=1.7,
                label=name,
            )
        ax.set_title(rf"$t={time_value:.2f}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(x,t)$")
    axes[-1].legend(loc="best")
    return fig


def save_figure(fig: plt.Figure, basename: str, *, copy_to_chapter: bool = False) -> None:
    LOCAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    CHAPTER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOCAL_FIG_DIR / f"{basename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(LOCAL_FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    if copy_to_chapter:
        fig.savefig(CHAPTER_FIG_DIR / f"{basename}.png", dpi=220, bbox_inches="tight")
        fig.savefig(CHAPTER_FIG_DIR / f"{basename}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_notebook(summary: dict[str, dict[str, float]]) -> nbf.NotebookNode:
    cells: list[nbf.NotebookNode] = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# Optimizer Benchmark for Burgers PINNs\n\n"
            "This notebook compares optimizers on the continuous-time Burgers equation.\n"
            "All runs share the same double-precision `4x20` `tanh` PINN, the same training points,\n"
            "and the same Adam warm start.\n"
            "Only the second-stage optimizer changes."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Main comparison\n\n"
            f"Final relative errors in this local benchmark are\n"
            f"`Adam={summary['Adam']['rel_l2']:.2e}`,\n"
            f"`L-BFGS={summary['L-BFGS']['rel_l2']:.2e}`,\n"
            f"`BFGS={summary['BFGS']['rel_l2']:.2e}`,\n"
            f"and `SSBFGS={summary['SSBFGS']['rel_l2']:.2e}`.\n\n"
            f"![Optimizer history]({HISTORY_FIG_BASENAME}.png)\n\n"
            f"![Optimizer slices]({SLICE_FIG_BASENAME}.png)"
        )
    )
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )


def update_pinns_md() -> None:
    text = PINN_MD_PATH.read_text()
    section = (
        "\n### Optimizer benchmarks for PINNs\n"
        "- [Burgers optimizer benchmark](burgers-optimizers.ipynb)\n"
        "    - PyTorch [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/sciml-book/sciml_notebook/blob/main/pinns/burgers-optimizers.ipynb)\n"
    )
    if "### Optimizer benchmarks for PINNs\n- [Burgers optimizer benchmark](burgers-optimizers.ipynb)\n" not in text:
        marker = (
            "### Failure modes in PINNs\n"
            "- [Convection Failure Modes](pinn-failure-modes.ipynb)\n"
            "    - PyTorch [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/github/sciml-book/sciml_notebook/blob/main/pinns/pinn-failure-modes.ipynb)\n"
        )
        if marker in text:
            text = text.replace(marker, marker + section)
        else:
            text += section
    PINN_MD_PATH.write_text(text)


def main() -> None:
    benchmark = BurgersBenchmark(BurgersConfig())

    print("running Adam warm start", flush=True)
    warm_model, warm_history = run_adam_warmstart(benchmark)

    print("running optimizer continuations", flush=True)
    adam_history = run_adam_continuation(benchmark, warm_model)
    lbfgs_history, lbfgs_model = run_bfgs_family(
        benchmark,
        warm_model,
        optimizer_name="L-BFGS",
        direction_builder=lbfgs_direction,
        state_updater=lbfgs_update,
    )
    bfgs_history, bfgs_model = run_bfgs_family(
        benchmark,
        warm_model,
        optimizer_name="BFGS",
        direction_builder=bfgs_direction,
        state_updater=bfgs_update,
    )
    ssbfgs_history, ssbfgs_model = run_bfgs_family(
        benchmark,
        warm_model,
        optimizer_name="SSBFGS",
        direction_builder=bfgs_direction,
        state_updater=ssbfgs_update,
    )

    histories = {
        "Adam": adam_history,
        "L-BFGS": lbfgs_history,
        "BFGS": bfgs_history,
        "SSBFGS": ssbfgs_history,
    }

    predictions = {
        "L-BFGS": benchmark.predict_grid(lbfgs_model),
        "BFGS": benchmark.predict_grid(bfgs_model),
        "SSBFGS": benchmark.predict_grid(ssbfgs_model),
    }
    summary = {
        name: {
            "rel_l2": float(history[-1]["rel_l2"]),
            "loss": float(history[-1]["loss"]),
        }
        for name, history in histories.items()
    }

    print("saving figures", flush=True)
    save_figure(plot_optimizer_history(histories), HISTORY_FIG_BASENAME, copy_to_chapter=True)
    save_figure(plot_optimizer_slices(benchmark, predictions), SLICE_FIG_BASENAME, copy_to_chapter=True)

    print("writing notebook", flush=True)
    NOTEBOOK_PATH.write_text(nbf.writes(build_notebook(summary)))
    update_pinns_md()

    for name, metrics in summary.items():
        print(f"{name}: loss={metrics['loss']:.3e} rel_l2={metrics['rel_l2']:.3e}")
    print(f"wrote notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
