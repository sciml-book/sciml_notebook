#!/usr/bin/env python3
"""Build a PINN failure-mode notebook from the Krishnapriyan convection benchmark."""

from __future__ import annotations

import os
import sys
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


STYLE_PATH = ROOT / "sciml_notebook" / "docs" / "sciml_style.mplstyle"
plt.style.use(STYLE_PATH)
torch.set_num_threads(1)

REPO_PBC_DIR = ROOT / "characterizing-pinns-failure-modes" / "pbc_examples"
if str(REPO_PBC_DIR) not in sys.path:
    sys.path.append(str(REPO_PBC_DIR))

from net_pbc import PhysicsInformedNN_pbc  # noqa: E402
from systems_pbc import convection_diffusion  # noqa: E402
from utils import sample_random, set_seed  # noqa: E402


NOTEBOOK_PATH = Path(__file__).with_name("pinn-failure-modes.ipynb")
PINN_MD_PATH = Path(__file__).with_name("pinns.md")
LOCAL_FIG_DIR = Path(__file__).parent
CHAPTER_FIG_DIR = ROOT / "sciml-book" / "chapters" / "06-pinns" / "figs"
FAILURE_FIG_BASENAME = "convection-pinn-failure"


@dataclass(frozen=True)
class ConvectionConfig:
    """Configuration for one convection benchmark run."""

    beta: float
    seed: int
    xgrid: int = 128
    nt: int = 100
    n_f: int = 1000
    layers: tuple[int, ...] = (50, 50, 50, 50, 1)
    lr: float = 1.0
    L: float = 1.0
    activation: str = "tanh"
    loss_style: str = "mean"
    u0_str: str = "sin(x)"


def build_convection_data(cfg: ConvectionConfig) -> dict[str, np.ndarray]:
    x = np.linspace(0.0, 2.0 * np.pi, cfg.xgrid, endpoint=False).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, cfg.nt).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    x_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    t_noinitial = t[1:]
    x_noboundary = x[1:]
    X_inner, T_inner = np.meshgrid(x_noboundary, t_noinitial)
    x_inner = np.hstack((X_inner.flatten()[:, None], T_inner.flatten()[:, None]))
    x_f_train = sample_random(x_inner, cfg.n_f)

    u_vals = convection_diffusion(
        cfg.u0_str,
        nu=0.0,
        beta=cfg.beta,
        source=0.0,
        xgrid=cfg.xgrid,
        nt=cfg.nt,
    )
    exact = u_vals.reshape(len(t), len(x))

    x_u_train = np.hstack((X[0:1, :].T, T[0:1, :].T))
    u_train = exact[0:1, :].T
    bc_lb = np.hstack((X[:, 0:1], T[:, 0:1]))
    x_bc_ub = np.full_like(t, 2.0 * np.pi)
    bc_ub = np.hstack((x_bc_ub, t))

    return {
        "x": x,
        "t": t,
        "X_star": x_star,
        "X_u_train": x_u_train,
        "u_train": u_train,
        "X_f_train": x_f_train,
        "bc_lb": bc_lb,
        "bc_ub": bc_ub,
        "exact": exact,
        "u_star": u_vals.reshape(-1, 1),
    }


def run_convection_case(cfg: ConvectionConfig, *, verbose: bool = False) -> dict[str, object]:
    set_seed(cfg.seed)
    data = build_convection_data(cfg)
    layers = [int(data["X_u_train"].shape[-1]), *cfg.layers]
    forcing = np.zeros(data["X_f_train"].shape[0], dtype=float)

    model = PhysicsInformedNN_pbc(
        "convection",
        data["X_u_train"],
        data["u_train"],
        data["X_f_train"],
        data["bc_lb"],
        data["bc_ub"],
        layers,
        forcing,
        0.0,
        cfg.beta,
        0.0,
        "LBFGS",
        cfg.lr,
        "DNN",
        cfg.L,
        cfg.activation,
        cfg.loss_style,
    )
    model.dnn.train()
    model.optimizer.step(lambda: model.loss_pinn(verbose=verbose))

    prediction = model.predict(data["X_star"]).reshape(cfg.nt, cfg.xgrid)
    pred_star = prediction.reshape(-1, 1)
    exact_star = data["u_star"]

    rel_l2 = float(np.linalg.norm(exact_star - pred_star, 2) / np.linalg.norm(exact_star, 2))
    abs_l1 = float(np.mean(np.abs(exact_star - pred_star)))
    linf = float(np.linalg.norm(exact_star - pred_star, np.inf) / np.linalg.norm(exact_star, np.inf))

    return {
        "config": cfg,
        "x": data["x"].ravel(),
        "t": data["t"].ravel(),
        "exact": data["exact"],
        "prediction": prediction,
        "abs_error": np.abs(data["exact"] - prediction),
        "rel_l2": rel_l2,
        "abs_l1": abs_l1,
        "linf": linf,
    }


def run_convection_sweep(
    betas: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0),
    seeds: tuple[int, ...] = (0, 1, 2),
    *,
    verbose: bool = False,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for beta in betas:
        for seed in seeds:
            cfg = ConvectionConfig(beta=beta, seed=seed)
            results.append(run_convection_case(cfg, verbose=verbose))
    return results


def summarize_sweep(results: list[dict[str, object]]) -> dict[float, dict[str, np.ndarray]]:
    summary: dict[float, dict[str, np.ndarray]] = {}
    betas = sorted({float(item["config"].beta) for item in results})  # type: ignore[index]
    for beta in betas:
        values = [
            float(item["rel_l2"])
            for item in results
            if abs(float(item["config"].beta) - beta) < 1e-12  # type: ignore[index]
        ]
        arr = np.array(values, dtype=float)
        summary[beta] = {
            "values": arr,
            "mean": np.array(arr.mean(), ndmin=1),
            "std": np.array(arr.std(ddof=0), ndmin=1),
            "min": np.array(arr.min(), ndmin=1),
            "max": np.array(arr.max(), ndmin=1),
        }
    return summary


def plot_failure_summary(
    results: list[dict[str, object]],
    representative: dict[str, object],
) -> plt.Figure:
    summary = summarize_sweep(results)
    betas = np.array(sorted(summary), dtype=float)
    means = np.array([summary[beta]["mean"][0] for beta in betas], dtype=float)
    mins = np.array([summary[beta]["min"][0] for beta in betas], dtype=float)
    maxs = np.array([summary[beta]["max"][0] for beta in betas], dtype=float)

    fig = plt.figure(figsize=(12.5, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.2])

    ax_curve = fig.add_subplot(grid[0, :])
    ax_curve.semilogy(betas, means, color="black", marker="o", lw=1.8)
    ax_curve.fill_between(betas, mins, maxs, color="0.7", alpha=0.35)
    for beta in betas:
        values = summary[beta]["values"]
        ax_curve.scatter(
            np.full_like(values, beta),
            values,
            color="0.35",
            s=18,
            zorder=3,
        )
    ax_curve.set_xlabel(r"Convection coefficient $\beta$")
    ax_curve.set_ylabel("Relative $L^2$ error")
    ax_curve.set_title("Vanilla PINN error rises sharply as convection strengthens")
    ax_curve.set_xticks(betas)

    exact = representative["exact"]  # type: ignore[assignment]
    prediction = representative["prediction"]  # type: ignore[assignment]
    abs_error = representative["abs_error"]  # type: ignore[assignment]
    x = representative["x"]  # type: ignore[assignment]
    t = representative["t"]  # type: ignore[assignment]
    beta = float(representative["config"].beta)  # type: ignore[index]
    extent = [float(t.min()), float(t.max()), float(x.min()), float(x.max())]

    heatmap_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[1, 2]),
    ]
    fields = [exact.T, prediction.T, abs_error.T]
    titles = ["Exact solution", f"Vanilla PINN at $\\beta={beta:.0f}$", "Absolute error"]
    cmaps = ["viridis", "viridis", "magma"]
    vmin = min(float(exact.min()), float(prediction.min()))
    vmax = max(float(exact.max()), float(prediction.max()))
    norms = [(vmin, vmax), (vmin, vmax), (0.0, float(abs_error.max()))]

    for ax, field, title, cmap, (lo, hi) in zip(heatmap_axes, fields, titles, cmaps, norms):
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

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


def build_notebook(results: list[dict[str, object]], representative: dict[str, object]) -> nbf.NotebookNode:
    summary = summarize_sweep(results)
    beta20 = summary[20.0]["mean"][0]
    beta30 = summary[30.0]["mean"][0]
    beta40 = summary[40.0]["mean"][0]

    cells: list[nbf.NotebookNode] = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# When Vanilla PINNs Fail on Convection\n\n"
            "This notebook reproduces the convection benchmark from Krishnapriyan et al.\n"
            "The PDE is\n"
            "$\\frac{\\partial u}{\\partial t}+\\beta\\frac{\\partial u}{\\partial x}=0$\n"
            "on a periodic domain with initial condition $u(x,0)=\\sin(x)$.\n"
            "The exact solution is a translation in space.\n"
            "The shape does not change.\n"
            "That makes convection a clean test of whether a PINN can preserve transport."
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. What the reproduced sweep shows\n\n"
            f"In this local run, the mean relative $L^2$ error is {beta20:.2e} at $\\beta=20$.\n"
            f"It rises to {beta30:.2e} at $\\beta=30$ and {beta40:.2e} at $\\beta=40$.\n"
            "The heatmaps below show what that failure looks like in space and time.\n\n"
            f"![Convection failure summary]({FAILURE_FIG_BASENAME}.png)"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from rewrite_pinn_failure_modes import (\n"
            "    ConvectionConfig,\n"
            "    plot_failure_summary,\n"
            "    run_convection_case,\n"
            "    run_convection_sweep,\n"
            ")\n"
            "\n"
            "results = run_convection_sweep(verbose=False)\n"
            "representative = run_convection_case(ConvectionConfig(beta=30.0, seed=0))\n"
            "plot_failure_summary(results, representative);"
        )
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. Why this example matters\n\n"
            "The network is not failing because the exact solution is complicated.\n"
            "The solution is a shifted sine wave.\n"
            "The difficulty comes from the optimization problem induced by the PINN loss.\n"
            "That is the point the chapter makes before turning to curriculum and sequence-to-sequence remedies."
        )
    )
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    )


def update_pinns_md() -> None:
    text = PINN_MD_PATH.read_text()
    section = (
        "\n### Failure modes in PINNs\n"
        "- [Convection Failure Modes](pinn-failure-modes.ipynb)\n"
        "    - PyTorch [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        "(https://colab.research.google.com/github/sciml-book/sciml_notebook/blob/main/pinns/pinn-failure-modes.ipynb)\n"
    )
    if "### Failure modes in PINNs\n- [Convection Failure Modes](pinn-failure-modes.ipynb)\n" not in text:
        insert_after = (
            "### NTK and spectral bias in PINNs\n"
            "- [Mixed-Signal Poisson PINN](ntk-spectral-bias.ipynb)\n"
            "    - PyTorch [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            "(https://colab.research.google.com/github/sciml-book/sciml_notebook/blob/main/pinns/ntk-spectral-bias.ipynb)\n"
        )
        if insert_after in text:
            text = text.replace(insert_after, insert_after + section)
        else:
            text += section
    PINN_MD_PATH.write_text(text)


def main() -> None:
    print("running convection sweep", flush=True)
    results = run_convection_sweep(verbose=False)
    print("running representative beta=30 case", flush=True)
    representative = run_convection_case(ConvectionConfig(beta=30.0, seed=0), verbose=False)

    print("saving figure", flush=True)
    save_figure(plot_failure_summary(results, representative), FAILURE_FIG_BASENAME, copy_to_chapter=True)

    print("writing notebook", flush=True)
    NOTEBOOK_PATH.write_text(nbf.writes(build_notebook(results, representative)))

    print("updating notebook index", flush=True)
    update_pinns_md()

    summary = summarize_sweep(results)
    for beta in sorted(summary):
        mean = summary[beta]["mean"][0]
        lo = summary[beta]["min"][0]
        hi = summary[beta]["max"][0]
        print(f"beta={beta:.0f} mean={mean:.4e} range=[{lo:.4e}, {hi:.4e}]")
    print(f"wrote notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
