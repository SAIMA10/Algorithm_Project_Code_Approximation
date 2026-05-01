"""
Running the experiment for the 3/4-approximation MAX SAT algorithm as per requirements.
Grid: num_vars in {100,200,300,400,500} x num_clauses in {100,200,300,400,500}.
Averaged over NUM_SEEDS random instances per cell.
"""

import json, random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_structures import MaxSATInstance, Clause
from src.lp_solver import solve_lp_relaxation
from src.johnson import johnson_assignment
from src.lp_rounding import lp_rounding_assignment
from src.evaluation import evaluate_assignment
from src.approx_algo import three_quarter_approximation  

def generate_instance(num_vars, num_clauses, max_clause_len=3, seed=0):
    rng = random.Random(seed)
    clauses = []
    for _ in range(num_clauses):
        k = rng.randint(1, min(max_clause_len, num_vars))
        chosen = rng.sample(range(num_vars), k)
        pos, neg = [], []
        for v in chosen:
            (pos if rng.random() < 0.5 else neg).append(v)
        clauses.append(Clause(pos, neg, float(rng.randint(1, 10))))
    return MaxSATInstance(num_vars=num_vars, clauses=clauses)


def run_one(instance):
    """
    Run all three algorithms on one instance.
    Returns: (lp_val, johnson_val, lp_round_val, best_val, method_chosen)
    """
    best_assign, best_val, method, lp_val = three_quarter_approximation(instance)

    y_star, _, _ = solve_lp_relaxation(instance)
    j_val  = evaluate_assignment(instance, johnson_assignment(instance))
    lr_val = evaluate_assignment(instance, lp_rounding_assignment(instance, y_star=y_star))

    return lp_val, j_val, lr_val, best_assign, best_val, method


VAR_SIZES    = [100, 200, 300, 400, 500]
CLAUSE_SIZES = [100, 200, 300, 400, 500]
NUM_SEEDS    = 10
MAX_LEN      = 3

print(f"Running grid experiment: {len(VAR_SIZES)}x{len(CLAUSE_SIZES)} cells, "
      f"{NUM_SEEDS} seeds each, max_clause_len={MAX_LEN}")
print(f"Central algorithm: three_quarter_approximation() from approx_algo.py\n")

records = []

for nv in VAR_SIZES:
    for nc in CLAUSE_SIZES:
        lp_list, j_list, lr_list, best_list, method_list = [], [], [], [], []
        for seed in range(NUM_SEEDS):
            best_assign_list = []
            inst = generate_instance(nv, nc, max_clause_len=MAX_LEN,
                                     seed=seed * 7919 + nv * 31 + nc)
            lp_val, j_val, lr_val, best_assign, best_val, method = run_one(inst)
            lp_list.append(lp_val)
            j_list.append(j_val)
            lr_list.append(lr_val)
            best_list.append(best_val)
            best_assign_list.append(best_assign)
            method_list.append(method)

        avg_lp      = np.mean(lp_list)
        avg_j       = np.mean(j_list)
        avg_lr      = np.mean(lr_list)
        avg_best    = np.mean(best_list)
        j_ratio     = avg_j    / avg_lp
        lr_ratio    = avg_lr   / avg_lp
        best_ratio  = avg_best / avg_lp
        pct_johnson = method_list.count("johnson") / NUM_SEEDS * 100

        rec = dict(vars=nv, clauses=nc,
                   avg_lp=avg_lp, avg_j=avg_j, avg_lr=avg_lr, avg_best=avg_best,
                   j_ratio=j_ratio, lr_ratio=lr_ratio, best_ratio=best_ratio,
                   pct_johnson=pct_johnson)
        records.append(rec)
        print(f"  vars={nv:3d}, clauses={nc:3d}:  LP={avg_lp:8.1f}  "
              f"j_ratio={j_ratio:.4f}  lr_ratio={lr_ratio:.4f}  "
              f"best_ratio(3/4 algo)={best_ratio:.4f}  "
              f"johnson_chosen={pct_johnson:.0f}%")

with open("results.json", "w") as f:
    json.dump(records, f, indent=2)
print("\nResults saved to results.json")



def get_matrix(records, key):
    M = np.zeros((len(VAR_SIZES), len(CLAUSE_SIZES)))
    for r in records:
        i = VAR_SIZES.index(r["vars"])
        j = CLAUSE_SIZES.index(r["clauses"])
        M[i, j] = r[key]
    return M

COLORS = {"johnson": "#2563EB", "lp_round": "#DC2626", "best": "#16A34A"}


# Plot 1: ratio vs num_clauses, one line per num_vars
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(
    "Approximation Ratio (algorithm value / LP optimum) vs Number of Clauses\n"
    f"Averaged over {NUM_SEEDS} random instances  |  max clause length = {MAX_LEN}",
    fontsize=11)

for ax, key, label, color in zip(
        axes,
        ["j_ratio",  "lr_ratio",  "best_ratio"],
        ["Johnson",  "LP Rounding", "3/4 Algorithm  (max of Johnson & LP-rounding)"],
        [COLORS["johnson"], COLORS["lp_round"], COLORS["best"]]):
    M = get_matrix(records, key)
    for vi, nv in enumerate(VAR_SIZES):
        ax.plot(CLAUSE_SIZES, M[vi, :], marker="o", label=f"vars={nv}", linewidth=1.8)
    ax.axhline(0.75, color="black", linestyle="--", linewidth=1.2, label="3/4 guarantee")
    ax.axhline(1.00, color="gray",  linestyle=":",  linewidth=1.0)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Number of Clauses")
    ax.set_ylabel("Ratio  (value / LP optimum)" if ax is axes[0] else "")
    ax.legend(fontsize=8); ax.set_ylim(0.70, 1.02); ax.grid(True, alpha=0.3)
    # directly shows how quality changes as clause count grows.
plt.tight_layout()
plt.savefig("plot1_ratio_vs_clauses.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot1_ratio_vs_clauses.png")


#Plot 2: ratio vs num_vars, one line per num_clauses
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle(
    "Approximation Ratio (algorithm value / LP optimum) vs Number of Variables\n"
    f"Averaged over {NUM_SEEDS} random instances  |  max clause length = {MAX_LEN}",
    fontsize=11)

for ax, key, label, color in zip(
        axes,
        ["j_ratio",  "lr_ratio",  "best_ratio"],
        ["Johnson",  "LP Rounding", "3/4 Algorithm  (max of Johnson & LP-rounding)"],
        [COLORS["johnson"], COLORS["lp_round"], COLORS["best"]]):
    M = get_matrix(records, key)
    for ci, nc in enumerate(CLAUSE_SIZES):
        ax.plot(VAR_SIZES, M[:, ci], marker="s", label=f"clauses={nc}", linewidth=1.8)
    ax.axhline(0.75, color="black", linestyle="--", linewidth=1.2, label="3/4 guarantee")
    ax.axhline(1.00, color="gray",  linestyle=":",  linewidth=1.0)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Number of Variables")
    ax.set_ylabel("Ratio  (value / LP optimum)" if ax is axes[0] else "")
    ax.legend(fontsize=8); ax.set_ylim(0.70, 1.02); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot2_ratio_vs_vars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot2_ratio_vs_vars.png")


# Plot 4: all three algorithms on one axis for vars=100
# change the variable number to plot for different variable plot 
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(
    f"All Three Algorithms — vars=100  (averaged over {NUM_SEEDS} seeds)\n"
    "'3/4 Algorithm' is three_quarter_approximation() from approx_algo.py",
    fontsize=10)

nv = 100
for key, label, color, ls in [
        ("j_ratio",    "Johnson",                    COLORS["johnson"],  "-"),
        ("lr_ratio",   "LP Rounding",                COLORS["lp_round"], "-"),
        ("best_ratio", "3/4 Algorithm (best of both)", COLORS["best"],   "-")]:
    row = sorted([r for r in records if r["vars"] == nv], key=lambda r: r["clauses"])
    ax.plot(CLAUSE_SIZES, [r[key] for r in row],
            marker="o", label=label, color=color, linewidth=2, linestyle=ls)

ax.axhline(0.75, color="black", linestyle="--", linewidth=1.4, label="3/4 guarantee")
ax.axhline(1.00, color="gray",  linestyle=":",  linewidth=1.0)
ax.set_xlabel("Number of Clauses"); ax.set_ylabel("Ratio  (value / LP optimum)")
ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0.70, 1.02)
plt.tight_layout()
plt.savefig("plot4_comparison_vars100.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot4_comparison_vars100.png")


# Plot 5: % of times Johnson wins
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(
    "% of Instances Where 3/4 Algorithm Chose Johnson over LP-Rounding\n"
    f"(averaged over {NUM_SEEDS} seeds)", fontsize=10)

for nv in VAR_SIZES:
    row = sorted([r for r in records if r["vars"] == nv], key=lambda r: r["clauses"])
    ax.plot(CLAUSE_SIZES, [r["pct_johnson"] for r in row],
            marker="o", label=f"vars={nv}", linewidth=1.8)

ax.axhline(50, color="gray", linestyle="--", linewidth=1.0)
ax.set_xlabel("Number of Clauses"); ax.set_ylabel("% instances Johnson chosen by 3/4 algo")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-5, 105)
plt.tight_layout()
plt.savefig("plot5_johnson_win_rate.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved plot5_johnson_win_rate.png")


# Other plots:
# # Plot 3: heatmaps
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# fig.suptitle(
#     f"Approximation Ratio Heatmaps  (value / LP optimum)  —  averaged over {NUM_SEEDS} seeds",
#     fontsize=11)

# for ax, key, label in zip(
#         axes,
#         ["best_ratio", "j_ratio",  "lr_ratio"],
#         ["3/4 Algorithm (best of both)", "Johnson", "LP Rounding"]):
#     M = get_matrix(records, key)
#     im = ax.imshow(M, vmin=0.90, vmax=1.0, cmap="RdYlGn", origin="lower", aspect="auto")
#     ax.set_xticks(range(len(CLAUSE_SIZES))); ax.set_xticklabels(CLAUSE_SIZES)
#     ax.set_yticks(range(len(VAR_SIZES)));   ax.set_yticklabels(VAR_SIZES)
#     ax.set_xlabel("Number of Clauses"); ax.set_ylabel("Number of Variables")
#     ax.set_title(label, fontsize=10)
#     for i in range(len(VAR_SIZES)):
#         for j in range(len(CLAUSE_SIZES)):
#             ax.text(j, i, f"{M[i,j]:.3f}", ha="center", va="center", fontsize=8)
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     # Each cell shows the average ratio for a specific (vars,clauses) pair.
# plt.tight_layout()
# plt.savefig("plot3_heatmap.png", dpi=150, bbox_inches="tight")
# plt.close()
# print("Saved plot3_heatmap.png")