import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# ==========================
# 配置
# ==========================

# 单次 explain 输出目录
RUN_DIR = Path("./experiments/results/california_run1")

# 想要在横轴上显示的 Target distinctiveness 网格
# 这里采用与原图类似的 6 个点
ALPHA_GRID = np.linspace(0.2, 1.2, 6)

# 目前只画 DISCO 一条线；后续你可以加入其它方法
METHOD_ORDER = ["DISCO"]


# ==========================
# 1. 从 explain 输出收集记录
# ==========================

def collect_records_from_run(run_dir: Path) -> pd.DataFrame:
    """
    仿照 alpha_target_auc_pier.ipynb 中的 records 结构，
    但数据来源是一次 explain 跑出的 diagnostics.json。

    这里使用 |peak_tau| 作为“target distinctiveness”的原始度量，
    后面会把它归一化映射到 ALPHA_GRID。
    """
    records = []

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # 目录结构：t=<idx>/x_id=<global_index>/d=<intervention.name>/diagnostics.json
    for t_dir in sorted(run_dir.glob("t=*")):
        try:
            t_idx = int(t_dir.name.split("=")[1])
        except Exception:
            continue

        for q_dir in sorted(t_dir.glob("x_id=*")):
            try:
                x_id = int(q_dir.name.split("=")[1])
            except Exception:
                continue

            query_label = f"t={t_idx}_x={x_id}"

            for d_dir in sorted(q_dir.glob("d=*")):
                interv_name = d_dir.name.split("=", 1)[1]
                diag_path = d_dir / "diagnostics.json"
                if not diag_path.exists():
                    continue

                with open(diag_path) as f:
                    diag = json.load(f)

                # 需要 auc_te 和 peak_tau；peak_tau 可能为 None
                if "auc_te" not in diag or "peak_tau" not in diag:
                    continue
                if diag["peak_tau"] is None:
                    continue

                auc_te = float(diag["auc_te"])
                peak_tau = float(diag["peak_tau"])

                # 使用 |peak_tau| 作为原始 distinctiveness
                distinctiveness = abs(peak_tau)

                records.append(
                    {
                        "method": "DISCO",
                        "target_idx": t_idx,
                        "x_id": x_id,
                        "query_label": query_label,
                        "intervention": interv_name,
                        "distinctiveness_raw": distinctiveness,
                        # 与 notebook 对齐的字段名
                        "auc_abs": auc_te,
                    }
                )

    if not records:
        raise RuntimeError(
            "No usable diagnostics.json (with 'auc_te' and 'peak_tau') "
            f"found under {run_dir}"
        )

    return pd.DataFrame(records)


df = collect_records_from_run(RUN_DIR)
print(f"Collected {len(df)} records from {RUN_DIR}")


# ==========================
# 2. 把 distinctiveness 映射到 ALPHA_GRID
# ==========================

# 原始 distinctiveness（来自 |peak_tau|）
dist = df["distinctiveness_raw"].values
dist_min = float(dist.min())
dist_max = float(dist.max())

if dist_max > dist_min:
    # 归一化到 [0, 1]
    dist_norm = (dist - dist_min) / (dist_max - dist_min)
else:
    # 所有点一样，退化成全 0
    dist_norm = np.zeros_like(dist)

# 映射到最近的 Target distinctiveness 网格点
# 索引范围 [0, len(ALPHA_GRID)-1]
idx = np.round(dist_norm * (len(ALPHA_GRID) - 1)).astype(int)
idx = np.clip(idx, 0, len(ALPHA_GRID) - 1)

df["alpha_target"] = ALPHA_GRID[idx]
df["seed"] = 0  # 这里没有多 seed，就用占位
# 和原 notebook 的列名保持一致
df["pier_auc_abs_diff"] = np.nan
df["pier_nrmse"] = np.nan
df["pier_corr"] = np.nan

print("Sample of records with assigned alpha_target:")
print(df.head())


# ==========================
# 3. 按 notebook 方式 summary
# ==========================

results_df = df  # 命名对齐 notebook

summary = (
    results_df.groupby(["method", "alpha_target"])
    .agg(
        mean_auc=("auc_abs", "mean"),
        std_auc=("auc_abs", "std"),
        mean_delta=("pier_auc_abs_diff", "mean"),
        std_delta=("pier_auc_abs_diff", "std"),
        mean_corr=("pier_corr", "mean"),
        std_corr=("pier_corr", "std"),
        count=("auc_abs", "count"),
    )
    .reset_index()
)

summary["sem_auc"] = summary["std_auc"] / np.sqrt(summary["count"].clip(lower=1))
summary["sem_delta"] = summary["std_delta"] / np.sqrt(summary["count"].clip(lower=1))
summary["sem_corr"] = summary["std_corr"] / np.sqrt(summary["count"].clip(lower=1))

print("Summary table:")
print(summary)


# ==========================
# 4. 画图：复用原 notebook 风格
# ==========================

plt.rcParams["font.family"] = "Times New Roman"

base_methods = [m for m in METHOD_ORDER if m.lower() != "oracle"]

fill_palette = sns.color_palette("Set2", n_colors=len(base_methods))
line_palette = sns.color_palette("Set1", n_colors=len(base_methods))

color_map_fill = {m: c for m, c in zip(base_methods, fill_palette)}
color_map_line = {m: c for m, c in zip(base_methods, line_palette)}

if "Oracle" in METHOD_ORDER:
    color_map_fill["Oracle"] = (0.2, 0.2, 0.2)
    color_map_line["Oracle"] = (0.0, 0.0, 0.0)

linestyles = ["-", "--", "-.", ":"]
markers = ["o", "s", "D", "^", "v", "P", "*", "X", "h"]

style_map = {}
for i, method in enumerate(METHOD_ORDER):
    style_map[method] = (
        linestyles[i % len(linestyles)],
        markers[i % len(markers)],
    )

fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))

mean_key = "mean_auc"
sem_key = "sem_auc"
y_label = "AUC-PIER"
title = "PIER magnitude"

ax.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.2)
    spine.set_color("black")

ax.set_xlabel("Target distinctiveness", fontsize=12)
ax.set_ylabel(y_label, fontsize=12)
ax.set_title(title, fontsize=12)

handles = []
labels = []

alphas_sorted = np.sort(summary["alpha_target"].unique())

for method in METHOD_ORDER:
    data = summary[summary["method"] == method].sort_values("alpha_target")
    if data.empty:
        continue

    linestyle, marker = style_map[method]

    ax.fill_between(
        data["alpha_target"],
        data[mean_key] - data[sem_key],
        data[mean_key] + data[sem_key],
        color=color_map_fill[method],
        alpha=0.25,
        zorder=1,
    )

    line, = ax.plot(
        data["alpha_target"],
        data[mean_key],
        marker=marker,
        markersize=7,
        linestyle=linestyle,
        linewidth=2.0,
        color=color_map_line[method],
        label=method,
        zorder=2,
    )

    handles.append(line)
    labels.append(method)

ax.set_xticks(alphas_sorted)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.7)

if handles:
    ax.legend(
        handles,
        labels,
        fontsize=10,
        frameon=True,
        framealpha=0.8,
        edgecolor="black",
        loc="best",
    )

plt.tight_layout()

pdf_path = RUN_DIR / "pier_magnitude_california_grid.pdf"
png_path = RUN_DIR / "pier_magnitude_california_grid.png"
plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to:\n  {pdf_path}\n  {png_path}")

plt.show()

