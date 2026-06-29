#!/usr/bin/env python3
import argparse
import glob
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


ATTACK_STYLES = {
    "completion": {"color": "#0f172a", "marker": "o", "display": "Fake Completion"},
    "completion+cmb": {"color": "#1e3a8a", "marker": "o", "display": "Combined"},
    "ignore": {"color": "#1d4ed8", "marker": "o", "display": "Ignore"},
    "naive": {"color": "#2563eb", "marker": "o", "display": "Naive"},
    "paradox": {"color": "#0284c7", "marker": "o", "display": "PARADOX"},
    "poisonedRAG": {"color": "#0891b2", "marker": "o", "display": "PoisonedRAG"},
    "topicattack": {"color": "#38bdf8", "marker": "o", "display": "Topic"},
}

LEGEND_ORDER = [
    "naive",
    "ignore",
    "completion",
    "completion+cmb",
    "topicattack",
    "poisonedRAG",
    "paradox",
]

FALLBACK_STYLES = [
    {"color": "#075985", "marker": "o"},
    {"color": "#0369a1", "marker": "o"},
    {"color": "#0284c7", "marker": "o"},
    {"color": "#0ea5e9", "marker": "o"},
]

OURS_STYLE = {
    "color": "#dc2626",
    "edgecolor": "#7f1d1d",
    "marker": "*",
    "display": "Ours",
}


@dataclass(frozen=True)
class PanelSpec:
    title: str
    detail_dir: str
    model_tag: str
    chain_mode: str


def strip_chain_mode_suffix(name: str) -> tuple[str, str]:
    if name.endswith("_thought_query"):
        return name[: -len("_thought_query")], "thought_query"
    if name.endswith("_query"):
        return name[: -len("_query")], "query"
    if name.endswith("_thought"):
        return name[: -len("_thought")], "thought_query"
    raise ValueError(
        f"Cannot infer chain mode from filename stem: {name}. "
        "Expected suffix: _query, _thought_query, or _thought."
    )


def infer_attack_name_and_mode_from_path(path: str) -> tuple[str, str]:
    base = os.path.basename(path)
    if not base.startswith("details_") or not base.endswith(".tsv"):
        raise ValueError(f"Unexpected detail filename: {base}")

    name = base[len("details_") : -len(".tsv")]
    return strip_chain_mode_suffix(name)


def remove_model_tag(name: str, model_tag: str) -> str:
    suffix = f"_{model_tag}"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return name


def normalize_attack_name(name: str, model_tag: str) -> str:
    name = remove_model_tag(name, model_tag)

    # ReAct filenames often include dataset/seed prefixes.
    for prefix in (
        "hotpotqa_seed1_",
        "2wikimultihopqa_seed1_",
        "musique_seed1_",
        "seed1_",
    ):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break

    name = name.replace("completion_realcmb", "completion+cmb")
    name = name.replace("completion_real", "completion")

    if name.startswith("ours"):
        return "ours"
    return name


def load_detail_files(detail_dir: str, chain_mode: str, model_tag: str) -> pd.DataFrame:
    pattern = os.path.join(detail_dir, "details_*.tsv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No detail files found with pattern: {pattern}")

    dfs = []
    skipped_wrong_mode = 0
    skipped_wrong_model = 0
    required_cols = [
        "path_divergence",
        "attack_target_over_correct_progress",
    ]

    for path in paths:
        attack_with_model, inferred_mode = infer_attack_name_and_mode_from_path(path)
        if inferred_mode != chain_mode:
            skipped_wrong_mode += 1
            continue
        if not attack_with_model.endswith(f"_{model_tag}"):
            skipped_wrong_model += 1
            continue

        df = pd.read_csv(path, sep="\t")
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df["attack_name"] = remove_model_tag(attack_with_model, model_tag)
        df["attack_label"] = normalize_attack_name(attack_with_model, model_tag)
        df["model_tag"] = model_tag
        df["chain_mode"] = inferred_mode
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No detail files found for chain_mode={chain_mode}, model_tag={model_tag} "
            f"in {detail_dir}. Skipped wrong mode: {skipped_wrong_mode}, "
            f"skipped wrong model: {skipped_wrong_model}"
        )

    return pd.concat(dfs, axis=0, ignore_index=True)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    summary = (
        details.groupby(["attack_name", "attack_label", "model_tag", "chain_mode"], as_index=False)
        .agg(
            num_examples=("path_divergence", "count"),
            path_divergence_mean=("path_divergence", "mean"),
            attack_target_over_correct_progress_mean=(
                "attack_target_over_correct_progress",
                "mean",
            ),
        )
    )

    summary["is_ours"] = summary["attack_label"].eq("ours")
    baselines = summary[~summary["is_ours"]].sort_values("path_divergence_mean")
    ours = summary[summary["is_ours"]]
    return pd.concat([baselines, ours], axis=0).reset_index(drop=True)


def load_panel(spec: PanelSpec) -> pd.DataFrame:
    details = load_detail_files(
        detail_dir=spec.detail_dir,
        chain_mode=spec.chain_mode,
        model_tag=spec.model_tag,
    )
    return summarize(details)


def get_attack_style(label: str) -> dict[str, str]:
    if label in ATTACK_STYLES:
        return ATTACK_STYLES[label]

    idx = abs(hash(label)) % len(FALLBACK_STYLES)
    return {
        **FALLBACK_STYLES[idx],
        "display": label,
    }


def compute_limits(summary: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = summary["path_divergence_mean"]
    ys = summary["attack_target_over_correct_progress_mean"]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0:
        x_min_pad = max(abs(x_min) * 0.02, 0.001)
        x_max_pad = max(abs(x_max) * 0.11, 0.005)
    else:
        x_min_pad = max(x_range * 0.04, 0.001)
        x_max_pad = max(x_range * 0.16, 0.005)

    if y_range == 0:
        y_min_pad = max(abs(y_min) * 0.02, 0.001)
        y_max_pad = max(abs(y_max) * 0.16, 0.008)
    else:
        y_min_pad = max(y_range * 0.04, 0.001)
        y_max_pad = max(y_range * 0.18, 0.008)

    return (x_min - x_min_pad, x_max + x_max_pad), (y_min - y_min_pad, y_max + y_max_pad)


def annotate_ours(ax, summary: pd.DataFrame) -> None:
    for _, row in summary[summary["is_ours"]].iterrows():
        ax.annotate(
            OURS_STYLE["display"],
            (
                row["path_divergence_mean"],
                row["attack_target_over_correct_progress_mean"],
            ),
            xytext=(0, 20),
            textcoords="offset points",
            fontsize=53,
            fontweight="bold",
            color=OURS_STYLE["edgecolor"],
            ha="center",
            va="bottom",
        )


def draw_panel(ax, summary: pd.DataFrame, title: str) -> None:
    baseline = summary[~summary["is_ours"]]
    ours = summary[summary["is_ours"]]

    for _, row in baseline.iterrows():
        style = get_attack_style(row["attack_label"])
        ax.scatter(
            row["path_divergence_mean"],
            row["attack_target_over_correct_progress_mean"],
            s=680,
            marker=style["marker"],
            c=style["color"],
            edgecolors="#111827",
            linewidths=1.4,
            alpha=0.92,
            zorder=3,
        )

    ax.scatter(
        ours["path_divergence_mean"],
        ours["attack_target_over_correct_progress_mean"],
        s=1500,
        marker=OURS_STYLE["marker"],
        c=OURS_STYLE["color"],
        edgecolors=OURS_STYLE["edgecolor"],
        linewidths=1.8,
        zorder=4,
    )
    annotate_ours(ax, summary)

    xlim, ylim = compute_limits(summary)
    ax.axhline(0.0, linestyle="--", linewidth=1.7, color="#475569", alpha=0.75, zorder=1)
    ax.set_title(title, fontsize=58, fontweight="bold", pad=30)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", linewidth=1.1, alpha=0.35)
    ax.tick_params(axis="both", labelsize=58, width=1.8, length=9)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)


def collect_legend_labels(panels: list[tuple[PanelSpec, pd.DataFrame]]) -> list[str]:
    seen = set()
    labels = []

    for _, summary in panels:
        for label in summary.loc[~summary["is_ours"], "attack_label"]:
            if label not in seen:
                seen.add(label)
                labels.append(label)

    preferred_order = LEGEND_ORDER
    return sorted(
        labels,
        key=lambda label: (
            preferred_order.index(label) if label in preferred_order else len(preferred_order),
            label,
        ),
    )


def make_legend_handles(labels: list[str]) -> list[Line2D]:
    handles = []
    for label in labels:
        style = get_attack_style(label)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markerfacecolor=style["color"],
                markeredgecolor="#111827",
                markeredgewidth=1.2,
                markersize=27,
                label=style["display"],
            )
        )

    handles.append(
        Line2D(
            [0],
            [0],
            marker=OURS_STYLE["marker"],
            linestyle="None",
            markerfacecolor=OURS_STYLE["color"],
            markeredgecolor=OURS_STYLE["edgecolor"],
            markeredgewidth=1.2,
            markersize=40,
            label=OURS_STYLE["display"],
        )
    )
    return handles


def plot_2x2(panels: list[tuple[PanelSpec, pd.DataFrame]], output_path: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 58,
            "axes.labelsize": 58,
            "axes.titlesize": 58,
            "xtick.labelsize": 58,
            "ytick.labelsize": 58,
            "legend.fontsize": 50,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(25.0, 22), sharex=False, sharey=False)

    for ax, (spec, summary) in zip(axes.flat, panels):
        draw_panel(ax, summary, spec.title)

    fig.supxlabel("Reasoning Path Divergence Score", fontsize=58, x=0.53, y=0.090)
    fig.supylabel("Target Redirection Score", fontsize=58, x=0.020, y=0.56)

    legend_handles = make_legend_handles(collect_legend_labels(panels))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.53, -0.030),
        handletextpad=0.7,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.19, wspace=0.14, hspace=0.26)
    for ax in axes[1, :]:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 - 0.012, pos.width, pos.height])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"
    fig.savefig(pdf_output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    print(f"Saved: {pdf_output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one readable 2x2 ROSCOE scatter PNG for qwen/llama/qwq/deepseek."
    )
    parser.add_argument(
        "--react-detail-dir",
        default="/home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa_target_path_divergence",
        help="Directory with ReAct details TSV files for qwen25 and llama70b.",
    )
    parser.add_argument(
        "--webthinker-detail-dir",
        default="/home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/target_path_divergence",
        help="Directory with WebThinker details TSV files for qwq and deepseek_32b.",
    )
    parser.add_argument("--chain-mode", default="thought_query", choices=["query", "thought_query"])
    parser.add_argument("--qwen-model-tag", default="qwen25")
    parser.add_argument("--llama-model-tag", default="llama70b")
    parser.add_argument("--qwq-model-tag", default="qwq")
    parser.add_argument("--deepseek-model-tag", default="deepseek_32b")
    parser.add_argument(
        "--output-path",
        default="/home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/target_path_divergence/roscoe_2x2_qwen_llama_qwq_deepseek.png",
    )
    args = parser.parse_args()

    specs = [
        PanelSpec("ReAct: Qwen", args.react_detail_dir, args.qwen_model_tag, args.chain_mode),
        PanelSpec("ReAct: Llama", args.react_detail_dir, args.llama_model_tag, args.chain_mode),
        PanelSpec("WebThinker: QwQ", args.webthinker_detail_dir, args.qwq_model_tag, args.chain_mode),
        PanelSpec("WebThinker: DeepSeek", args.webthinker_detail_dir, args.deepseek_model_tag, args.chain_mode),
    ]
    panels = [(spec, load_panel(spec)) for spec in specs]
    plot_2x2(panels, args.output_path)


if __name__ == "__main__":
    main()

"""
python /home/work/Redteaming/Rag_uni/scripts/ROSCOE_2x2_graph.py \
  --react-detail-dir /home/work/Redteaming/rag-exp/results/adv_targeted_results/hotpotqa_target_path_divergence \
  --webthinker-detail-dir /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/target_path_divergence \
  --chain-mode thought_query \
  --output-path /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/target_path_divergence/roscoe_2x2_qwen_llama_qwq_deepseek.png

  python /home/work/Redteaming/Rag_uni/scripts/ROSCOE_2x2_graph.py \
  --react-detail-dir /home/work/Redteaming/rag-exp/results/adv_targeted_results/musique_target_path_divergence \
  --webthinker-detail-dir /home/work/Redteaming/Rag_uni/results/webthinker/musique/target_path_divergence \
  --chain-mode thought_query \
  --output-path /home/work/Redteaming/Rag_uni/results/webthinker/musique/target_path_divergence/roscoe_2x2_qwen_llama_qwq_deepseek.png


python /home/work/Redteaming/Rag_uni/scripts/ROSCOE_2x2_graph.py \
  --react-detail-dir /home/work/Redteaming/rag-exp/results/adv_targeted_results/2wikimultihopqa_target_path_divergence \
  --webthinker-detail-dir /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/target_path_divergence \
  --chain-mode thought_query \
  --output-path /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/target_path_divergence/roscoe_2x2_qwen_llama_qwq_deepseek.png
"""
