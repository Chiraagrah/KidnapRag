#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Draw target/correct score curves using the compact Figure-2 design, but with
previously remembered plot data semantics:

x-axis:
    offsets [-1, 0, 1, 2, 3, 4, 5] -> [Prev, B, C1, C2, C3, C4, M]

y-axis for --metric-mode score:
    target_y  = mean(target_score)  per offset, after optional prediction filter
    correct_y = mean(correct_score) per offset, after optional prediction filter

This script does NOT re-detect trigger steps. It uses the offset field already
written in each step_probability_results item.
"""

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


BASE_COLOR = "#7A7A7A"       # Correct
HIGHLIGHT_COLOR = "#E24A33"  # Target
SECONDARY_COLOR = "#4C72B0"
DEFAULT_WINDOW_OFFSETS = [-1, 0, 1, 2, 3, 4, 5]


# =============================================================================
# Basic helpers
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        f = float(value)
        if np.isnan(f):
            return None
        return f
    except Exception:
        return None


def mean_or_nan(values: List[float]) -> float:
    clean = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    return float(np.mean(clean)) if clean else float("nan")


def parse_offsets(offsets: str) -> List[int]:
    parsed = [int(x.strip()) for x in offsets.split(",") if x.strip()]
    if not parsed:
        raise ValueError("--window-offsets produced an empty list.")
    return parsed


def make_window_labels(offsets: List[int]) -> List[str]:
    labels: List[str] = []
    max_offset = max(offsets)
    for offset in offsets:
        if offset == -1:
            labels.append("Pre")
        elif offset == 0:
            labels.append("B")
        elif offset == 5:
            labels.append("M")
        elif offset > 0:
            labels.append(f"C{offset}")

        else:
            labels.append(f"Pre{abs(offset)}")
    return labels


def load_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        raise ValueError(f"Empty file: {path}")

    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            if isinstance(obj.get("data"), list):
                return obj["data"]
            return [obj]
    except json.JSONDecodeError:
        pass

    records: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def get_item_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("type") in {"summary", "error"}:
            continue
        if "error" in r:
            continue
        if "step_probability_results" in r:
            items.append(r)
    return items


def prediction_matches_filter(item: Dict[str, Any], prediction_filter: Optional[str], case_sensitive: bool) -> bool:
    if not prediction_filter:
        return True
    prediction = str(item.get("prediction", ""))
    if case_sensitive:
        return prediction_filter in prediction
    return prediction_filter.lower() in prediction.lower()


def resolve_input_files(input_files: Optional[List[str]], input_dir: Optional[str], filename_glob: str) -> List[str]:
    files: List[str] = []
    if input_files:
        for item in input_files:
            expanded = sorted(glob.glob(item))
            files.extend(expanded if expanded else [item])
    if input_dir:
        files.extend(sorted(glob.glob(os.path.join(input_dir, filename_glob))))

    unique: List[str] = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def infer_panel_title(path: str) -> str:
    name = os.path.basename(path)
    name = os.path.splitext(name)[0]
    name = name.replace("ours납치", "Hijack ")
    name = name.replace("회_", " / ")
    name = name.replace("_", " ")
    return name


# =============================================================================
# Metric extraction using remembered data semantics
# =============================================================================

def get_step_values_for_metric_mode(step: Dict[str, Any], metric_mode: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return correct_value, target_value, single_value."""
    if metric_mode in {"score", "step_potential"}:
        return (
            safe_float(step.get("correct_step_potential", step.get("correct_score"))),
            safe_float(step.get("target_step_potential", step.get("target_score"))),
            None,
        )

    if metric_mode == "pairwise":
        return (
            safe_float(step.get("correct_pairwise_preference")),
            safe_float(step.get("target_pairwise_preference")),
            None,
        )

    if metric_mode in {"score_margin", "step_potential_margin"}:
        return (
            None,
            None,
            safe_float(step.get("target_minus_correct_step_potential", step.get("target_minus_correct_score"))),
        )

    if metric_mode == "margin":
        return (None, None, safe_float(step.get("target_minus_correct_pairwise_preference")))

    if metric_mode == "nll_margin":
        return (None, None, safe_float(step.get("log_likelihood_margin_target_over_correct")))

    raise ValueError(f"Unsupported metric_mode: {metric_mode}")


def collect_offset_values(
    records: List[Dict[str, Any]],
    window_offsets: List[int],
    metric_mode: str,
    prediction_filter: Optional[str],
    prediction_filter_case_sensitive: bool,
    require_full_window: bool,
) -> Tuple[Dict[int, List[float]], Dict[int, List[float]], Dict[int, List[float]], Dict[str, Any]]:
    item_records = get_item_records(records)
    before_filter_count = len(item_records)
    item_records = [
        item for item in item_records
        if prediction_matches_filter(item, prediction_filter, prediction_filter_case_sensitive)
    ]

    target_by_offset = {offset: [] for offset in window_offsets}
    correct_by_offset = {offset: [] for offset in window_offsets}
    single_by_offset = {offset: [] for offset in window_offsets}

    required_offsets = set(window_offsets)
    used_count = 0
    missing_full_window_count = 0
    skipped_incomplete_window_count = 0

    for item in item_records:
        steps = item.get("step_probability_results", [])
        if not isinstance(steps, list) or not steps:
            continue

        step_by_offset: Dict[int, Dict[str, Any]] = {}
        for step in steps:
            if "offset" not in step:
                continue
            try:
                offset = int(step["offset"])
            except Exception:
                continue
            step_by_offset[offset] = step

        has_full_window = required_offsets.issubset(set(step_by_offset.keys()))
        if not has_full_window:
            missing_full_window_count += 1
            if require_full_window:
                skipped_incomplete_window_count += 1
                continue

        has_any_value = False
        for offset in window_offsets:
            step = step_by_offset.get(offset)
            if step is None:
                continue
            correct_value, target_value, single_value = get_step_values_for_metric_mode(step, metric_mode)
            if correct_value is not None:
                correct_by_offset[offset].append(correct_value)
                has_any_value = True
            if target_value is not None:
                target_by_offset[offset].append(target_value)
                has_any_value = True
            if single_value is not None:
                single_by_offset[offset].append(single_value)
                has_any_value = True

        if has_any_value:
            used_count += 1

    stats = {
        "num_item_records_before_prediction_filter": before_filter_count,
        "num_item_records_after_prediction_filter": len(item_records),
        "prediction_filter": prediction_filter,
        "prediction_filter_case_sensitive": prediction_filter_case_sensitive,
        "num_items_used": used_count,
        "num_items_missing_full_window": missing_full_window_count,
        "num_items_skipped_incomplete_window": skipped_incomplete_window_count,
        "require_full_window": require_full_window,
    }
    return target_by_offset, correct_by_offset, single_by_offset, stats


def load_window_series_for_file(
    path: str,
    window_offsets: List[int],
    metric_mode: str,
    prediction_filter: Optional[str],
    prediction_filter_case_sensitive: bool,
    require_full_window: bool,
) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    records = load_records(path)
    target_by_offset, correct_by_offset, single_by_offset, stats = collect_offset_values(
        records=records,
        window_offsets=window_offsets,
        metric_mode=metric_mode,
        prediction_filter=prediction_filter,
        prediction_filter_case_sensitive=prediction_filter_case_sensitive,
        require_full_window=require_full_window,
    )

    target_means = [mean_or_nan(target_by_offset[offset]) for offset in window_offsets]
    correct_means = [mean_or_nan(correct_by_offset[offset]) for offset in window_offsets]
    single_means = [mean_or_nan(single_by_offset[offset]) for offset in window_offsets]

    two_line = metric_mode in {"score", "step_potential", "pairwise"}
    if two_line:
        no_values = all(np.isnan(v) for v in target_means) and all(np.isnan(v) for v in correct_means)
    else:
        no_values = all(np.isnan(v) for v in single_means)
    if no_values:
        raise ValueError(f"No plottable values found in {path}. metric_mode={metric_mode!r}")

    return {
        "input_file": path,
        "window_offsets": window_offsets,
        "window_labels": make_window_labels(window_offsets),
        "target_by_offset": target_by_offset,
        "correct_by_offset": correct_by_offset,
        "single_by_offset": single_by_offset,
        "target_means": target_means,
        "correct_means": correct_means,
        "single_means": single_means,
        "stats": stats,
    }


# =============================================================================
# Figure-2 design plotting
# =============================================================================

def annotate_delta_signs(ax: Any, x_values: List[Any], y_values: List[float], y_offset: float, color: str, fontsize: float = 6.0) -> None:
    previous_value: Optional[float] = None
    for x_value, y_value in zip(x_values, y_values):
        if np.isnan(y_value):
            continue
        if previous_value is not None:
            if y_value > previous_value:
                sign = "+"
            elif y_value < previous_value:
                sign = "-"
            else:
                sign = ""
            if sign:
                ax.text(
                    x_value,
                    y_value + y_offset,
                    sign,
                    ha="center",
                    va="bottom",
                    fontsize=fontsize,
                    fontweight="bold",
                    color=color,
                    clip_on=False,
                )
        previous_value = y_value


def get_ylabel(metric_mode: str) -> str:
    if metric_mode in {"score", "step_potential"}:
        return "Answer Preference Score"
    if metric_mode == "pairwise":
        return "Pairwise Finish Preference"
    if metric_mode in {"score_margin", "step_potential_margin"}:
        return "Target - Correct Score"
    if metric_mode == "margin":
        return "Target - Correct Pairwise Preference"
    if metric_mode == "nll_margin":
        return "Target-over-Correct Log-Likelihood Margin"
    return metric_mode


def plot_panels_with_figure2_design(
    panels: List[Tuple[str, Dict[str, Any]]],
    metric_mode: str,
    save_path: str,
    pdf_save_path: Optional[str],
    annotate_deltas: bool,
) -> Dict[str, Any]:
    if not panels:
        raise ValueError("No panels to plot.")

    base_font_size = 11
    title_font_size = 10
    label_font_size = 10
    tick_font_size = 10
    legend_font_size = 10
    font_family = "DejaVu Sans"

    two_line = metric_mode in {"score", "step_potential", "pairwise"}
    figsize = (7.2, 2.42) if len(panels) == 4 else (max(3.2, 1.8 * len(panels)), 2.42)

    plt.rcParams.update({"font.family": font_family, "font.size": base_font_size})
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharey=False)
    if len(panels) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None
    panel_axis_ranges: List[Dict[str, Any]] = []

    for ax, (title, panel_data) in zip(axes, panels):
        x_labels = panel_data["window_labels"]

        if two_line:
            plotted_groups = [panel_data["target_means"], panel_data["correct_means"]]
        else:
            plotted_groups = [panel_data["single_means"]]

        panel_y_values = [value for values in plotted_groups for value in values if not np.isnan(value)]
        if not panel_y_values:
            raise ValueError(f"No valid y-values found for panel: {title}")

        panel_y_min = min(panel_y_values)
        panel_y_max = max(panel_y_values)
        panel_y_padding = (panel_y_max - panel_y_min) * 0.10 if panel_y_max > panel_y_min else max(abs(panel_y_max) * 0.16, 0.05)
        panel_y_axis_min = panel_y_min - panel_y_padding
        panel_y_axis_max = panel_y_max + panel_y_padding
        panel_y_annotation_offset = panel_y_padding * 0.12
        panel_y_ticks = [panel_y_axis_min, (panel_y_axis_min + panel_y_axis_max) / 2.0, panel_y_axis_max]

        panel_axis_ranges.append({
            "title": title,
            "y_min": panel_y_min,
            "y_max": panel_y_max,
            "y_axis_min": panel_y_axis_min,
            "y_axis_max": panel_y_axis_max,
            "y_ticks": panel_y_ticks,
        })

        if two_line:
            target_line, = ax.plot(
                x_labels,
                panel_data["target_means"],
                marker="o",
                color=HIGHLIGHT_COLOR,
                linewidth=1.4,
                markersize=3.5,
                label="Target",
            )
            correct_line, = ax.plot(
                x_labels,
                panel_data["correct_means"],
                marker="o",
                color=BASE_COLOR,
                linewidth=1.4,
                markersize=3.5,
                label="Correct",
            )
            if annotate_deltas:
                annotate_delta_signs(ax, x_labels, panel_data["target_means"], panel_y_annotation_offset, HIGHLIGHT_COLOR, fontsize=8)
                annotate_delta_signs(ax, x_labels, panel_data["correct_means"], panel_y_annotation_offset, BASE_COLOR, fontsize=8)
            if legend_handles is None:
                legend_handles = [target_line, correct_line]
                legend_labels = ["Target", "Correct"]
        else:
            single_line, = ax.plot(
                x_labels,
                panel_data["single_means"],
                marker="o",
                color=SECONDARY_COLOR,
                linewidth=1.4,
                markersize=3.5,
                label=metric_mode,
            )
            ax.axhline(y=0.0, color="#B0B0B0", linewidth=0.8, linestyle="--")
            if annotate_deltas:
                annotate_delta_signs(ax, x_labels, panel_data["single_means"], panel_y_annotation_offset, SECONDARY_COLOR, fontsize=8)
            if legend_handles is None:
                legend_handles = [single_line]
                legend_labels = [metric_mode]

        ax.set_title(title, fontsize=title_font_size, fontweight="bold", pad=8)
        ax.tick_params(axis="x", labelsize=tick_font_size, width=0.7, length=2.5, pad=1)
        ax.tick_params(axis="y", labelsize=tick_font_size, width=0.7, length=2.5, pad=1)
        ax.set_ylim(panel_y_axis_min, panel_y_axis_max)
        ax.set_yticks(panel_y_ticks)
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.supxlabel("Reasoning Steps", fontsize=label_font_size, fontfamily=font_family, x=0.54, y=0.12) ##여기
    fig.supylabel(get_ylabel(metric_mode), fontsize=label_font_size, fontfamily=font_family, x=0.015)

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="center right",          # 변경
            ncol=len(legend_handles),
            fontsize=legend_font_size,
            bbox_to_anchor=(1.02, 0.15), # 변경: Reasoning Steps와 같은 y
            handlelength=1.1,
            columnspacing=1.2,
            handletextpad=0.4,
        )

    fig.subplots_adjust(left=0.09, right=0.995, top=0.8, bottom=0.29, wspace=0.35)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    if pdf_save_path is not None:
        fig.savefig(pdf_save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[SAVE] Figure: {save_path}")
    if pdf_save_path is not None:
        print(f"[SAVE] PDF: {pdf_save_path}")

    for panel_name, panel_data in panels:
        print(f"\n{panel_name}")
        for label, offset, target_mean, correct_mean, single_mean in zip(
            panel_data["window_labels"],
            panel_data["window_offsets"],
            panel_data["target_means"],
            panel_data["correct_means"],
            panel_data["single_means"],
        ):
            print(
                f"{label:>8} offset={offset:>3}: "
                f"target_mean={target_mean}, correct_mean={correct_mean}, single_mean={single_mean}, "
                f"target_count={len(panel_data['target_by_offset'][offset])}, "
                f"correct_count={len(panel_data['correct_by_offset'][offset])}, "
                f"single_count={len(panel_data['single_by_offset'][offset])}"
            )
        print("Window stats:")
        for k, v in panel_data["stats"].items():
            print(f"  {k}: {v}")

    return {
        "figure": "remembered_offset_score_figure2_design",
        "metric_mode": metric_mode,
        "save_path": save_path,
        "pdf_save_path": pdf_save_path,
        "panel_axis_ranges": panel_axis_ranges,
        "panels": [
            {
                "title": title,
                "input_file": panel_data["input_file"],
                "window_offsets": panel_data["window_offsets"],
                "window_labels": panel_data["window_labels"],
                "target_means": panel_data["target_means"],
                "correct_means": panel_data["correct_means"],
                "single_means": panel_data["single_means"],
                "stats": panel_data["stats"],
            }
            for title, panel_data in panels
        ],
    }


def write_plot_data_json(save_path: str, plot_data: Dict[str, Any]) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(plot_data, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Plot data: {save_path}")


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw remembered target/correct offset score data using the compact Figure-2 design."
    )
    parser.add_argument("--input-files", nargs="*", default=None)
    parser.add_argument("--input-dir", type=str, default=None)
    parser.add_argument("--filename-glob", type=str, default="*.jsonl")
    parser.add_argument("--panel-titles", nargs="*", default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--figure-png", type=str, default="finish_answer_preference_score_kill_only.png")
    parser.add_argument("--figure-pdf", type=str, default=None)
    parser.add_argument("--plot-data-json", type=str, default="finish_answer_preference_score_kill_only_plot_data.json")
    parser.add_argument("--metric-mode", type=str, default="score", choices=["score", "step_potential", "pairwise", "score_margin", "step_potential_margin", "margin", "nll_margin"])
    parser.add_argument("--prediction-filter", type=str, default="kill yourself")
    parser.add_argument("--prediction-filter-case-sensitive", action="store_true")
    parser.add_argument("--window-offsets", type=str, default=",".join(str(x) for x in DEFAULT_WINDOW_OFFSETS))
    parser.add_argument("--allow-incomplete-window", action="store_true")
    parser.add_argument("--annotate-deltas", action="store_true", help="Add +/- annotations between consecutive points, as in the Figure-2 design.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    input_files = resolve_input_files(args.input_files, args.input_dir, args.filename_glob)
    if not input_files:
        raise ValueError("No input files resolved. Use --input-files or --input-dir.")

    if args.panel_titles is not None and len(args.panel_titles) != len(input_files):
        raise ValueError(f"--panel-titles count ({len(args.panel_titles)}) must match number of input files ({len(input_files)}).")

    window_offsets = parse_offsets(args.window_offsets)
    require_full_window = not args.allow_incomplete_window

    figure_png_path = os.path.join(args.output_dir, args.figure_png)
    figure_pdf = os.path.splitext(args.figure_png)[0] + ".pdf" if args.figure_pdf is None else args.figure_pdf
    figure_pdf_path = os.path.join(args.output_dir, figure_pdf)
    plot_data_path = os.path.join(args.output_dir, args.plot_data_json)

    print("=" * 100)
    print("Configuration")
    print("=" * 100)
    print(f"input_files         : {input_files}")
    print(f"output_dir          : {args.output_dir}")
    print(f"metric_mode         : {args.metric_mode}")
    print(f"prediction_filter   : {args.prediction_filter}")
    print(f"window_offsets      : {window_offsets}")
    print(f"window_labels       : {make_window_labels(window_offsets)}")
    print(f"require_full_window : {require_full_window}")
    print("Data policy         : use offset field directly; no trigger re-detection")
    print("Y policy            : score -> mean(target_score), mean(correct_score) by offset")

    panels: List[Tuple[str, Dict[str, Any]]] = []
    skipped_files: List[Dict[str, str]] = []
    for i, path in enumerate(input_files):
        title = args.panel_titles[i] if args.panel_titles is not None else infer_panel_title(path)
        try:
            panel_data = load_window_series_for_file(
                path=path,
                window_offsets=window_offsets,
                metric_mode=args.metric_mode,
                prediction_filter=args.prediction_filter,
                prediction_filter_case_sensitive=args.prediction_filter_case_sensitive,
                require_full_window=require_full_window,
            )
            panels.append((title, panel_data))
        except Exception as e:
            skipped_files.append({"title": title, "path": path, "reason": repr(e)})
            print(f"[SKIP] failed to load {path}: {repr(e)}")

    if not panels:
        raise ValueError("No valid input files remained after skipping missing/invalid files.")

    plot_data = plot_panels_with_figure2_design(
        panels=panels,
        metric_mode=args.metric_mode,
        save_path=figure_png_path,
        pdf_save_path=figure_pdf_path,
        annotate_deltas=args.annotate_deltas,
    )
    plot_data["skipped_files"] = skipped_files
    write_plot_data_json(plot_data_path, plot_data)

    print("\nDone.")
    print("Saved:")
    print(f"  {figure_png_path}")
    print(f"  {figure_pdf_path}")
    print(f"  {plot_data_path}")


if __name__ == "__main__":
    main()

"""
python /home/work/Redteaming/Rag_uni/scripts/step_potential_draw_graph_chanwoo.py \
  --input-files \
    /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/step_preference/finish_answer_preference_qwen25.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/step_preference/finish_answer_preference_llama70b.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/step_preference/finish_answer_preference_qwq.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/step_preference/finish_answer_preference_deepseek_32b.jsonl \
  --panel-titles "ReAct: Qwen" "ReAct: Llama" "WebThinker: QwQ" "WebThinker: DeepSeek" \
  --output-dir /home/work/Redteaming/Rag_uni/results/webthinker/hotpotqa/step_preference/finish_answer_preference_graph \
  --figure-png finish_answer_preference_score_kill_only.png \
  --metric-mode score \
  --prediction-filter "kill yourself" \
  --window-offsets=-1,0,1,2,3,4

  
  
python /home/work/Redteaming/Rag_uni/scripts/step_potential_draw_graph_chanwoo.py \
  --input-files \
    /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/step_preference/finish_answer_preference_qwen25.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/step_preference/finish_answer_preference_llama70b.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/step_preference/finish_answer_preference_qwq.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/step_preference/finish_answer_preference_deepseek_32b.jsonl \
  --panel-titles "ReAct: Qwen" "ReAct: Llama" "WebThinker: QwQ" "WebThinker: DeepSeek" \
  --output-dir /home/work/Redteaming/Rag_uni/results/webthinker/2wikimultihopqa/step_preference/finish_answer_preference_graph \
  --figure-png finish_answer_preference_score_kill_only.png \
  --metric-mode score \
  --prediction-filter "kill yourself" \
  --window-offsets=-1,0,1,2,3,4

python /home/work/Redteaming/Rag_uni/scripts/step_potential_draw_graph_chanwoo.py \
  --input-files \
    /home/work/Redteaming/Rag_uni/results/webthinker/musique/step_preference/finish_answer_preference_qwen25.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/musique/step_preference/finish_answer_preference_llama70b.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/musique/step_preference/finish_answer_preference_qwq.jsonl \
    /home/work/Redteaming/Rag_uni/results/webthinker/musique/step_preference/finish_answer_preference_deepseek_32b.jsonl \
  --panel-titles "ReAct: Qwen" "ReAct: Llama" "WebThinker: QwQ" "WebThinker: DeepSeek" \
  --output-dir /home/work/Redteaming/Rag_uni/results/webthinker/musique/step_preference/finish_answer_preference_graph \
  --figure-png finish_answer_preference_score_kill_only.png \
  --metric-mode score \
  --prediction-filter "kill yourself" \
  --window-offsets=-1,0,1,2,3,4



"""