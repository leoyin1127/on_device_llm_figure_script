from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from radar_style import (
    color_pair,
    format_section_labels,
    legend_label,
    make_base_axes,
    place_section_labels,
)


DATA_PATH = Path(__file__).parent / "data" / "OSS Benchmarking Results - Eurorad.csv"
OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass(frozen=True)
class PlotSpec:
    key: str
    title: str
    description: str
    group_matchers: Dict[str, Sequence[str]]

    @property
    def png_name(self) -> str:
        return f"eurorad_radar_plot_{self.key}.png"

    @property
    def pdf_name(self) -> str:
        return f"eurorad_radar_plot_{self.key}.pdf"


PLOT_SPECS: Sequence[PlotSpec] = (
    PlotSpec(
        key="a",
        title="",
        description="All families with high-capacity OSS variants (H)",
        group_matchers={
            "gpt-5": ("gpt-5",),
            "gpt-o4-mini": ("o4-mini", "gpt-o4", "chatgpt-4o"),
            "DeepSeek-R1-0528": ("deepseek",),
            "Qwen3-235B": ("qwen3",),
            "gpt-oss-20B (H)": ("oss-20b (h)",),
            "gpt-oss-120B (H)": ("oss-120b (h)",),
        },
    ),
    PlotSpec(
        key="b",
        title="",
        description="OSS-120B ladder (L/M/H) compared with gpt-5 reference",
        group_matchers={
            "gpt-5": ("gpt-5",),
            "gpt-oss-120B (L)": ("oss-120b (l)",),
            "gpt-oss-120B (M)": ("oss-120b (m)",),
            "gpt-oss-120B (H)": ("oss-120b (h)",),
        },
    ),
    PlotSpec(
        key="c",
        title="",
        description="OSS-20B ladder (L/M/H) compared with gpt-5 reference",
        group_matchers={
            "gpt-5": ("gpt-5",),
            "gpt-oss-20B (L)": ("oss-20b (l)",),
            "gpt-oss-20B (M)": ("oss-20b (m)",),
            "gpt-oss-20B (H)": ("oss-20b (h)",),
        },
    ),
)


def load_accuracy_by_section(df: pd.DataFrame, model_columns: Iterable[str]) -> Dict[str, Dict[str, float]]:
    """Compute per-section accuracy for each model column."""

    accuracy: Dict[str, Dict[str, float]] = {}
    for section, section_df in df.groupby("Section", dropna=True, sort=False):
        section_accuracy: Dict[str, float] = {}
        for col in model_columns:
            correct = (section_df[col] == section_df["FinalDiagnosis"]).sum()
            total = len(section_df)
            section_accuracy[col] = (correct / total) * 100 if total else 0.0
        section_key = section if isinstance(section, str) else str(section)
        accuracy[section_key] = section_accuracy
    return accuracy


def build_model_groups(model_cols: Sequence[str], matchers: Dict[str, Sequence[str]]) -> Dict[str, List[str]]:
    """Assign columns to model groups based on substring matchers."""

    groups: Dict[str, List[str]] = {group: [] for group in matchers}
    for col in model_cols:
        lowered = col.lower()
        for group, patterns in matchers.items():
            if any(pattern in lowered for pattern in patterns):
                groups[group].append(col)
                break
    # Drop empty groups to avoid plotting zeroed polygons
    return {group: cols for group, cols in groups.items() if cols}


def average_accuracy_by_group(
    sections: Sequence[str],
    accuracy_by_section: Dict[str, Dict[str, float]],
    model_groups: Dict[str, List[str]],
) -> Dict[str, List[float]]:
    """Mean accuracy across models in each group per section."""

    averaged: Dict[str, List[float]] = {}
    for group, models in model_groups.items():
        section_values: List[float] = []
        for section in sections:
            section_scores = [
                accuracy_by_section.get(section, {}).get(model, 0.0) for model in models
            ]
            section_scores = [score for score in section_scores if score is not None]
            mean_score = float(np.mean(section_scores)) if section_scores else 0.0
            section_values.append(mean_score)
        averaged[group] = section_values
    return averaged


def render_plot(
    spec: PlotSpec,
    sections: Sequence[str],
    wrapped_sections: Sequence[str],
    accuracy_by_section: Dict[str, Dict[str, float]],
    model_cols: Sequence[str],
) -> None:
    """Produce a single radar plot and write PNG/PDF outputs."""

    model_groups = build_model_groups(model_cols, spec.group_matchers)
    if not model_groups:
        print(f"[warning] Plot {spec.key}: no matching model columns found; skipping.")
        return

    averaged = average_accuracy_by_group(sections, accuracy_by_section, model_groups)
    if not any(any(vals) for vals in averaged.values()):
        print(f"[warning] Plot {spec.key}: all averages are zero; skipping.")
        return

    all_scores = [
        score
        for section_scores in averaged.values()
        for score in section_scores
        if isinstance(score, (int, float))
    ]
    if not all_scores:
        print(f"[warning] Plot {spec.key}: no scores to plot; skipping.")
        return

    max_score = float(np.nanmax(all_scores))
    min_score = float(np.nanmin(all_scores))
    tick_start = max(0, int(np.floor(min_score / 5.0) * 5))
    tick_end = int(np.ceil(max_score / 5.0) * 5)
    radial_ticks = list(range(tick_start, tick_end + 5, 5))

    fig, ax, angles = make_base_axes(wrapped_sections, radial_tickvals=radial_ticks)
    place_section_labels(ax, angles, wrapped_sections)
    angles_closed = np.concatenate([angles, [angles[0]]])

    legend_seen: set[str] = set()
    for group_name, values in averaged.items():
        if not values or not any(values):
            continue

        values_array = np.asarray(values, dtype=float)
        values_closed = np.concatenate([values_array, [values_array[0]]])
        family = legend_label(group_name)
        line_color, fill_color = color_pair(family)

        ax.plot(
            angles_closed,
            values_closed,
            color=line_color,
            linewidth=3.0 if family == "gpt-5" else 2.4,
            linestyle="--" if family == "gpt-5" else "-",
            marker="o",
            markersize=6.4 if family == "gpt-5" else 5.6,
            markerfacecolor=line_color,
            markeredgecolor="#FFFFFF",
            markeredgewidth=1.3,
            alpha=0.98,
            label=family if family not in legend_seen else None,
            zorder=3,
        )
        ax.fill(
            angles_closed,
            values_closed,
            color=fill_color,
            zorder=1,
        )
        legend_seen.add(family)

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.02),
        frameon=True,
        fontsize=13,
        title="Model family",
        title_fontsize=13,
        borderpad=0.8,
        labelspacing=0.5,
        borderaxespad=0.0,
    )
    if legend:
        legend.get_frame().set_facecolor("#FFFFFF")
        legend.get_frame().set_edgecolor("#C4CBD3")
        legend.get_frame().set_linewidth(1.0)

    ax.set_title(spec.title, pad=32, fontsize=21, color="#1F2430")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.08)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / spec.png_name
    pdf_path = OUTPUT_DIR / spec.pdf_name
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"- {spec.key.upper()} saved: {png_path.name}, {pdf_path.name}")

    # Summary output in console
    for group_name, values in averaged.items():
        if not values:
            continue
        family = legend_label(group_name)
        mean_score = float(np.mean(values))
        print(f"    {family:>18}: {mean_score:5.2f}%")

    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    metadata_cols = {
        "case_id",
        "Section",
        "OriginalDescription",
        "PostDescription",
        "DifferentialDiagnosisList",
        "FinalDiagnosis",
        "DiseaseLeak",
    }
    model_cols = [col for col in df.columns if col not in metadata_cols]

    accuracy_by_section = load_accuracy_by_section(df, model_cols)
    sections = list(accuracy_by_section.keys())
    wrapped_sections = format_section_labels(sections, line_break="\n")

    print("\n=== Eurorad radar plots ===")
    print(f"Sections analysed: {len(sections)} • Models detected: {len(model_cols)}\n")

    for spec in PLOT_SPECS:
        print(f"{spec.key.upper()}. {spec.description}")
        render_plot(spec, sections, wrapped_sections, accuracy_by_section, model_cols)
        print()


if __name__ == "__main__":
    main()
