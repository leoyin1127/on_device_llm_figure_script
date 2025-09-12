from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import plotly.graph_objects as go


FONT_FAMILY = "Times New Roman"


"""Legend and color helpers for consistent plotting.

Note: Section labels should be used exactly as they appear in the CSV.
No abbreviation or wrapping helpers are provided here on purpose.
"""


# Consistent palette across figures
LINE_COLORS: Dict[str, str] = {
    # Baselines / families
    "gpt-5": "#d62728",
    "GPT-5": "#d62728",
    "gpt-o4": "#1f77b4",
    "gpt-o4-mini": "#1f77b4",
    "ChatGPT-4o": "#1f77b4",
    "DeepSeek-R1-0528": "#e377c2",
    "DeepSeek": "#e377c2",
    "Qwen3-235B": "#2ca02c",
    "Qwen3": "#2ca02c",
    # OSS-20B family (warm oranges)
    "gpt-oss-20B": "#ff6e40",
    "gpt-oss-20B (L)": "#ffb347",
    "gpt-oss-20B (M)": "#ffd166",
    "gpt-oss-20B (H)": "#ff6e40",
    "OSS-20B (L)": "#ffb347",
    "OSS-20B (M)": "#ffd166",
    "OSS-20B (H)": "#ff6e40",
    # OSS-120B family (purples)
    "gpt-oss-120B": "#9400d3",
    "gpt-oss-120B (L)": "#4b0082",
    "gpt-oss-120B (M)": "#8a2be2",
    "gpt-oss-120B (H)": "#9400d3",
    "OSS-120B (L)": "#4b0082",
    "OSS-120B (M)": "#8a2be2",
    "OSS-120B (H)": "#9400d3",
}


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def color_pair(name: str, fill_alpha: float = 0.15) -> Tuple[str, str]:
    line = LINE_COLORS.get(name, "#7f7f7f")
    return line, _rgba(line, fill_alpha)


def legend_label(name: str) -> str:
    """Normalize model names to standard legend labels, preserving L/M/H.

    Standard families:
      - gpt-5
      - gpt-o4 (covers ChatGPT-4o, gpt-o4-mini, etc.)
      - DeepSeek-R1-0528
      - Qwen3-235B
      - gpt-oss-20B (+ optional (L|M|H))
      - gpt-oss-120B (+ optional (L|M|H))
    """
    key = name.lower()
    # Variants (keep the suffix if present)
    variant = None
    if "(l)" in key:
        variant = " (L)"
    elif "(m)" in key:
        variant = " (M)"
    elif "(h)" in key:
        variant = " (H)"

    if "gpt-5" in key:
        return "gpt-5"
    if "chatgpt-4o" in key or "chatgpt4o" in key or "o4" in key:
        return "gpt-o4"
    if "deepseek" in key:
        return "DeepSeek-R1-0528"
    if "qwen3" in key:
        return "Qwen3-235B"
    if "oss-20b" in key:
        return f"gpt-oss-20B{variant or ''}"
    if "oss-120b" in key:
        return f"gpt-oss-120B{variant or ''}"
    return name


def make_base_figure(theta_labels: List[str], width: int = 1900, height: int = 1200) -> go.Figure:
    """Create a standardized polar figure with consistent layout.

    - Legend space on the right; domain set to avoid overlap
    - Ticks at 0..100
    - Wrapped angular tick labels
    """
    fig = go.Figure(
        layout=go.Layout(
            template="plotly_white",
            width=width,
            height=height,
            # Wider right margin for legend; compact left
            margin=dict(l=100, r=250, t=80, b=90),
            font=dict(family=FONT_FAMILY, size=16, color="#000"),
            # Place legend to the right, fully outside the polar domain
            legend=dict(
                orientation="v",
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#eee",
                borderwidth=1,
                font=dict(size=11),
                tracegroupgap=8,
            ),
            polar=dict(
                # Wider polar domain; legend sits in reserved right margin
                domain=dict(x=[0.02, 0.75], y=[0.02, 0.98]),
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(
                    showline=False,
                    ticks="",
                    gridcolor="rgba(0,0,0,0.15)",
                    tickvals=[0, 20, 40, 60, 80, 100],
                    range=[0, 100],
                    tickfont=dict(size=11),
                    angle=90,
                ),
                angularaxis=dict(
                    rotation=90,
                    direction="clockwise",
                    gridcolor="rgba(0,0,0,0.15)",
                    tickfont=dict(size=12),
                    showticklabels=True,
                    categoryorder="array",
                    categoryarray=theta_labels,
                ),
            ),
        )
    )
    return fig


def add_angular_labels(fig: go.Figure, labels: List[str], radius: float = 104, size: int = 16) -> None:
    # No-op now; kept for potential future use without breaking imports
    return None
