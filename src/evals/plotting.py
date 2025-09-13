from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def plot_evaluation_bar(
    df: pd.DataFrame,
    combo_col: str | None = None,
    title: str = "Steering Evaluation",
    x_title: str = "Label Combination",
    y_title: str = "Average Probability",
    output_path: str | Path | None = None,
    width: int = 900,
    height: int = 500,
):
    if combo_col is None:
        combo_col = df.select_dtypes(include=["object", "category"]).columns[0]

    method_cols = [c for c in df.columns if c != combo_col]
    palette = ['#FF563F', '#F5C0B8',  '#55C89F', '#363432', '#F9DA81']
    if len(method_cols) > len(palette):
        repeats = -(-len(method_cols) // len(palette))
        palette *= repeats
    palette = palette[: len(method_cols)]

    fig = px.bar(
        df,
        x=combo_col,
        y=method_cols,
        color_discrete_sequence=palette,
        template="plotly_white",
        width=width,
        height=height,
    )
    fig.update_layout(
        title={"text": title, "x": 0.5, "y": 0.96, "xanchor": "center", "yanchor": "top"},
        barmode="group",
        margin={"l": 40, "r": 40, "t": 100, "b": 80},
        legend={"title": {"text": ""}, "orientation": "h", "y": 1.0, "x": 0.5, "xanchor": "center", "yanchor": "bottom"},
    )
    if output_path is not None:
        output_path = Path(output_path)
        fig.write_image(str(output_path))
    return fig



