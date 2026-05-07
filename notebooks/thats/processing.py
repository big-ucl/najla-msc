import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from activitygraphs.config import load_config
    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)
    cfg = load_config(project_root, data="toronto")


@app.cell
def _():
    CRS = "EPSG:4326"
    return


@app.cell
def _():
    import geopandas as gpd
    import polars as pl

    return (pl,)


@app.cell
def _():
    from activitygraphs.data.toronto import TorontoData

    return (TorontoData,)


@app.cell
def _(TorontoData):
    data = TorontoData.load(cfg.data, project_root)
    return (data,)


@app.cell
def _(data):
    data.user_journeys_df
    return


@app.cell
def _(data):
    data.locations_gdf.explore()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Exploratory Data Analysis
    """)
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _():
    from activitygraphs.base import Mode
    from activitygraphs.data.toronto import MANUAL_MODE_MAP, MODE_MAP

    return MANUAL_MODE_MAP, MODE_MAP, Mode


@app.cell
def _():
    import itertools

    return (itertools,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Trip modes

    First: investigate labeling behaviour
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Proportion of unlabeled trips
    """)
    return


@app.cell(hide_code=True)
def _(data):
    data.inputs.raw_journeys_df.select(
        "predicted_modes", "manual_mode"
    ).null_count() / len(data.inputs.raw_journeys_df)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    How often do people label their trips?
    """)
    return


@app.cell
def _(alt, data, pl):
    _mode_label_freq = data.inputs.raw_journeys_df.group_by("app_id").agg(pct_labeled=pl.col("manual_mode").is_not_null().sum() / pl.len())
    
    _chart = _mode_label_freq.plot.bar(x=alt.X("pct_labeled", bin=alt.Bin(maxbins=20)), y="count()").properties(title="Distribution of proportion of manually-labeled trip modes per user")
    _chart
    return


@app.cell(hide_code=True)
def _(alt, itertools, pl):
    def plot_enum_confusion_matrix(
        df: pl.DataFrame, pred_col: str, actual_col: str, enum, scale_type="linear"
    ) -> alt.Chart:
        base = pl.DataFrame(
            list(itertools.product(enum, enum)),
            orient="row",
            schema=[pred_col, actual_col],
        )

        modes = df.group_by(pred_col, actual_col).agg(count=pl.len())
        modes = base.join(modes, on=[pred_col, actual_col], how="left").fill_null(
            0
        )
        modes = modes.sort("predicted_modes", "manual_mode")

        labels = (
            alt
            .Chart(modes)
            .mark_text()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                text=alt.Text(
                    "count:Q",
                ),
            )
        )

        boxes = (
            alt
            .Chart(modes)
            .mark_rect()
            .encode(
                x=f"{pred_col}:N",
                y=f"{actual_col}:N",
                color=alt.Color(
                    "count:Q", scale=alt.Scale(type=scale_type, domainMin=1)
                ),
                tooltip=[pred_col, actual_col, "count"],
            )
        )

        return (boxes + labels).properties(
            width=400,
            height=400,
            title="Confusion matrix of GPS predicted vs manually-entered modes",
        )

    return (plot_enum_confusion_matrix,)


@app.cell(hide_code=True)
def _():
    drop_down_log_scale = mo.ui.dropdown(
        ["log", "linear"], label="Color scale", value="linear"
    )
    return (drop_down_log_scale,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Comparison between predicted mode and manually-entered mode. In genereal, it's somewhat confused.
    """)
    return


@app.cell(hide_code=True)
def _(
    MANUAL_MODE_MAP,
    MODE_MAP,
    Mode,
    data,
    drop_down_log_scale,
    pl,
    plot_enum_confusion_matrix,
):
    _mode_enum = [m for m in Mode if m not in [Mode.COACH]]
    _df = data.inputs.raw_journeys_df.select(
        pl.col("predicted_modes").replace_strict(MODE_MAP, default=Mode.UNKNOWN),
        pl.col("manual_mode").replace_strict(
            MANUAL_MODE_MAP, default=Mode.UNKNOWN
        ),
    )

    _chart = plot_enum_confusion_matrix(
        _df,
        "predicted_modes",
        "manual_mode",
        Mode,
        scale_type=drop_down_log_scale.value,
    )

    mo.vstack([drop_down_log_scale, _chart])
    return


@app.cell
def _():
    # Purposes (1-2 hrs max)

    # TODO Link purposes from activities to trips, then compare with accuracy from manually labeled trip purposes
    # TODO What proportion of unlabled trip purposes actually have corresponding data
    # TODO Rule finding ML algorithm (perhaps)
    return


if __name__ == "__main__":
    app.run()
