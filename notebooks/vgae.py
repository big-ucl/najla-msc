import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium", app_title="VGAE")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# VGAE for subgraph query completion""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import plotting
    return (plotting,)


@app.cell
def _():
    import numpy as np
    import polars as pl
    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Dataset generation

    Generate a synthetic graph, population and schedules.
    """
    )
    return


@app.cell
def _():
    from synthetic import SyntheticGraph

    synth = SyntheticGraph.example()
    return (synth,)


@app.cell
def _(plotting, synth):
    plotting.draw_synthetic_network(synth)
    return


@app.cell
def _(np, synth):
    from synthetic import SyntheticGenerator

    n_samples = 1000

    _rng = np.random.default_rng(seed=42)
    _generator = SyntheticGenerator(synth, _rng)
    _generator.generate_population(n_samples)
    _generator.generate_schedules()

    schedules = _generator.build()
    schedules
    return (schedules,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Conversion to PyG dataset""")
    return


@app.cell
def _(schedules):
    from datasets import convert_to_pyg_dataset

    dataset = convert_to_pyg_dataset(schedules)
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    from datasets import train_test_split

    train_set, test_set = train_test_split(dataset, random_state=42)
    train_set, test_set
    return


if __name__ == "__main__":
    app.run()
