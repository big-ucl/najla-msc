import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import pprint

    import marimo as mo
    import polars as pl

    from pathlib import Path
    return mo, pl


@app.cell
def _(mo):
    from hydra import initialize_config_dir, compose
    from hydra.core.config_store import ConfigStore
    from config import LTDSConfig

    cs = ConfigStore.instance()
    cs.store(name="ltds_config", node=LTDSConfig)

    project_root = mo.notebook_dir().parent

    def load_config() -> LTDSConfig:
        _config_dir = str(project_root / "activitygraphs/conf")

        with initialize_config_dir(version_base=None, config_dir=_config_dir):
            return compose(config_name="config")


    cfg = load_config()
    print(f"Configuration loaded: {cfg}")
    return cfg, project_root


@app.cell
def _(cfg, pl, project_root):
    _path = project_root / cfg.paths.data_raw_ltds / cfg.files.raw_household

    pl.read_excel(_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
