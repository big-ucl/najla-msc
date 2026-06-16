"""
Module: activitygraphs/archive/exploration/main.py

Description:
    Entry point for the interactive graph-exploration dashboard.

    Loads a processed ActivityDataset and pre-computed graph metric results from
    disk, then launches a Plotly Dash web application that renders an interactive
    scatter plot of graph metrics so that individual household activity graphs can
    be inspected visually.

    Configuration is managed by Hydra (a framework that reads YAML config files
    from the `conf/` directory). The config path and name are set via the
    @hydra.main decorator below.

    Usage (from the project root):
        python -m archive.exploration.main

    The Dash server will start in debug mode and is accessible at
    http://127.0.0.1:8050 by default.
"""

import hydra
import polars as pl
from config import Config
from archive.exploration.dataprocessing import ActivityDataset
from archive.exploration.graphs import ActivityGraph
from hydra.core.config_store import ConfigStore
from plotting import build_dash_graph_scatter

# Register the Config dataclass with Hydra's ConfigStore so that config values
# are automatically validated and type-checked against the Config schema on load.
cs = ConfigStore.instance()
cs.store(name="ltds_config", node=Config)  # "ltds_config" matches the config file name in conf/


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    """
    Description:
        Main entry point decorated with @hydra.main so that Hydra automatically
        reads the YAML configuration file and passes it as a typed Config object.

        Loads the processed dataset and pre-computed graph metrics, then launches
        an interactive Dash scatter-plot for exploring the metric space.

    Input:
      - cfg (Config): Hydra-injected configuration object. Key fields used:
            - cfg.data.paths.act_dataset (str): path to the saved ActivityDataset directory.
            - cfg.data.name (str): dataset name used for sub-directory and file prefixes.

    Output:
      - None (starts a blocking Dash web server; press Ctrl+C to stop).
    """
    # Load the cleaned travel-survey dataset from its saved Parquet files
    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)

    # Build the activity graph (node + edge DataFrames) from the loaded dataset
    graph = ActivityGraph.from_dataset(dataset)

    # Load pre-computed graph metrics from a Parquet file for visualisation
    results = pl.read_parquet("data/processed/TEST_graphs.parquet")

    # Downsample to 1000 rows if the results are large, for faster rendering
    if len(results) > 1000:
        results = results.sample(1000, seed=42)  # seed=42 ensures reproducible sampling

    # Build and launch the Dash scatter-plot dashboard in debug mode
    build_dash_graph_scatter(results, graph).run(debug=True)


if __name__ == "__main__":
    main()
