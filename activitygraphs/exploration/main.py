import hydra
import polars as pl
from config import Config
from exploration.dataprocessing import ActivityDataset
from exploration.graphs import ActivityGraph
from hydra.core.config_store import ConfigStore
from plotting import build_dash_graph_scatter

cs = ConfigStore.instance()
cs.store(name="ltds_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    dataset = ActivityDataset.load(cfg.data.paths.act_dataset, cfg.data.name)
    graph = ActivityGraph.from_dataset(dataset)

    results = pl.read_parquet("data/processed/TEST_graphs.parquet")

    if len(results) > 1000:
        results = results.sample(1000, seed=42)

    build_dash_graph_scatter(results, graph).run(debug=True)


if __name__ == "__main__":
    main()
