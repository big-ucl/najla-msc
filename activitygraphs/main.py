"""Hydra entry point: registers config store and dispatches to ``comparison_experiment``."""

import hydra
from hydra.core.config_store import ConfigStore

from activitygraphs.config import Config
from activitygraphs.run import comparison_experiment

cs = ConfigStore.instance()
cs.store(name="geneva_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    comparison_experiment(cfg)


if __name__ == "__main__":
    main()
