import hydra
from hydra.core.config_store import ConfigStore

from activitygraphs.config import Config
from activitygraphs.run import geneva_experiment

cs = ConfigStore.instance()
cs.store(name="geneva_config", node=Config)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: Config):
    geneva_experiment(cfg)


if __name__ == "__main__":
    main()
