import hydra

from hydra.core.config_store import ConfigStore
from config import LTDSConfig

cs = ConfigStore.instance()
cs.store(name="ltds_config", node=LTDSConfig)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: LTDSConfig):
    print(cfg)


if __name__ == "__main__":
    main()
