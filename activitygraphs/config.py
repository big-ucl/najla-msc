from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore


@dataclass
class Files:
    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str

@dataclass
class Paths:
    data_processed: Path
    data_raw: Path
    data_raw_ltds: Path

@dataclass
class LTDSConfig:
    files: Files
    paths: Paths

cs = ConfigStore.instance()
cs.store(name="ltds_config", node=LTDSConfig)

def load_config(project_root: Path) -> LTDSConfig:
    _config_dir = str(project_root / "activitygraphs/conf")

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        return compose(config_name="config")
