from dataclasses import dataclass
from pathlib import Path
from hydra.core.config_store import ConfigStore


@dataclass
class Files:
    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str

@dataclass
class Paths:
    data_raw: Path
    data_raw_ltds: Path

@dataclass
class LTDSConfig:
    files: Files
    paths: Paths

cs = ConfigStore.instance()
cs.store(name="ltds_config", node=LTDSConfig)
