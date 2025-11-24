from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@dataclass
class RawFiles:
    raw_trips: str


@dataclass
class RawGenevaTPGFiles(RawFiles):
    pass


@dataclass
class RawLTDSFiles(RawFiles):
    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str


@dataclass
class DataPaths:
    processed: Path
    raw: Path
    act_dataset: Path
    graphs: Path
    metrics: Path


@dataclass
class DataConfig:
    name: str
    files: RawFiles
    paths: DataPaths


@dataclass
class OutputPaths:
    reports: Path
    figures: Path


@dataclass
class Config:
    data: DataConfig
    paths: OutputPaths


# cs = ConfigStore.instance()
# cs.store(name="ltds_config", node=Config)


def load_config(project_root: Path, verbose=True) -> Config:
    _config_dir = str(project_root / "activitygraphs/conf")

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config")

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cfg
