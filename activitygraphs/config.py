from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


@dataclass
class GTFSInputs:
    directory: str

    stops: str
    stop_times: str
    trips: str
    routes: str
    agency: str
    calendar: str
    calendar_dates: str
    transfers: str


@dataclass
class OvertureInputs:
    directory: str
    land_use: str
    place: str


@dataclass
class GenevaBoundaryInputs:
    directory: str

    geneva_subsectors: str
    french_postcodes: str
    swiss_postcodes: str
    swiss_localities: str
    swiss_boundaries: str


@dataclass
class Inputs:
    raw_journeys: str
    overture: OvertureInputs


@dataclass
class LTDSInputs(Inputs):
    raw_household: str
    raw_person: str
    raw_trip: str
    raw_stage: str


@dataclass
class GenevaInputs(Inputs):
    boundaries: GenevaBoundaryInputs
    gtfs: GTFSInputs
    statistics: Path


@dataclass
class DataPaths:
    processed: Path
    raw: Path

    external: Path
    gtfs: Path

    act_dataset: Path
    graphs: Path
    metrics: Path


@dataclass
class DataConfig:
    name: str
    inputs: Inputs
    paths: DataPaths


@dataclass
class LTDSDataConfig(DataConfig):
    inputs: LTDSInputs


@dataclass
class GenevaDataConfig(DataConfig):
    inputs: GenevaInputs


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
