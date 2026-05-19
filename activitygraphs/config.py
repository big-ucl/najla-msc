from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

# ===================================================
# Small sub-inputs
# ===================================================


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
class TorontoBoundaryInputs:
    directory: str

    metropolitan_areas: str
    census_tracts: str
    dissemination_areas: str


@dataclass
class StatsInputs:
    directory: str


@dataclass
class GenevaStatsInputs(StatsInputs):
    file: str


@dataclass
class TorontoStatsInputs(StatsInputs):
    population: str
    jobs: str


# ===================================================
# Dataset-specific input
# ===================================================


@dataclass
class Inputs:
    raw_journeys: str
    overture: OvertureInputs
    statistics: StatsInputs


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


@dataclass
class TorontoInputs(Inputs):
    boundaries: TorontoBoundaryInputs


# ===================================================
# Paths for datasets
# ===================================================


@dataclass
class DataPaths:
    processed: Path
    raw: Path

    external: Path
    gtfs: Path

    pyg_datasets: Path


# ===================================================
# Data configs
# ===================================================


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
class TorontoDataConfig(DataConfig):
    inputs: TorontoInputs


# ===================================================
# Overall config (outputs + main)
# ===================================================


@dataclass
class OutputPaths:
    reports: Path
    figures: Path
    models: Path


@dataclass
class Config:
    data: DataConfig
    paths: OutputPaths


# cs = ConfigStore.instance()
# cs.store(name="ltds_config", node=Config)


def load_config(project_root: Path, verbose=True, data: Literal["ltds", "geneva", "toronto"] = "geneva") -> Config:
    _config_dir = str(project_root / "activitygraphs/conf")

    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cfg
