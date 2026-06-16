"""
Hydra/OmegaConf dataclass configuration tree and ``load_config`` helper.

Hydra is a framework for managing complex application configurations.  This
module defines a hierarchy of Python dataclasses that mirrors the YAML files
stored in ``activitygraphs/conf/``.  Hydra reads those YAML files and
populates these dataclasses at runtime, giving typed, auto-completed access
to every configuration value.

The class hierarchy is:
  Config
  ├── data : DataConfig  (one of LTDSDataConfig / GenevaDataConfig / TorontoDataConfig)
  │   ├── inputs : Inputs  (one of LTDSInputs / GenevaInputs / TorontoInputs)
  │   │   ├── overture : OvertureInputs
  │   │   ├── statistics : StatsInputs  (GenevaStatsInputs / TorontoStatsInputs)
  │   │   └── boundaries : GenevaBoundaryInputs / TorontoBoundaryInputs  (dataset-specific)
  │   └── paths : DataPaths
  ├── paths : OutputPaths
  └── train : TrainConfig

Usage example:
    from activitygraphs.config import load_config
    from pathlib import Path
    cfg = load_config(Path("."), data="toronto")
    print(cfg.data.name)          # "toronto"
    print(cfg.train.epochs)       # e.g. 50
"""

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
    """
    Description: Holds the file-name strings for each standard GTFS (General
    Transit Feed Specification) text file inside a single transit feed directory.
    GTFS is the open standard used by transit agencies to publish schedules.

    Input:
      - directory (str): relative or absolute path to the folder that contains
                         the GTFS .txt files.
      - stops (str): filename of stops.txt — one row per physical stop/station.
      - stop_times (str): filename of stop_times.txt — departure/arrival times
                          per trip and stop.
      - trips (str): filename of trips.txt — individual vehicle runs on a route.
      - routes (str): filename of routes.txt — route definitions and metadata.
      - agency (str): filename of agency.txt — transit agency details.
      - calendar (str): filename of calendar.txt — weekly service patterns.
      - calendar_dates (str): filename of calendar_dates.txt — exceptions to
                              the regular weekly pattern (holidays, special days).
      - transfers (str): filename of transfers.txt — transfer rules between stops.

    Output:
      - (GTFSInputs): dataclass instance; accessed as cfg.data.inputs.gtfs.*
    """

    # Path to the folder containing all GTFS .txt files for this feed.
    directory: str

    # File containing the list of stops (stop_id, stop_name, stop_lat, stop_lon).
    stops: str
    # File containing departure and arrival times for every trip at every stop.
    stop_times: str
    # File linking trip_id to route_id, service_id, and shape_id.
    trips: str
    # File describing each route (route_id, route_short_name, route_type, etc.).
    routes: str
    # File with agency details (agency_id, agency_name, agency_url, etc.).
    agency: str
    # File defining regular weekly service (Monday–Sunday flags per service_id).
    calendar: str
    # File listing date-specific exceptions (added/removed service on a given date).
    calendar_dates: str
    # File specifying transfer rules between pairs of stops.
    transfers: str


@dataclass
class OvertureInputs:
    """
    Description: File names for Overture Maps data files. Overture Maps is an
    open mapping dataset that provides land-use polygons and point-of-interest
    (place) data, used to enrich location nodes with contextual features such as
    nearby amenities or land-use categories.

    Input:
      - directory (str): path to the folder containing the Overture data files.
      - land_use (str): filename of the land-use GeoParquet/GeoJSON file
                        (polygons describing how land is used: residential,
                        commercial, park, etc.).
      - place (str): filename of the place/POI file (points of interest with
                     categories such as restaurant, school, hospital).

    Output:
      - (OvertureInputs): dataclass instance; accessed as cfg.data.inputs.overture.*
    """

    # Path to the folder containing all Overture data files.
    directory: str
    # Land-use polygon file — used to compute land-use mix features for each zone.
    land_use: str
    # Place / POI point file — used to count nearby amenities for each location.
    place: str


@dataclass
class GenevaBoundaryInputs:
    """
    Description: File names for all boundary shapefiles needed to build the
    spatial location graph for the Geneva / MTMC survey dataset.  Geneva spans
    both Swiss and French territory, so multiple boundary sources are needed.

    Input:
      - directory (str): path to the folder containing all boundary shapefiles.
      - geneva_subsectors (str): shapefile for Geneva's administrative subsectors
                                 (the finest spatial unit used in the MTMC survey).
      - french_postcodes (str): shapefile for French postal code zones (communes
                                on the French side of the Geneva agglomeration).
      - swiss_postcodes (str): shapefile for Swiss postal code zones.
      - swiss_localities (str): shapefile for Swiss locality polygons.
      - swiss_boundaries (str): shapefile for higher-level Swiss administrative
                                boundaries (cantons / municipalities).

    Output:
      - (GenevaBoundaryInputs): dataclass instance; accessed as
                                cfg.data.inputs.boundaries.* for Geneva configs.
    """

    # Path to the folder holding all Geneva boundary files.
    directory: str

    # Shapefile of Geneva's finest administrative zones (subsectors).
    geneva_subsectors: str
    # Shapefile of French postal code areas bordering Geneva.
    french_postcodes: str
    # Shapefile of Swiss postal code areas.
    swiss_postcodes: str
    # Shapefile of Swiss localities (named populated places).
    swiss_localities: str
    # Shapefile of Swiss national / cantonal / municipal boundaries.
    swiss_boundaries: str


@dataclass
class TorontoBoundaryInputs:
    """
    Description: File names for all boundary shapefiles needed to build the
    spatial location graph for the Toronto / TTS survey dataset.  Statistics
    Canada provides hierarchical census geographic boundaries at several levels
    of granularity.

    Input:
      - directory (str): path to the folder containing the boundary shapefiles.
      - metropolitan_areas (str): shapefile for Census Metropolitan Areas (CMAs) —
                                  the coarsest unit, used to clip the study region.
      - census_tracts (str): shapefile for census tracts — medium-grain zones of
                             roughly 2,500–8,000 people each.
      - dissemination_areas (str): shapefile for dissemination areas (DAs) — the
                                   finest standard census geography (~400–700 people).

    Output:
      - (TorontoBoundaryInputs): dataclass instance; accessed as
                                 cfg.data.inputs.boundaries.* for Toronto configs.
    """

    # Path to the folder containing all Toronto / Statistics Canada boundary files.
    directory: str

    # Shapefile of Census Metropolitan Areas — used to define the study boundary.
    metropolitan_areas: str
    # Shapefile of census tracts — used as medium-resolution spatial units.
    census_tracts: str
    # Shapefile of dissemination areas — the finest spatial units in the TTS survey.
    dissemination_areas: str


@dataclass
class StatsInputs:
    """
    Description: Base class for census / statistics data file configurations.
    Subclasses add dataset-specific file names on top of the shared directory
    path.  Statistics data provides population counts and job densities that
    are used as node features in the graph.

    Input:
      - directory (str): path to the folder containing the statistics files.

    Output:
      - (StatsInputs): base dataclass; not instantiated directly — use a subclass.
    """

    # Shared path to the folder containing census or statistics data files.
    directory: str


@dataclass
class GenevaStatsInputs(StatsInputs):
    """
    Description: Census statistics file config for the Geneva dataset.  Geneva
    uses a single raster/grid file (e.g. a GeoTIFF or GeoParquet) that encodes
    population and employment figures on a regular spatial grid.

    Input:
      - directory (str): inherited from StatsInputs — path to the stats folder.
      - file (str): filename of the single statistics grid file.

    Output:
      - (GenevaStatsInputs): dataclass instance; accessed as
                             cfg.data.inputs.statistics.* for Geneva configs.
    """

    # Filename of the grid/raster statistics file (population + employment data).
    file: str


@dataclass
class TorontoStatsInputs(StatsInputs):
    """
    Description: Census statistics file config for the Toronto dataset.  Toronto
    provides separate CSV files for population and job counts, one row per
    dissemination area.

    Input:
      - directory (str): inherited from StatsInputs — path to the stats folder.
      - population (str): filename of the population CSV (one row per DA with
                          total resident population).
      - jobs (str): filename of the employment / jobs CSV (one row per DA with
                    total job count).

    Output:
      - (TorontoStatsInputs): dataclass instance; accessed as
                              cfg.data.inputs.statistics.* for Toronto configs.
    """

    # Filename of the CSV containing population counts per dissemination area.
    population: str
    # Filename of the CSV containing job / employment counts per dissemination area.
    jobs: str


# ===================================================
# Dataset-specific input
# ===================================================


@dataclass
class Inputs:
    """
    Description: Base class that lists the raw data files shared by all supported
    travel-survey datasets.  Subclasses (LTDSInputs, GenevaInputs, TorontoInputs)
    add survey-specific files on top of these shared fields.

    Input:
      - raw_journeys (str): filename of the main processed journeys/trips file
                            (already cleaned into the USER_JOURNEY_SCHEMA format).
      - overture (OvertureInputs): nested config pointing to Overture Maps files
                                   used for location feature enrichment.
      - statistics (StatsInputs): nested config pointing to census statistics files
                                  used for population/job node features.

    Output:
      - (Inputs): base dataclass; not instantiated directly — use a subclass.
    """

    # Filename of the pre-processed journeys parquet file (USER_JOURNEY_SCHEMA).
    raw_journeys: str
    # Nested config for Overture Maps land-use and place data files.
    overture: OvertureInputs
    # Nested config for census / statistics files (population, jobs).
    statistics: StatsInputs


@dataclass
class LTDSInputs(Inputs):
    """
    Description: Raw-file inputs specific to the London Travel Demand Survey (LTDS).
    The LTDS provides separate files for households, persons, trips, and stages
    (each stage = one leg of a trip).

    Input:
      - raw_household (str): filename of the household-level survey file.
      - raw_person (str): filename of the person-level survey file.
      - raw_trip (str): filename of the trip-level survey file.
      - raw_stage (str): filename of the stage/leg-level survey file.
      - (plus all fields inherited from Inputs)

    Output:
      - (LTDSInputs): dataclass instance; accessed as cfg.data.inputs.* for LTDS.
    """

    # Household-level LTDS survey file (household demographics, car ownership, etc.).
    raw_household: str
    # Person-level LTDS file (age, gender, employment status of each respondent).
    raw_person: str
    # Trip-level LTDS file (one row per complete trip: origin, destination, purpose).
    raw_trip: str
    # Stage-level LTDS file (one row per leg/mode within a trip).
    raw_stage: str


@dataclass
class GenevaInputs(Inputs):
    """
    Description: Raw-file inputs specific to the Geneva Microcensus on Mobility
    and Transport (MTMC) survey.  In addition to the shared journey file, the
    Geneva dataset requires GTFS transit data and geographic boundary files.

    Input:
      - boundaries (GenevaBoundaryInputs): nested config for Geneva boundary files.
      - gtfs (GTFSInputs): nested config for the Geneva-area GTFS transit feed.
      - (plus all fields inherited from Inputs)

    Output:
      - (GenevaInputs): dataclass instance; accessed as cfg.data.inputs.* for Geneva.
    """

    # Nested config for all spatial boundary shapefiles for the Geneva region.
    boundaries: GenevaBoundaryInputs
    # Nested config for the GTFS transit feed files (stops, routes, timetables).
    gtfs: GTFSInputs


@dataclass
class TorontoInputs(Inputs):
    """
    Description: Raw-file inputs specific to the Toronto Transportation Tomorrow
    Survey (TTS).  The TTS provides separate CSV files for persons, households,
    and activities (trips).

    Input:
      - boundaries (TorontoBoundaryInputs): nested config for Toronto boundary files.
      - raw_person (str): filename of the person-level TTS survey file.
      - raw_household (str): filename of the household-level TTS survey file.
      - raw_activities (str): filename of the activities/trips TTS survey file
                              (one row per activity episode: origin, destination,
                              purpose, mode).
      - (plus all fields inherited from Inputs)

    Output:
      - (TorontoInputs): dataclass instance; accessed as cfg.data.inputs.* for Toronto.
    """

    # Nested config for all spatial boundary shapefiles for the Toronto region.
    boundaries: TorontoBoundaryInputs

    # Person-level TTS file (respondent demographics and home location).
    raw_person: str
    # Household-level TTS file (household size, vehicle ownership, etc.).
    raw_household: str
    # Activity/trip-level TTS file (each activity episode made by each person).
    raw_activities: str


# ===================================================
# Paths for datasets
# ===================================================


@dataclass
class DataPaths:
    """
    Description: Stores all filesystem paths relevant to a single dataset.
    These paths point to different stages of the data pipeline: raw survey
    downloads, processed / cleaned data, external (third-party) data, and
    PyTorch Geometric dataset caches.

    Input:
      - processed (Path): directory where cleaned/processed data files are saved.
      - raw (Path): directory containing unmodified raw survey/GTFS downloads.
      - external (Path): directory for third-party external data (Overture, stats).
      - gtfs (Path): directory for the GTFS transit feed files (may overlap with raw).
      - pyg_datasets (Path): directory where PyG InMemoryDataset caches are stored.

    Output:
      - (DataPaths): dataclass instance; accessed as cfg.data.paths.* in the code.
    """

    # Directory where the cleaned, schema-validated data files are written.
    processed: Path
    # Directory containing the original, unmodified raw survey downloads.
    raw: Path

    # Directory for third-party / externally sourced data files.
    external: Path
    # Directory containing the GTFS feed text files for this dataset.
    gtfs: Path

    # Directory where PyTorch Geometric InMemoryDataset .pt cache files are stored.
    pyg_datasets: Path


# ===================================================
# Data configs
# ===================================================


@dataclass
class DataConfig:
    """
    Description: Base configuration for a single travel-survey dataset.
    Groups together the dataset name, all raw input file references, and the
    filesystem paths needed throughout the pipeline.

    Input:
      - name (str): short string identifier for the dataset (e.g. "geneva",
                    "toronto", "ltds").  Used in file names and W&B run names.
      - inputs (Inputs): nested config pointing to all raw input files.
      - paths (DataPaths): nested config with all filesystem paths.

    Output:
      - (DataConfig): base dataclass; typically accessed as cfg.data in user code.
    """

    # Short name of the dataset used in file names and logging (e.g. "toronto").
    name: str
    # Nested config holding file names for all raw input data files.
    inputs: Inputs
    # Nested config holding filesystem paths for raw, processed, and cache dirs.
    paths: DataPaths


@dataclass
class LTDSDataConfig(DataConfig):
    """
    Description: Dataset configuration for the London Travel Demand Survey (LTDS).
    Overrides the ``inputs`` field with the LTDS-specific LTDSInputs subclass.

    Input:
      - inputs (LTDSInputs): LTDS-specific inputs with household/person/trip/stage files.
      - (plus name and paths inherited from DataConfig)

    Output:
      - (LTDSDataConfig): dataclass instance; accessed as cfg.data when data=ltds.
    """

    # Overrides parent field with the LTDS-specific input file configuration.
    inputs: LTDSInputs


@dataclass
class GenevaDataConfig(DataConfig):
    """
    Description: Dataset configuration for the Geneva MTMC survey.
    Overrides the ``inputs`` field with the Geneva-specific GenevaInputs subclass.

    Input:
      - inputs (GenevaInputs): Geneva-specific inputs including GTFS and boundary files.
      - (plus name and paths inherited from DataConfig)

    Output:
      - (GenevaDataConfig): dataclass instance; accessed as cfg.data when data=geneva.
    """

    # Overrides parent field with the Geneva-specific input file configuration.
    inputs: GenevaInputs


@dataclass
class TorontoDataConfig(DataConfig):
    """
    Description: Dataset configuration for the Toronto Transportation Tomorrow
    Survey (TTS).  Overrides the ``inputs`` field with the Toronto-specific
    TorontoInputs subclass.

    Input:
      - inputs (TorontoInputs): Toronto-specific inputs with person/household/activity files.
      - (plus name and paths inherited from DataConfig)

    Output:
      - (TorontoDataConfig): dataclass instance; accessed as cfg.data when data=toronto.
    """

    # Overrides parent field with the Toronto-specific input file configuration.
    inputs: TorontoInputs


# ===================================================
# Overall config (outputs + main)
# ===================================================


@dataclass
class OutputPaths:
    """
    Description: Stores filesystem paths for all experiment output artefacts.
    These are written by the training and evaluation code and are independent of
    any specific dataset.

    Input:
      - reports (Path): directory where result parquet files and text reports are saved.
      - figures (Path): directory where plots and visualisation images are saved.
      - models (Path): directory where trained PyTorch model checkpoints are saved.

    Output:
      - (OutputPaths): dataclass instance; accessed as cfg.paths.* in user code.
    """

    # Directory for result tables, metric CSVs, and other textual experiment reports.
    reports: Path
    # Directory for matplotlib / plotly figures generated during evaluation.
    figures: Path
    # Directory for saved PyTorch Lightning model checkpoint files (.ckpt / .pt).
    models: Path


@dataclass
class TrainConfig:
    """
    Description: All configuration options that control the training loop.
    Includes the number of epochs, debugging shortcuts, learning-rate scheduling,
    and Weights & Biases (W&B) experiment tracking settings.

    Input:
      - epochs (int): total number of full passes over the training dataset.
      - fast_dev_run (bool): if True, run only a single train/val/test batch
                             (PyTorch Lightning's fast sanity-check mode).
      - overfit_batches (int): if > 0, train and evaluate on this many batches only
                               (useful for debugging model capacity).
      - schedule_lr (bool): if True, use a cosine annealing learning-rate scheduler;
                            otherwise use a fixed learning rate.
      - debug (bool): if True, enable extra logging and possibly smaller dataset.
      - wandb (bool): if True, log metrics to Weights & Biases.
      - wandb_project (str): name of the W&B project to log runs under.
      - wandb_entity (str | None): W&B team/user entity; None uses the default.

    Output:
      - (TrainConfig): dataclass instance; accessed as cfg.train.* in user code.
    """

    # Number of training epochs (complete passes over the training set).
    epochs: int

    # If True, only run 1 batch of train/val/test — a quick sanity check.
    fast_dev_run: bool
    # If > 0, use only this many batches for both training and validation.
    overfit_batches: int
    # If True, apply a cosine annealing learning-rate schedule during training.
    schedule_lr: bool
    # If True, activate debug mode (extra logging, smaller data, etc.).
    debug: bool

    # If True, log training metrics to the Weights & Biases platform.
    wandb: bool
    # W&B project name under which all runs are grouped.
    wandb_project: str
    # W&B entity (team or user) to log runs to; None means use the default entity.
    wandb_entity: str | None


@dataclass
class Config:
    """
    Description: The top-level configuration object that Hydra populates when the
    application is launched.  It bundles together all dataset, output-path, and
    training settings into a single object that is passed through the entire
    experiment pipeline.

    Input:
      - data (DataConfig): dataset-specific configuration (name, input files, paths).
                           At runtime this will be one of LTDSDataConfig,
                           GenevaDataConfig, or TorontoDataConfig depending on the
                           ``data=`` Hydra override.
      - paths (OutputPaths): filesystem paths for reports, figures, and saved models.
      - train (TrainConfig): all training hyper-parameters and W&B settings.

    Output:
      - (Config): the root configuration object; passed as ``cfg`` to main() and
                  all downstream experiment functions.
    """

    # Dataset-level config (which survey, which files, which directories).
    data: DataConfig
    # Output directory config (where to save results, figures, and model checkpoints).
    paths: OutputPaths
    # Training hyper-parameter and experiment-tracking config.
    train: TrainConfig


# cs = ConfigStore.instance()
# cs.store(name="ltds_config", node=Config)


def load_config(project_root: Path, verbose=True, data: Literal["ltds", "geneva", "toronto"] = "geneva") -> Config:
    """
    Description: Load the Hydra configuration for the specified dataset by
    reading the YAML files from ``<project_root>/activitygraphs/conf/``.
    This is the recommended way to obtain a ``Config`` object outside of a
    Hydra-decorated ``main()`` function (e.g. in notebooks or scripts).

    Hydra's ``initialize_config_dir`` + ``compose`` API is used instead of
    ``@hydra.main`` so that this function can be called multiple times
    within the same Python process (e.g. in a Jupyter notebook cell).

    Input:
      - project_root (Path): absolute or relative path to the repository root.
                             The config directory is resolved as
                             ``project_root/activitygraphs/conf``.
      - verbose (bool): if True, print the fully-resolved YAML to stdout so
                        the user can verify all values. Defaults to True.
      - data (Literal["ltds", "geneva", "toronto"]): which dataset config group
                        to load.  Maps to a YAML file in ``conf/data/``.
                        Defaults to "geneva".

    Output:
      - (Config): a fully populated OmegaConf ``Config`` object whose fields
                  mirror the dataclass hierarchy defined above.
    """
    # Build the absolute path string to the Hydra config directory.
    _config_dir = str(project_root / "activitygraphs/conf")

    # Use Hydra's compose API to merge the base config with the dataset override.
    # initialize_config_dir sets the search path; compose resolves all YAML files.
    with initialize_config_dir(version_base=None, config_dir=_config_dir):
        # overrides=[f"data={data}"] selects the correct data/*.yaml config file.
        cfg = compose(config_name="config", overrides=[f"data={data}"])

    # Optionally print the fully-resolved YAML for human inspection.
    if verbose:
        print(f"Loaded config file \n{OmegaConf.to_yaml(cfg)}")

    return cfg
