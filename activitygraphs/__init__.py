"""
activitygraphs package initialiser.

This package implements graph machine learning models for activity-based transport
modelling. The core research question is: given a traveller's history of visited
locations, which locations are likely to be included in their *choice set* for the
next activity destination?

The package is organised into the following sub-modules:

  - base       : shared enums (Mode, Purpose), schema definitions, and constants
  - config     : Hydra/OmegaConf dataclass configuration tree and loader
  - main       : Hydra entry-point that kicks off the comparison experiment
  - run        : high-level experiment runners, model builders, and result saving
  - utils      : schema validation helpers, geometry converters, and I/O utilities
  - mapping    : interactive Folium map visualisation helpers
  - ml/        : dataset, datamodule, models, baselines, and training logic

Typical entry point:
    python -m activitygraphs.main data=toronto
"""
