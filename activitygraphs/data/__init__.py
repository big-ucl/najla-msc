"""
Dataset loaders for Geneva, Toronto, and LTDS survey data.

This package contains one module per travel survey dataset, each providing:
  - A frozen dataclass holding the raw parsed input files (e.g. ``GenevaInputs``)
  - A ``NetworkData`` subclass providing a cleaned, filterable view of the data
  - ``load_files``: reads raw data from disk into the inputs dataclass
  - ``build_*_data``: parses and standardises raw inputs into a NetworkData instance
  - Location builder helpers for each geographic representation used in the survey

Supported datasets:
  - ``geneva``: Geneva TPG travel survey (Switzerland). Locations: subsectors, Swiss and
    French municipalities, PT stops.
  - ``toronto``: Toronto THATS travel survey (Canada). Locations: census tracts (subsectors).
  - ``ltds``: London Travel Demand Survey (UK). Archived/exploratory module.
  - ``gtfs``: GTFS feed parser shared across datasets that include PT routing.
  - ``overture``: Overture Maps POI and land-use downloader, shared across datasets.
  - ``statistics``: Census population and job enrichment functions, shared across datasets.
"""
