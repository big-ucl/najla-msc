from dataclasses import dataclass
from pathlib import Path

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