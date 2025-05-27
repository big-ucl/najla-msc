from pathlib import Path
from enum import Enum

import polars as pl


def convert_excel_to_parquet(data_path: Path, *files: list[Path]) -> list[Path]:
    new_files = []

    for file in files:
        df = pl.read_excel(data_path / file)

        new_file = file.with_suffix(".parquet")
        df.write_parquet(data_path / new_file)
        new_files.append(new_file)

    return new_files


class _PolarsEnum(Enum):
    @classmethod
    def names(cls):
        return [member.value for member in cls]

    @classmethod
    def polars_enum(cls):
        return pl.Enum(cls)


class Purposes(_PolarsEnum):
    MISSING = "Missing"
    NOT_ASKED = "Not asked"
    HOME = "Home"
    WORK = "Work - Usual workplace"
    WORK_DELIVERY = "Work - Delivery/loading"
    WORK_OTHER = "Work - Other"
    ENTERTAINMENT = "Entertainment/recreation"
    SHOPPING_FOOD = "Shopping - Food"
    PERSONAL_BUSINESS = "Personal business / use services"
    EDUCATION = "Education (as a pupil)"
    HOTEL = "Hotel / holiday home"
    ESCORT_WORK = "Drop off/pick up someone to/from work"
    ESCORT_SCHOOL = "Drop off/pick up someone to/from school"
    ESCORT_HEALTH = "Drop off/pick up someone to/from health visit"
    WORSHIP = "Worship or religious observance"
    OTHER = "Other"
    HEALTH = "Health or medical visit"
    ESCORT_OTHER = "Drop off/pick up someone to/from other place"
    SPORT = "Participate in Sport"
    LEISURE = "Leisure trip - enjoyment"
    SOCIAL_VISIT = "Visit friends/relatives at home"
    SOCIAL_OTHER = "Other Social"
    SHOPPING_OTHER = "Shopping - Other"


class LandUses(_PolarsEnum):
    MISSING = "Missing"
    NOT_ASKED = "Not asked"
    RESIDENTIAL = "Residential"
    OFFICE = "Office"
    FACTORY = "Factory/warehouse"
    SCHOOL = "School/College"
    SHOPS = "Shops"
    PUBLIC_BUILDING = "Public Buildings"
    OPEN_SPACE = "Open space"
    MYSTERY = "MYSTERY LAND USE"
    WORSHIP = "Place of worship"
    OTHER = "Other"
    HOSPITAL = "Hospital"
    GP = "GP/Dentist/Other health service"
