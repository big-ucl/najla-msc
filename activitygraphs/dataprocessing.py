from pathlib import Path

import polars as pl


def convert_excel_to_parquet(data_path: Path, *files: list[Path]) -> list[Path]:
    new_files = []

    for file in files:
        df = pl.read_excel(data_path / file)

        new_file = file.with_suffix(".parquet")
        df.write_parquet(data_path / new_file)
        new_files.append(new_file)

    return new_files
