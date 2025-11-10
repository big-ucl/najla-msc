from pathlib import Path

import polars as pl

import activitygraphs.exploration.dataprocessing as dp
from activitygraphs.config import DataConfig


def _read_raw_data(data_cfg: DataConfig, project_root=None) -> pl.DataFrame:
    project_root = project_root if project_root is not None else Path(".")
    data_path = project_root / data_cfg.paths.raw

    raw_geneva_df = dp.read_from_parquet(
        data_path / data_cfg.files.raw_trips,
        schema={
            "id_utilisateur": pl.String,
            "id_deplacement": pl.Categorical,
            "id_trajet": pl.Int8,
            "jour_depart": pl.String,
            "motif_depart": pl.Categorical,
            "motif_arrivee": pl.Categorical,
            "date": pl.String,
            "lieu_depart_trajet": pl.String,
            "lieu_arrivee_trajet": pl.String,
            "mode": pl.Categorical,
            "ligne_trajet": pl.String,
        },
    )

    raw_geneva_df = raw_geneva_df.with_columns(pl.col("jour_depart").str.to_date(format="%+", strict=False))

    return raw_geneva_df


def _handle_null_values(raw_geneva_df: pl.DataFrame) -> pl.DataFrame:
    # Remove null rows
    cols_remove_null_rows = ["jour_depart"]
    removed_rows_df = raw_geneva_df.drop_nulls(subset=cols_remove_null_rows)

    # Remove empty rows from certain columns
    cols_remove_empty_rows = ["date", "motif_depart", "motif_arrivee"]
    removed_rows_df = removed_rows_df.filter(pl.any_horizontal(pl.col(cols_remove_empty_rows) != ""))

    # Impute empty rows to `NA` category for location columns
    cols_impute_empty_to_na = ["lieu_depart_trajet", "lieu_arrivee_trajet"]
    imputed_na_rows_df = removed_rows_df.with_columns(pl.col(cols_impute_empty_to_na).replace(old="", new="NA"))

    # Impute empty values to unknown for mode column
    cols_impute_empty_to_unknown = ["mode"]
    imputed_unknown_rows_df = imputed_na_rows_df.with_columns(
        pl.col(cols_impute_empty_to_unknown).replace(old="", new="mode_unknown")
    )

    # Impute empty values of `ligne_trajet` to UNKNOWN or NA for ligne column depending on if mode is applicable
    applicable_modes = ["mode_bus", "mode_tramway", "mode_bateau_navette"]
    imputed_line_df = imputed_unknown_rows_df.with_columns(
        pl.when((pl.col("ligne_trajet") == "") & pl.col("mode").is_in(applicable_modes))
        .then(pl.lit("UNKNOWN"))
        .otherwise("ligne_trajet")
        .alias("ligne_trajet")
    ).with_columns(
        pl.when(pl.col("ligne_trajet") == "").then(pl.lit("NA")).otherwise("ligne_trajet").alias("ligne_trajet")
    )

    return imputed_line_df
