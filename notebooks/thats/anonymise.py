import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")

with app.setup:
    import math

    import marimo as mo
    import geopandas as gpd
    import polars as pl

    from pathlib import Path

    project_root = Path(mo.notebook_dir().parent.parent)


@app.cell
def _():
    from dataclasses import dataclass, asdict

    return asdict, dataclass


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Anonymising the THATS survey
    """)
    return


@app.cell
def _():
    raw_path = Path(
        "/home/luca/Documents/THATS/THATS Survey Data for Luca/Main Survey (GTHA)"
    )
    raw_demos_path = raw_path / "Demographics"
    raw_trips_path = raw_path / "Trips and Activities"
    return raw_demos_path, raw_trips_path


@app.cell
def _(dataclass, raw_demos_path, raw_trips_path):
    _indivs = pl.read_excel(
        raw_demos_path / "IndDem_October 25.xlsx",
        schema_overrides={
            "THATS HHID": pl.String,
            "THATS PersonID": pl.String,
            "THATS AppID": pl.String,
        },
    ).lazy()

    _hhs = pl.read_excel(
        raw_demos_path / "HHDem_October 25 With LU and Transit Access Data.xlsx",
        schema_overrides={
            "THATS HHID": pl.String,
        },
    ).lazy()
    _trips = pl.read_excel(raw_trips_path / "Trips.xlsx").lazy()
    _activs = (
        pl
        .read_csv(
            raw_trips_path / "HrAct_Full_November 07.csv",
        )
        .with_columns(
            pl.col("THATS HHID", "THATS PersonID").cast(pl.Int64).cast(pl.String)
        )
        .lazy()
    )


    @dataclass
    class THATSData:
        hhs: pl.LazyFrame
        indivs: pl.LazyFrame
        trips: pl.LazyFrame
        activs: pl.LazyFrame


    raw_thats = THATSData(_hhs, _indivs, _trips, _activs)
    return THATSData, raw_thats


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 1 - Remove non-useful columns and personal information and sensitive attributes
    """)
    return


@app.cell
def _():
    indiv_drop_cols = [
        "TTS HHID",
        "TTS PersonID",
        "TTS PersonNumber",
        "THATS Survey Email",
        "THATS age - 0.8",
        "THATS agerange",
        "THATS agerangedesc",
        "THATS age(-0.8)range",
        "TTS agerange",
        "TTS agerangedesc",
        "TTS agecategory",
        "TTS agecategorydesc",
    ]

    hhs_drop_cols = [
        "TTS HHID",
        "ContactEmail",
        "TTS ContactName",
        "TTS ContactPhone",
        "THATS Ethnicity",
        "THATS HomePostalCode",
        "PR",
        "CDuid",
        "CSDname",
        "CCScode",
        "SAC",
        "CTname",
        "ER",
        "DPL",
        "FED13uid",
        "POP_CNTR_RA",
        "dauid",
        "DisBlock",
        "LAT",
        "LONG",
        "Comm_Name",
        "H_DMT",
        "PO",
        "QI",
        "PopulationDensity",
        "LandAreainSquareKm",
        "cbd_latitude",
        "cbd_longitude",
        "distance_to_cbd",
        "Distance_to_CBDPER1000",
        "geometry",
        "index_right",
        "FID",
        "OBJECTID",
        "Name",
        "Shape_Leng",
        "Shape_Area",
        "Shape__Are",
        "Shape__Len",
        "HasAccessToTransit",
    ]

    activ_drop_cols = [
        "THATS Survey Email",
    ]
    return activ_drop_cols, hhs_drop_cols, indiv_drop_cols


@app.cell
def _(THATSData, activ_drop_cols, hhs_drop_cols, indiv_drop_cols):
    def drop_non_useful_columns(thats: THATSData) -> THATSData:
        hhs = thats.hhs.drop(hhs_drop_cols)
        indivs = thats.indivs.drop(indiv_drop_cols)
        activs = thats.activs.drop(activ_drop_cols)

        return THATSData(hhs, indivs, thats.trips, activs)

    return (drop_non_useful_columns,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 2: Reshuffle IDs
    """)
    return


@app.cell
def _():
    id_equivalences = {
        "hh_id": {
            "indivs": "THATS HHID",
            "hhs": "THATS HHID",
            "activs": "THATS HHID",
        },
        "person_id": {
            "indivs": "THATS PersonID",
            "activs": "THATS PersonID",
        },
        "app_id": {
            "indivs": "THATS AppID",
            "trips": "user_uuid",
            "activs": "THATS AppID",
        },
        "trip_id": {
            "trips": "trip_id",
            "activs": [
                "1stStrtTripID",
                "2ndStrtTripID",
                "3rdStrtTripID",
                "4thStrtTripID",
                "5thStrtTripID",
                "6thStrtTripID",
                "7thStrtTripID",
                "1stEndTripID",
                "2ndEndTripID",
                "3rdEndTripID",
                "4thEndTripID",
                "5thEndTripID",
                "6thEndTripID",
                "7thEndTripID",
            ],
        },
        "leg_id": {
            "trips": "section_id",
        },
    }
    return (id_equivalences,)


@app.cell
def _(THATSData, asdict):
    def extract_ids(
        df: pl.LazyFrame, cols: str | list[str], id_col: str
    ) -> pl.DataFrame:
        return (
            df
            .select(pl.concat_list(pl.col(cols).cast(str)).explode().alias(id_col))
            .unique()
            .drop_nulls()
        )


    def generate_id_map(
        thats: THATSData, id_col: str, equivalences: dict[str, str | list[str]]
    ) -> pl.LazyFrame:
        all_ids = pl.concat([
            extract_ids(df, equivalences[df_name], id_col)
            for df_name, df in asdict(thats).items()
            if df_name in equivalences
        ]).unique()

        # Set as longer than log10(num_ids) for all IDs in THATS
        max_num_digits = 8

        id_map = all_ids.with_columns(
            pl
            .row_index(name=f"{id_col}_new")
            .cast(str)
            .str.pad_start(max_num_digits, "0")
        )

        return id_map


    def replace_ids_in_cols(
        df_name: str,
        df: pl.LazyFrame,
        id_map: pl.LazyFrame,
        id_col: str,
        new_id_col: str,
        df_id_cols: str | list[str],
    ) -> pl.LazyFrame:
        df_id_cols = [df_id_cols] if isinstance(df_id_cols, str) else df_id_cols

        for df_id_col in df_id_cols:
            df = df.with_columns(pl.col(df_id_col).cast(str))
            df = (
                df
                .join(id_map, left_on=df_id_col, right_on=id_col, how="left")
                .with_columns(pl.col(new_id_col).alias(df_id_col).cast(str))
                .drop(new_id_col)
            )

            if (id_col, df_name) != ("trip_id", "activs"):
                df = df.rename({df_id_col: id_col})

        return df


    def reshuffle_ids(
        thats: THATSData, id_equivalences: dict[str, dict[str, str | list[str]]]
    ) -> THATSData:
        new_thats = asdict(thats)

        for id_col, equivalences in id_equivalences.items():
            new_id_col = f"{id_col}_new"
            id_map = generate_id_map(thats, id_col, equivalences)

            for df_name, df in new_thats.items():
                if df_name in equivalences:
                    df_id_cols = equivalences[df_name]
                    df = replace_ids_in_cols(
                        df_name, df, id_map, id_col, new_id_col, df_id_cols
                    )

                new_thats[df_name] = df

        return THATSData(**new_thats)

    return (reshuffle_ids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 3: Remove children from dataset
    """)
    return


@app.cell
def _(THATSData):
    def remove_children(thats: THATSData) -> THATSData:
        adult_indivs = thats.indivs.filter(pl.col("THATS age") >= 18)
        adult_person_ids = adult_indivs.select("person_id")
        adult_app_ids = adult_indivs.select("app_id")

        adult_activs = thats.activs.join(adult_person_ids, on="person_id")
        adult_trips = thats.trips.join(adult_app_ids, on="app_id")

        return THATSData(thats.hhs, adult_indivs, adult_trips, adult_activs)

    return (remove_children,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 4: Replace latitudes and longitudes with Census Tracts / DAs
    """)
    return


@app.cell
def _():
    coordinate_columns = {
        "hhs": [
            ["THATS HomeLon", "THATS HomeLat"],
        ],
        "trips": [
            ["start_loc_lon", "start_loc_lat"],
            ["end_loc_lon", "end_loc_lat"],
            ["start_loc_lon_section", "start_loc_lat_section"],
            ["end_loc_lon_section", "end_loc_lat_section"],
        ],
    }
    return (coordinate_columns,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    - Dissemination Area (DA) & Census Tract (CT) boundaries: [link](https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21)
    """)
    return


@app.cell
def _():
    boundaries_dir = project_root / "data/external/boundaries/canada"
    CRS = "EPSG:4326"
    return CRS, boundaries_dir


@app.cell
def _(boundaries_dir):
    ct = gpd.read_file(
        boundaries_dir / "lct_000b21a_e.zip", columns=["CTUID"]
    ).rename(columns={"CTUID": "zone_id"})
    da = gpd.read_file(
        boundaries_dir / "lda_000b21a_e.zip", columns=["DAUID"]
    ).rename(columns={"DAUID": "zone_id"})
    return ct, da


@app.cell
def _(CRS, THATSData, asdict, coordinate_columns, ct, da):
    def add_zone_id_from_coords(
        df: pl.LazyFrame, zones: pl.LazyFrame, lon: str, lat: str, zone_name: str
    ) -> pl.LazyFrame:
        zone_name = lon.replace("Lon", zone_name).replace("lon", zone_name)
        zones_crs = zones.estimate_utm_crs()

        coords = df.select(lon, lat).collect().to_pandas()
        coords = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(coords[lon], coords[lat]), crs=CRS
        ).to_crs(zones_crs)

        zones = coords.sjoin(zones.to_crs(zones_crs), how="left").to_crs(CRS)
        zones = pl.Series(zone_name, zones["zone_id"])

        return df.with_columns(zones)


    def replace_coords_with_zones(thats: THATSData) -> THATSData:
        new_thats = asdict(thats)

        for df_name, coord_cols in coordinate_columns.items():
            for col_pair in coord_cols:
                lon, lat = col_pair

                new_thats[df_name] = (
                    new_thats[df_name]
                    .pipe(add_zone_id_from_coords, ct, lon, lat, "CT")
                    .pipe(add_zone_id_from_coords, da, lon, lat, "DA")
                    .drop(lon, lat)
                )

        return THATSData(**new_thats)

    return (replace_coords_with_zones,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Step 5: Write results to file
    """)
    return


@app.cell
def _(
    drop_non_useful_columns,
    id_equivalences,
    raw_thats,
    remove_children,
    replace_coords_with_zones,
    reshuffle_ids,
):
    thats = drop_non_useful_columns(raw_thats)
    thats = reshuffle_ids(thats, id_equivalences)
    thats = remove_children(thats)
    thats = replace_coords_with_zones(thats)

    thats
    return (thats,)


@app.cell
def _(thats):
    output_dir = project_root / "data/raw/THATS"

    thats.hhs.sink_parquet(output_dir / "hhs.parquet")
    thats.indivs.sink_parquet(output_dir / "indivs.parquet")
    thats.trips.sink_parquet(output_dir / "trips.parquet")
    thats.activs.sink_parquet(output_dir / "activs.parquet")
    return (output_dir,)


@app.cell
def _(output_dir):
    pl.read_parquet(output_dir / "activs.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
