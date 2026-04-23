from pathlib import Path

import geopandas as gpd


def add_population_job_statistics(
    locations: gpd.GeoDataFrame, path: Path, normalise: bool = True, project_root: Path | None = None
) -> gpd.GeoDataFrame:
    project_root: Path = project_root if project_root is not None else Path(".")

    utm_crs = locations.estimate_utm_crs()
    original_locations = locations

    statistics = gpd.read_file(project_root / path)
    statistics["population"] = statistics["D_POP_HA"] * statistics["GEOM_AREA"] / 10000
    statistics["jobs"] = statistics["D_EMP_HA"] * statistics["GEOM_AREA"] / 10000
    statistics = statistics.rename(columns={"GRID_ID": "id"}).to_crs(utm_crs)
    statistics = statistics[["id", "population", "jobs", "geometry"]]

    locations = locations.reset_index()[["loc_id", "geometry"]].to_crs(utm_crs).copy()
    locations["area"] = locations.geometry.area

    intersection = locations.sjoin(statistics.to_crs(utm_crs), how="left", predicate="intersects")
    stats_by_sector = intersection.groupby("loc_id")[["population", "jobs"]].sum()

    if normalise:
        stats_by_sector = stats_by_sector.merge(locations[["loc_id", "area"]], on="loc_id")
        stats_by_sector["population"] = stats_by_sector["population"] / stats_by_sector["area"]
        stats_by_sector["jobs"] = stats_by_sector["jobs"] / stats_by_sector["area"]
        stats_by_sector = stats_by_sector.drop(columns=["area"])

    stats_by_sector = original_locations.merge(stats_by_sector, on="loc_id", how="right")
    stats_by_sector["area"] = stats_by_sector.to_crs(utm_crs).geometry.area

    return stats_by_sector
