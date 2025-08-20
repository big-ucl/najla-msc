import polars as pl


def check_schema(df: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Checks that a DataFrame matches provided Schema.

    Args:
        df (pl.DataFrame): the dataframe to be checked
        schema (pl.Schema): the schema to check against

    Raises:
        ValueError: if the DataFrame does not match the schema

    Returns:
        pl.DataFrame: the valid DataFrame
    """
    df_items = set(df.schema.items())
    schema_items = set(schema.items())

    difference = df_items ^ schema_items

    if difference:
        raise ValueError(
            f"Schemas do not match:\nExpected: {schema}\nGot:      {df.schema}\nDifferent elements: {difference}"
        )

    return df
