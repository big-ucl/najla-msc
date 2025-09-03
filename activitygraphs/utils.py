import polars as pl
import torch


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


def check_shape(tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Checks that a Tensor has a given shape, ignores size of dimension if size is set to -1.

    Args:
        tensor (torch.Tensor): the tensor
        shape (tuple[int, ...]): a tuple of dimension sizes

    Raises:
        ValueError: if the shape is invalid

    Returns:
        torch.Tensor: the checked tensor
    """
    if len(tensor.shape) != len(shape):
        raise ValueError(f"Tensor has invalid number of dims, got shape {tensor.shape}, expected {shape}")

    for i, (t, s) in enumerate(zip(tensor.shape, shape)):
        if t != s and s != -1:
            raise ValueError(
                f"Tensor has invalid shape along dimension {i}, got shape {tensor.shape}, expected {shape}"
            )

    return tensor
