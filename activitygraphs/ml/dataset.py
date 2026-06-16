"""
ActivityDataset, FittedScalers, and the ``load_dataset`` entry point for the ML pipeline.

Overview
--------
This module is responsible for turning raw travel-survey data into PyTorch
tensors ready for training a graph neural network.

Key concepts
~~~~~~~~~~~~
* **Network graph**: A single graph whose nodes represent locations (zones,
  stops, …) and whose edges represent connections (roads, transit links).
  All users *share* this graph structure — only the per-user node features
  (e.g. "did this user visit this location?") differ.

* **ActivityDataset**: The main PyTorch-Geometric Dataset class.  On the
  first call it processes and saves five ``.pt`` files; on subsequent calls it
  simply loads them.  ``get(i)`` returns one ``pyg.data.Data`` object for user
  ``i`` whose node feature matrix concatenates the shared network features with
  user ``i``'s spatial features.

* **FittedScalers**: A dataclass that bundles the five ``StandardScaler``
  objects (one per feature group) fitted on the training split.

* **load_dataset**: The top-level entry point used by the training script.
  It builds (or loads) the dataset, splits it into train/val/test, fits (or
  loads) the scalers, applies normalisation, and returns everything.

* **GenevaDataset / load_gva_dataset**: Legacy support for the older Geneva
  pickle-based graph format.  Not used in the main training pipeline.
"""

import json
import pickle
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data.data import BaseData
from tqdm import tqdm

from activitygraphs.config import Config
from activitygraphs.dataprocessing import SPATIAL_FEATURE_NAMES, convert_to_torch, load_data
from activitygraphs.ml.metrics import compute_home_hop_distance
from activitygraphs.utils import get_project_root

# Full-`x` column index of `is_home` in the legacy GenevaDataset pickles only.
# The main ActivityDataset path derives this from the data (see ActivityDataset.is_home_col_idx).
_LEGACY_IS_HOME_COL_IDX = 37


class GenevaDataset(pyg.data.InMemoryDataset):
    """
    Description: Legacy in-memory dataset for the original Geneva travel survey.

    Wraps a list of per-user PyTorch-Geometric graph objects that were
    previously built and saved as a Python pickle file.  Each item in the
    list is one user's graph.

    This class is kept for backward compatibility.  New code should use
    ``ActivityDataset`` instead.

    Attributes:
      - _graphs (list): The list of ``pyg.data.Data`` graph objects loaded
        from the pickle file.
    """

    def __init__(self, dataset_path: Path):
        """
        Description: Load a pickled list of graphs from disk.

        Input:
          - dataset_path (Path): Path to the ``.pickle`` file containing
            the list of per-user graph objects.

        Output:
          - (GenevaDataset): Initialised dataset.
        """
        super().__init__()

        print("Loading dataset...")

        # Open the pickle file in binary-read mode and deserialise the list
        with open(dataset_path, "rb") as f:
            graphs = pickle.load(f)

        print(f"Loaded {len(graphs)} graphs. Applying transforms...")

        # Store all graphs in memory for fast indexed access
        self._graphs = graphs

        print("Transforms applied.")

    @property
    def num_classes(self):
        """
        Description: Return the number of prediction classes.

        Output:
          - (int): Always returns 1 because this is a binary (visited / not
            visited) prediction problem.
        """
        return 1  # self._infer_num_classes(self._graphs[0].y)

    def len(self) -> int:
        """
        Description: Return the total number of user graphs in the dataset.

        Output:
          - (int): Number of users (graphs).
        """
        return len(self._graphs)

    def get(self, idx: int):
        """
        Description: Retrieve one user's graph by index.

        Input:
          - idx (int): Zero-based user index.

        Output:
          - (pyg.data.Data): The graph object for user ``idx``.
        """
        return self._graphs[idx]


class ActivityDataset(pyg.data.Dataset):
    """
    Description: Main PyTorch-Geometric dataset for travel-survey activity prediction.

    Architecture
    ~~~~~~~~~~~~
    All users share the **same** road/transit network graph (same nodes, same
    edges, same edge features).  What differs per user is:
      - ``spatial_features[i]``:  per-user, per-node features (e.g. visit counts,
        home indicator).  Shape: [num_users, num_nodes, num_spatial_features].
      - ``spatial_labels[i]``:    binary label — did user ``i`` visit node ``j``?
        Shape: [num_users, num_nodes, 1].
      - ``demographics[i]``:      user-level attributes (age, car ownership, …).
        Shape: [num_users, num_demographic_features].
      - ``distances[i]``:         per-user, per-node distance from user's home node.
        Shape: [num_users, num_nodes, 1].

    The five tensors / graph objects are saved to disk as ``.pt`` files the
    first time the dataset is constructed, and simply reloaded on every
    subsequent run (standard PyG processing protocol).

    ``get(i)`` builds a fresh ``pyg.data.Data`` for user ``i`` on the fly by
    concatenating the shared network node features with user ``i``'s spatial
    features.

    Class attributes:
      - PROCESSED_FILE_NAMES (list[str]): Names of the five cached files.

    Instance attributes:
      - network_graph (pyg.data.Data): The shared road-network graph.
      - spatial_features (torch.Tensor): All-user spatial node features,
        shape [num_users, num_nodes, num_spatial_features].
      - spatial_labels (torch.Tensor): All-user binary labels,
        shape [num_users, num_nodes, 1].
      - demographics (torch.Tensor): All-user demographic vectors,
        shape [num_users, num_demographic_features].
      - distances (torch.Tensor): All-user distance-from-home per node,
        shape [num_users, num_nodes, 1].
      - num_individuals (int): Number of users in the dataset.
      - num_nodes (int): Number of nodes in the network graph.
      - _is_scaled (bool): Flag preventing double-scaling by ``apply_scalers``.
    """

    # These are the five files that PyG's caching system expects after ``process()``
    PROCESSED_FILE_NAMES = [
        "network_graph.pt",      # the shared network graph (nodes + edges)
        "spatial_features.pt",   # per-user, per-node input features
        "spatial_labels.pt",     # per-user, per-node binary labels
        "demographics.pt",       # per-user demographic attributes
        "distances.pt",          # per-user, per-node distance from home
    ]

    def __init__(
        self,
        root: str,
        network_graph: pyg.data.Data | None = None,
        spatial_features: torch.Tensor | None = None,
        spatial_labels: torch.Tensor | None = None,
        demographics: torch.Tensor | None = None,
        distances: torch.Tensor | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ):
        """
        Description: Construct (or reload) the ActivityDataset.

        On the **first** call all five data arguments must be provided; the
        parent ``__init__`` triggers ``process()`` which saves them to
        ``root/processed/``.  On **subsequent** calls the arguments can be
        omitted (``None``) and the data is reloaded from the cached files.

        Input:
          - root (str): Directory used by PyG for caching; sub-folders
            ``raw/`` and ``processed/`` are created automatically.
          - network_graph (pyg.data.Data | None): The shared network graph.
          - spatial_features (torch.Tensor | None): Per-user spatial features,
            shape [num_users, num_nodes, num_spatial_features].
          - spatial_labels (torch.Tensor | None): Per-user binary labels,
            shape [num_users, num_nodes, 1].
          - demographics (torch.Tensor | None): Per-user demographics,
            shape [num_users, num_demographic_features].
          - distances (torch.Tensor | None): Per-user per-node distance from
            home, shape [num_users, num_nodes, 1].
          - transform (Callable | None): PyG on-the-fly transform applied to
            each graph returned by ``get()``.
          - pre_transform (Callable | None): Transform applied once during
            ``process()`` (e.g. positional encoding computation).
          - pre_filter (Callable | None): Filter applied during ``process()``.

        Output:
          - (ActivityDataset): Loaded and ready-to-use dataset.
        """
        # Temporarily store constructor arguments so ``process()`` can access them.
        # They are deleted after ``super().__init__`` to avoid keeping extra copies.
        self._network_graph_in = network_graph
        self._spatial_features_in = spatial_features
        self._spatial_labels_in = spatial_labels
        self._demographics_in = demographics
        self._distances_in = distances

        # PyG's __init__ calls process() if the processed files don't exist yet
        super().__init__(root, transform, pre_transform, pre_filter)

        # Clean up the temporary constructor arguments (no longer needed)
        del (
            self._network_graph_in,
            self._spatial_features_in,
            self._spatial_labels_in,
            self._demographics_in,
            self._distances_in,
        )

        # Load the five saved tensors / graph objects from disk
        processed_dir = Path(self.processed_dir)
        self.network_graph: pyg.data.Data = torch.load(processed_dir / "network_graph.pt", weights_only=False)
        # Shape: [num_users, num_nodes, num_spatial_features]
        self.spatial_features: torch.Tensor = torch.load(processed_dir / "spatial_features.pt", weights_only=True)
        # Shape: [num_users, num_nodes, 1]
        self.spatial_labels: torch.Tensor = torch.load(processed_dir / "spatial_labels.pt", weights_only=True)
        # Shape: [num_users, num_demographic_features]
        self.demographics: torch.Tensor = torch.load(processed_dir / "demographics.pt", weights_only=True)
        # Shape: [num_users, num_nodes, 1]
        self.distances: torch.Tensor = torch.load(processed_dir / "distances.pt", weights_only=True)

        # Validate that the network graph knows its own node count
        num_nodes = self.network_graph.num_nodes
        if num_nodes is None:
            raise ValueError("Network graph has `None` number of nodes")

        # Number of users in the dataset (first dimension of demographics)
        self.num_individuals: int = self.demographics.size(0)
        # Number of nodes in the shared network graph
        self.num_nodes: int = num_nodes
        # Guard flag to prevent accidental double-scaling
        self._is_scaled: bool = False

    @property
    def num_network_features(self) -> int:
        """
        Description: Return the number of network node-feature columns.

        The full node feature matrix ``x`` returned by ``get()`` concatenates
        network features first, then spatial features.  This property tells
        you where the network block ends and the spatial block begins.

        Output:
          - (int): Number of network node features.  Returns 0 when the
            network graph has no node features.
        """
        x = self.network_graph.x  # node feature matrix of the shared graph (may be None)
        return 0 if x is None else x.shape[1]

    @property
    def is_home_spatial_idx(self) -> int:
        """
        Description: Return the column index of ``is_home`` within the
        *spatial* feature block.

        The spatial feature block is ordered according to
        ``SPATIAL_FEATURE_NAMES`` defined in ``activitygraphs.dataprocessing``.
        This index is used when fitting scalers (the ``is_home`` column must
        be excluded from standardisation because it is binary).

        Output:
          - (int): Zero-based column index of ``is_home`` in the spatial block.
        """
        return SPATIAL_FEATURE_NAMES.index("is_home")

    @property
    def is_home_col_idx(self) -> int:
        """
        Description: Return the column index of ``is_home`` within the full
        concatenated node feature matrix ``x``.

        Since ``x = [network_features || spatial_features]``, the spatial
        columns start at index ``num_network_features``.

        Output:
          - (int): Zero-based column index of ``is_home`` in the full ``x``.
        """
        return self.num_network_features + self.is_home_spatial_idx

    @cached_property
    def home_hop_distance(self) -> np.ndarray:
        """
        Description: Compute and cache the all-pairs contiguity-hop distance
        matrix for the network graph.

        The hop distance between nodes ``u`` and ``v`` is the minimum number
        of edges that need to be traversed to go from ``u`` to ``v``.  This
        is used by the hop-band diagnostic metrics (see ``metrics.py``).

        The result is cached after the first access (``@cached_property``),
        so the expensive shortest-path computation runs at most once.

        Output:
          - (np.ndarray): Distance matrix of shape [num_nodes, num_nodes].
            Unreachable pairs are ``inf``.
        """
        return compute_home_hop_distance(self.network_graph.edge_index, self.num_nodes)

    @cached_property
    def median_realised_size(self) -> int:
        """
        Description: Compute and cache the median number of visited nodes
        per user (median |RG_i|).

        Used by ``run_experiment`` to set the ``k`` parameter (rank cutoff)
        for precision/recall/NDCG metrics so the cutoff adapts to the
        dataset's typical travel repertoire size.

        Output:
          - (int): Median visit count across all users.
        """
        # Collect the visit count (sum of binary labels) for every user
        sizes = [int(data.y.sum()) for data in self]
        return int(np.median(sizes))

    @property
    def raw_file_names(self) -> list[str]:
        """
        Description: List of raw input file names expected by PyG.

        Since ``ActivityDataset`` receives data directly through its
        constructor arguments rather than reading raw files, this returns
        an empty list (no raw files to download/check).

        Output:
          - (list[str]): Always an empty list.
        """
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """
        Description: List of processed file names that PyG checks to decide
        whether ``process()`` needs to run.

        If all five files already exist on disk, ``process()`` is skipped.

        Output:
          - (list[str]): The five ``.pt`` file names defined in
            ``PROCESSED_FILE_NAMES``.
        """
        return self.PROCESSED_FILE_NAMES

    def download(self) -> None:
        """
        Description: No-op download step.

        ``ActivityDataset`` does not download any files from the internet.
        This method exists only to satisfy the PyG Dataset interface.
        """
        pass

    def process(self):
        """
        Description: Validate the five input tensors and save them to disk.

        This method is called automatically by the PyG parent class on the
        *first* construction (when the processed files do not yet exist).
        It:
          1. Validates that node/user counts are consistent across tensors.
          2. Optionally applies ``pre_transform`` to the network graph
             (e.g. adds positional encodings).
          3. Saves all five objects as ``.pt`` files.

        Input (accessed via ``self._*_in`` attributes set in ``__init__``):
          - All five component Tensors / Data objects must be non-None.

        Output:
          - Five ``.pt`` files saved under ``self.processed_dir``.
        """
        # Retrieve the temporarily stored constructor arguments
        graph = self._network_graph_in   # shared network graph
        sf = self._spatial_features_in   # spatial features per user per node
        sl = self._spatial_labels_in     # binary labels per user per node
        demo = self._demographics_in     # demographic features per user
        dist = self._distances_in        # distance-from-home per user per node

        if graph is None or sf is None or sl is None or demo is None or dist is None:
            raise ValueError("First-time construction requires the five component Tensors/Data to be passed to init.")

        # Unpack dimension sizes for cross-validation
        num_sf_users, num_sf_nodes, _ = sf.shape        # spatial features shape
        num_sl_users, num_sl_nodes, num_sl_labels = sl.shape  # labels shape
        num_dm_users, _ = demo.shape                    # demographics shape
        num_di_users, num_di_nodes, num_di_feat = dist.shape  # distances shape

        # Ensure the node count matches across all tensors and the graph
        if not (graph.num_nodes == num_sf_nodes == num_sl_nodes == num_di_nodes):
            raise ValueError(
                f"Node count mismatch: network has {graph.num_nodes}, spatial_features has {num_sf_nodes}, "
                f"spatial_labels has {num_sl_nodes}, distances has {num_di_nodes}."
            )
        # Ensure the user count matches across all tensors
        if not (num_sf_users == num_sl_users == num_dm_users == num_di_users):
            raise ValueError(
                f"Individual count mismatch: spatial_features has {num_sf_users}, spatial_labels has {num_sl_users}, "
                f"demographics has {num_dm_users}, distances has {num_di_users}."
            )
        # Labels must have exactly one output column (binary)
        if num_sl_labels != 1:
            raise ValueError(f"spatial_labels last dim must be 1, got {num_sl_labels}.")
        # Distances must have exactly one column per node
        if num_di_feat != 1:
            raise ValueError(f"distances last dim must be 1, got {num_di_feat}.")

        # Make sure the network graph has node features, otherwise, fall back to a zero-width tensor.
        if graph.x is None:
            graph = graph.clone()
            # Create a placeholder with zero columns so concat operations work
            graph.x = torch.empty((graph.num_nodes, 0), dtype=sf.dtype)

        # Apply the pre-transform (e.g. positional encodings) to the network graph
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        # Save all five components as PyTorch tensors under the processed directory
        processed_dir = Path(self.processed_dir)
        torch.save(graph, processed_dir / "network_graph.pt")
        torch.save(sf, processed_dir / "spatial_features.pt")
        torch.save(sl, processed_dir / "spatial_labels.pt")
        torch.save(demo, processed_dir / "demographics.pt")
        torch.save(dist, processed_dir / "distances.pt")

    def len(self) -> int:
        """
        Description: Return the number of users (graphs) in the dataset.

        Output:
          - (int): Total number of individuals.
        """
        return self.num_individuals

    def get(self, i: int) -> pyg.data.Data:
        """
        Description: Build and return one user's graph on the fly.

        Concatenates the shared network node features with user ``i``'s
        spatial features to form the full node feature matrix ``x``.
        All other graph attributes (edge connectivity, edge features) come
        from the shared network graph.

        The returned object has the following key attributes:
          - ``x``:         full node feature matrix [num_nodes, num_net_feats + num_spatial_feats]
          - ``y``:         binary visit labels [num_nodes, 1]
          - ``edge_index``: graph connectivity [2, num_edges]
          - ``edge_attr``: edge features [num_edges, edge_dim]
          - ``graph_x``:   user demographics [1, num_demographic_feats]
          - ``user_id``:   integer user index [1]
          - ``distances``: per-node distance from home [num_nodes, 1]

        Input:
          - i (int): Zero-based user index in [0, num_individuals).

        Output:
          - (pyg.data.Data): User ``i``'s annotated graph.
        """
        if not 0 <= i < self.num_individuals:
            raise IndexError(f"Index {i} out of range for {self.num_individuals} individuals.")

        # Shared network node features (same for all users)
        network_x = self.network_graph.x
        # User-specific spatial features for this particular user
        spatial_x = self.spatial_features[i]

        # Handle the case where the network graph has no node features
        if network_x is None:
            network_x = torch.empty((self.num_nodes, 0), dtype=spatial_x.dtype)

        # Cast spatial features to same type as network graph features if mismatch
        if network_x.numel() > 0 and spatial_x.dtype != network_x.dtype:
            spatial_x = spatial_x.to(network_x.dtype)

        # Concatenate network and spatial features along the feature dimension (axis=1)
        # Result shape: [num_nodes, num_network_features + num_spatial_features]
        full_x = torch.cat([network_x, spatial_x], dim=1)

        # Binary labels: 1 if user i visited the node, 0 otherwise.  Shape: [num_nodes, 1]
        y = self.spatial_labels[i]
        # Demographic attribute vector for user i.  Shape: [num_demographic_features]
        graph_x = self.demographics[i]
        # Shared graph connectivity (COO format): [2, num_edges]
        edge_index = self.network_graph.edge_index
        # Shared edge feature matrix: [num_edges, edge_dim]
        edge_attr = self.network_graph.edge_attr

        # Build the PyG Data object with the standard node/edge fields
        data = pyg.data.Data(x=full_x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # Unsqueeze so batching works: shape becomes [1, num_demographic_features]
        # (when batched, graph_x is stacked along dim 0 producing [batch_size, num_demographic_features])
        data.graph_x = graph_x.unsqueeze(0)

        # Store the numeric user index so we can group metrics by user at eval time
        data.user_id = torch.tensor([i], dtype=torch.long)

        # Per-user distance-from-home for each node.  Shape: [num_nodes, 1]
        data.distances = self.distances[i].to(full_x.dtype)

        # Carry over any extra attributes from the network graph (pos, etc.)
        for key, value in self.network_graph:
            if key in ("x", "edge_index", "edge_attr", "y"):
                continue  # already added above; skip to avoid overwriting

            data[key] = value

        return data


@dataclass
class FittedScalers:
    """
    Description: Container holding the ``StandardScaler`` instances fitted on
    the training data for each feature group.

    Each scaler transforms its feature group to have roughly zero mean and
    unit variance.  Fitting is done only on the *training* subset so that
    the validation / test sets are scaled using training statistics only
    (avoiding data leakage).

    Attributes:
      - network_features (StandardScaler | None): Scaler for the shared
        network node features.  ``None`` if the network has no node features.
      - network_edges (StandardScaler | None): Scaler for the network edge
        features.  ``None`` if there are no edge features.
      - spatial (StandardScaler | None): Scaler for the per-user spatial node
        features.  ``None`` if all spatial columns were excluded.
      - demographics (StandardScaler | None): Scaler for the per-user
        demographic features.
      - distances (StandardScaler | None): Scaler for the per-user
        distance-from-home values.
      - exclude_spatial_cols (list[int] | None): Column indices that were
        *not* scaled in the spatial block (typically just ``is_home`` because
        it is binary).  Stored here so ``apply_scalers`` can skip them.
    """

    network_features: StandardScaler | None   # scaler for shared network node features
    network_edges: StandardScaler | None       # scaler for shared edge features
    spatial: StandardScaler | None             # scaler for per-user spatial features
    demographics: StandardScaler | None        # scaler for per-user demographic features
    distances: StandardScaler | None = None    # scaler for per-user distance-from-home
    exclude_spatial_cols: list[int] | None = None  # spatial columns skipped during scaling

    def save(self, path: Path) -> None:
        """
        Description: Serialise this FittedScalers object to a pickle file.

        Input:
          - path (Path): Destination file path (will be created or overwritten).

        Output:
          - None (writes file to disk).
        """
        with path.open("wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FittedScalers":
        """
        Description: Deserialise a FittedScalers object from a pickle file.

        Input:
          - path (Path): Path to a previously saved ``.pkl`` file.

        Output:
          - (FittedScalers): The loaded scaler container.
        """
        with path.open("rb") as f:
            return pickle.load(f)


def load_or_build_dataset(
    cfg: Config,
    project_root: Path | None = None,
    **build_kwargs,
) -> ActivityDataset:
    """
    Description: Load an existing ``ActivityDataset`` from disk, or build and
    save it from raw data if the processed files are missing.

    If the five processed ``.pt`` files already exist under
    ``cfg.data.paths.pyg_datasets/processed/``, the dataset is simply loaded.
    Otherwise, the raw travel-survey data is loaded from disk, converted to
    PyTorch tensors, and passed to ``ActivityDataset`` which triggers
    ``process()`` to save them.

    Input:
      - cfg (Config): Hydra configuration containing ``data.paths.pyg_datasets``,
        the path to the PyG dataset directory.
      - project_root (Path | None): Optional override for the project root.
        Auto-detected if ``None``.
      - **build_kwargs: Extra keyword arguments forwarded to the
        ``ActivityDataset`` constructor (e.g. ``pre_transform``).

    Output:
      - (ActivityDataset): The fully constructed and ready-to-use dataset.
    """
    # Resolve the project root (uses a sensible default if not provided)
    project_root = get_project_root(project_root)
    # Absolute path to the directory where the processed .pt files will be stored
    dataset_path = project_root / cfg.data.paths.pyg_datasets

    # Check whether all five expected processed files are already on disk
    if not all((dataset_path / file).exists() for file in ActivityDataset.PROCESSED_FILE_NAMES):
        # --- First-time build: load raw data and convert to tensors ---
        data, network_nodes, network_edges = load_data(cfg.data, project_root)
        # Convert raw DataFrames / arrays into PyTorch tensors and a PyG graph
        network_graph, spatial_features, spatial_labels, demographics, distances = convert_to_torch(
            data, network_nodes, network_edges
        )
        dataset_path = project_root / cfg.data.paths.pyg_datasets

        # Construct the dataset; __init__ triggers process() → saves .pt files
        return ActivityDataset(
            str(dataset_path), network_graph, spatial_features, spatial_labels, demographics, distances
        )

    # --- Subsequent runs: just load from the cached .pt files ---
    return ActivityDataset(root=str(dataset_path), **build_kwargs)


def split_indices(
    dataset: ActivityDataset,
    val_size: float,
    test_size: float,
    seed: int,
    cache_path: Path | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """
    Description: Split the dataset indices into train, validation, and test
    sets and optionally cache / reload them from a JSON file.

    The split is done in two steps to achieve the desired proportions:
      1. Hold out ``test_size`` fraction as the test set.
      2. Hold out ``val_size`` fraction from the *original* total as
         validation (re-scaled relative to what remains after removing test).

    Caching ensures that the same split is used across multiple runs with the
    same parameters, which is essential for fair comparisons.

    Input:
      - dataset (ActivityDataset): The full dataset (only its length is used).
      - val_size (float): Desired fraction of the *total* dataset for validation
        (e.g. 0.1 for 10 %).
      - test_size (float): Desired fraction of the *total* dataset for testing
        (e.g. 0.2 for 20 %).
      - seed (int): Random seed for reproducibility.
      - cache_path (Path | None): If provided, load the split from this JSON
        file if it exists, or save the newly created split to it.

    Output:
      - (tuple[list[int], list[int], list[int]]): Three lists of integer
        indices: ``(train_idx, val_idx, test_idx)``.
    """
    # If a cache file exists, reload the pre-computed split immediately
    if cache_path is not None and cache_path.exists():
        with cache_path.open() as f:
            indices = json.load(f)

        return indices["train"], indices["val"], indices["test"]

    # Step 1: split off the test set from all individuals
    train_val_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        random_state=seed,
    )

    # Step 2: split the remaining individuals into train and validation.
    # The validation fraction must be re-scaled relative to (1 - test_size)
    # so that it represents ``val_size`` of the *original* total.
    proportional_val_size = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=proportional_val_size,
        random_state=seed,
    )

    # Save the split to the cache file so future runs reuse the same partition
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump({"train": train_idx, "val": val_idx, "test": test_idx, "seed": seed}, f)

    return train_idx, val_idx, test_idx


def fit_scalers(
    dataset: ActivityDataset,
    train_idx: list[int],
    exclude_spatial_cols: list[int] | None = None,
    exclude_demographic_cols: list[int] | None = None,
) -> FittedScalers:
    """
    Description: Fit ``StandardScaler`` instances on the *training* split of
    each feature group and return them bundled in a ``FittedScalers`` object.

    Each ``StandardScaler`` centres features to zero mean and unit variance.
    Only training data is used for fitting so that test/validation statistics
    do not influence normalisation.

    Network node and edge features are shared across users so their scalers
    are fitted on the full network (not just the training users).

    Binary columns (e.g. ``is_home``) are excluded from spatial scaling
    because standardising a 0/1 column makes little sense.

    Input:
      - dataset (ActivityDataset): The unscaled full dataset.
      - train_idx (list[int]): Row indices of training users in the dataset.
      - exclude_spatial_cols (list[int] | None): Spatial feature column
        indices to skip during fitting and application (typically
        ``[is_home_spatial_idx]``).
      - exclude_demographic_cols (list[int] | None): Demographic column
        indices to skip.

    Output:
      - (FittedScalers): Fitted scalers for all five feature groups.
    """
    # Retrieve the shared network node and edge features
    network_features = dataset.network_graph.x    # shape: [num_nodes, num_net_feats]
    edge_attr = dataset.network_graph.edge_attr   # shape: [num_edges, edge_dim]

    # Network node features: no need to separate between train and test, the network is the same.
    network_scaler = None
    if network_features is not None and network_features.numel() > 0:
        network_scaler = StandardScaler()
        network_scaler.fit(network_features.numpy())

    # Network edge attributes: same as above (shared, not user-specific)
    edge_scaler = None
    if edge_attr is not None:
        edge_scaler = StandardScaler()
        edge_scaler.fit(edge_attr.numpy())

    # Spatial features: individual-specific, fit only on train. Skip when every column is excluded (e.g. only is_home).
    # Select only the training users' spatial features
    spatial_train = dataset.spatial_features[train_idx]   # shape: [len(train_idx), num_nodes, num_spatial]
    # Build a list of column indices to include (all except excluded ones)
    keep = [c for c in range(spatial_train.shape[-1]) if c not in (exclude_spatial_cols or [])]
    spatial_scaler = None
    if keep:
        # Flatten to 2D so StandardScaler sees rows of features, not 3D tensors
        spatial_flat = spatial_train[..., keep].reshape(-1, len(keep)).float().numpy()
        spatial_scaler = StandardScaler()
        spatial_scaler.fit(spatial_flat)

    # Demographics: individual-specific, fit only on train
    demo_train = dataset.demographics[train_idx]   # shape: [len(train_idx), num_demographic_feats]
    if exclude_demographic_cols:
        # Remove any columns that should be excluded from scaling
        keep = [c for c in range(demo_train.shape[-1]) if c not in exclude_demographic_cols]
        demo_train = demo_train[..., keep]
    demo_scaler = StandardScaler()
    demo_scaler.fit(demo_train.float().numpy())

    # Distance-from-home: individual-specific, fit only on train
    distances_train = dataset.distances[train_idx]   # shape: [len(train_idx), num_nodes, 1]
    distances_scaler = StandardScaler()
    # Reshape to 2D: [len(train_idx) * num_nodes, 1] so StandardScaler works correctly
    distances_scaler.fit(distances_train.reshape(-1, distances_train.shape[-1]).float().numpy())

    return FittedScalers(
        network_features=network_scaler,
        network_edges=edge_scaler,
        spatial=spatial_scaler,
        demographics=demo_scaler,
        distances=distances_scaler,
        exclude_spatial_cols=exclude_spatial_cols,
    )


def apply_scalers(dataset: ActivityDataset, scalers: FittedScalers) -> None:
    """
    Description: Apply the fitted scalers to the dataset's tensors **in-place**.

    Replaces each feature tensor on ``dataset`` with its standardised version
    using the scalers previously fitted by ``fit_scalers``.  The guard flag
    ``dataset._is_scaled`` is checked and set to prevent accidental
    double-application.

    Note: This modifies the dataset object directly — there is no return value.

    Input:
      - dataset (ActivityDataset): The dataset whose tensors will be rescaled.
      - scalers (FittedScalers): The fitted scalers from ``fit_scalers``.

    Output:
      - None (modifies ``dataset`` in-place).
    """
    # Guard: prevent calling this function twice on the same dataset object
    if dataset._is_scaled:
        raise RuntimeError("apply_scalers called twice on the same ActivityDataset")

    # Scale shared network node features (in-place replacement)
    if scalers.network_features is not None:
        x = dataset.network_graph.x.numpy()
        dataset.network_graph.x = torch.from_numpy(scalers.network_features.transform(x)).float()

    # Scale shared network edge features (in-place replacement)
    if scalers.network_edges is not None:
        ea = dataset.network_graph.edge_attr.numpy()
        dataset.network_graph.edge_attr = torch.from_numpy(scalers.network_edges.transform(ea)).float()

    # Scale per-user spatial features, skipping excluded columns (e.g. is_home)
    if scalers.spatial is not None:
        sf = dataset.spatial_features.float()
        n_cols = sf.shape[-1]                            # total number of spatial columns
        exclude = scalers.exclude_spatial_cols or []     # columns to leave unchanged
        keep = [c for c in range(n_cols) if c not in exclude]  # columns to scale

        # Flatten to 2D, select columns to scale, apply scaler, then restore shape
        flat = sf.reshape(-1, n_cols)[:, keep].numpy()
        scaled_keep = torch.from_numpy(scalers.spatial.transform(flat)).float()

        # Clone and selectively replace only the scaled columns
        result = sf.reshape(-1, n_cols).clone()
        result[:, keep] = scaled_keep
        dataset.spatial_features = result.reshape(sf.shape)

    # Scale per-user demographic features
    if scalers.demographics is not None:
        demo = dataset.demographics.float().numpy()
        dataset.demographics = torch.from_numpy(scalers.demographics.transform(demo)).float()

    # Scale per-user per-node distance-from-home values
    if scalers.distances is not None:
        dist = dataset.distances.float()
        shape = dist.shape                               # remember original shape for reshape
        flat = dist.reshape(-1, shape[-1]).numpy()       # flatten to 2D for scaler
        dataset.distances = torch.from_numpy(scalers.distances.transform(flat)).float().reshape(shape)

    # Mark the dataset as scaled to prevent double-application
    dataset._is_scaled = True


def load_dataset(
    cfg: Config,
    val_size: float,
    test_size: float,
    seed: int,
    project_root: Path | None = None,
    **build_kwargs,
) -> tuple[ActivityDataset, ActivityDataset, ActivityDataset, FittedScalers]:
    """
    Description: Top-level data pipeline — load, split, scale, and return the
    dataset as three subsets with the fitted scalers.

    Steps performed:
      1. Load or build the ``ActivityDataset`` (including positional encodings
         as a ``pre_transform`` if this is the first build).
      2. Load or compute the train/val/test index split (cached as a JSON file).
      3. Load or fit the ``StandardScaler`` objects on the training split
         (cached as a pickle file).
      4. Apply the scalers to the full dataset in-place.
      5. Return three ``ActivityDataset`` subsets (index slices) and the scalers.

    Cache files are named to encode the split seed and sizes so that different
    experimental configurations do not share caches.

    Input:
      - cfg (Config): Hydra configuration with ``data.paths.pyg_datasets``.
      - val_size (float): Fraction of total data for validation (e.g. 0.1).
      - test_size (float): Fraction of total data for testing (e.g. 0.2).
      - seed (int): Random seed for the train/val/test split.
      - project_root (Path | None): Optional project root override.
      - **build_kwargs: Extra keyword arguments forwarded to
        ``load_or_build_dataset``.

    Output:
      - (tuple): A 4-tuple
        ``(train_dataset, val_dataset, test_dataset, scalers)``
        where each dataset is an ``ActivityDataset`` view restricted to its
        split's indices.
    """
    # Resolve the project root and construct paths for the cache files
    project_root = get_project_root(project_root)
    pyg_dir = project_root / cfg.data.paths.pyg_datasets

    # Cache file names encode the split parameters for unambiguous retrieval
    splits_cache = pyg_dir / f"splits_{seed}_val{val_size}_test{test_size}.json"
    scalers_cache = pyg_dir / f"scalers_{seed}_val{val_size}_test{test_size}.pkl"

    # Pre-transform: add positional encodings to the network graph.
    # These are computed once during process() and saved with the graph.
    positional_encodings_transforms = T.Compose([
        T.AddRandomWalkPE(walk_length=20, attr_name=None),    # random-walk PE (length 20)
        T.AddLaplacianEigenvectorPE(k=8, attr_name=None),     # Laplacian eigenvector PE (8 dims)
    ])

    # Load or build the dataset; positional encodings are applied during first build
    dataset = load_or_build_dataset(
        cfg, project_root=project_root, pre_transform=positional_encodings_transforms, **build_kwargs
    )

    # Obtain train/val/test index lists (loaded from cache or freshly computed)
    train_idx, val_idx, test_idx = split_indices(dataset, val_size, test_size, seed, cache_path=splits_cache)

    # Obtain fitted scalers (loaded from cache or freshly fitted on training data)
    if scalers_cache.exists():
        scalers = FittedScalers.load(scalers_cache)
    else:
        # Exclude is_home from scaling (binary column should remain 0/1)
        scalers = fit_scalers(dataset, train_idx, exclude_spatial_cols=[dataset.is_home_spatial_idx])
        scalers.save(scalers_cache)

    # Apply the scalers to the full dataset in-place (all three splits share the tensors)
    apply_scalers(dataset, scalers)

    # Return index-sliced views of the dataset for each split plus the scalers
    return (
        cast(ActivityDataset, dataset[train_idx]),   # training subset
        cast(ActivityDataset, dataset[val_idx]),     # validation subset
        cast(ActivityDataset, dataset[test_idx]),    # test subset
        scalers,
    )


def load_gva_dataset(
    cfg: Config, test_size: float, seed: int, project_root: Path | None = None, graphs_name: str = "Graphs"
) -> tuple[GenevaDataset, GenevaDataset]:
    """
    Description: Load the legacy Geneva dataset, split it into train/test,
    fit scalers on the training set, and apply them to both splits.

    This function supports the older Geneva pickle-based format (``GenevaDataset``).
    Split and scaled datasets are cached as ``train.pickle`` and ``test.pickle``
    so that subsequent calls skip the expensive fitting step.

    Input:
      - cfg (Config): Hydra configuration with ``data.paths.pyg_datasets``.
      - test_size (float): Fraction of data for testing (e.g. 0.2 = 20 %).
      - seed (int): Random seed for the split.
      - project_root (Path | None): Optional project root override.
      - graphs_name (str): Base name of the pickle file containing the graphs
        (default ``"Graphs"`` → reads ``Graphs.pickle``).

    Output:
      - (tuple[GenevaDataset, GenevaDataset]): ``(train_dataset, test_dataset)``
        both with standardised node and edge features.
    """
    # Resolve the project root and build file paths
    project_root: Path = get_project_root(project_root)

    pyg_path = project_root / cfg.data.paths.pyg_datasets
    graphs_path = pyg_path / f"{graphs_name}.pickle"  # source file with all graphs

    # Cache paths for the split and processed datasets
    train_path = pyg_path / "train.pickle"
    test_path = pyg_path / "test.pickle"

    # If cached splits already exist, reload and return them immediately
    if train_path.exists() and test_path.exists():
        print(f"Loading {pyg_path} dataset...")

        with open(train_path, "rb") as f:
            train_dataset: GenevaDataset = pickle.load(f)

        with open(test_path, "rb") as f:
            test_dataset: GenevaDataset = pickle.load(f)

        return train_dataset, test_dataset

    print(f"Loading {graphs_path} graph list and building dataset...")

    # Build the full GenevaDataset from the source pickle
    dataset = GenevaDataset(graphs_path)
    # Split user indices into train and test
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=test_size, random_state=seed)

    # noinspection PyTypeChecker
    train_dataset: GenevaDataset = dataset[train_indices]   # training subset
    # noinspection PyTypeChecker
    test_dataset: GenevaDataset = dataset[test_indices]     # test subset

    print(f"Train size: {len(train_dataset)}. Splitting dataset: ")

    # Determine feature count and identify columns that are NOT the is_home binary flag
    num_features = train_dataset[0].x.shape[1]
    # non_home_cols: list of column indices to scale (all except the is_home column)
    non_home_cols = [i for i in range(num_features) if i != _LEGACY_IS_HOME_COL_IDX]

    # Stack all training node features (excluding is_home) for scaler fitting
    train_x = torch.cat([g.x[:, non_home_cols] for g in train_dataset]).numpy()
    # Stack all training edge attributes for scaler fitting
    train_edge_attr = torch.cat([g.edge_attr for g in train_dataset]).numpy()

    # Fit a StandardScaler for node features (trained on training data only)
    x_scaler = StandardScaler()
    x_scaler.fit(train_x)

    # Fit a StandardScaler for edge features (trained on training data only)
    e_scaler = StandardScaler()
    e_scaler.fit(train_edge_attr)

    print("Fitted scalers. Processing dataset:")

    # Transform both splits using the same scalers (fitted on training data only)
    for g in tqdm(train_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    for g in tqdm(test_dataset):
        replace_scaled_features(g, e_scaler, x_scaler, non_home_cols)

    print("Writing processed datasets")

    # Cache the scaled datasets so the next call can skip all of the above
    with open(train_path, "wb") as f:
        pickle.dump(train_dataset, f)

    with open(test_path, "wb") as f:
        pickle.dump(test_dataset, f)

    print("Done.")

    return train_dataset, test_dataset


def replace_scaled_features(g: BaseData, e_scaler: StandardScaler, x_scaler: StandardScaler, non_home_cols: list[int]):
    """
    Description: Scale node features and edge attributes of a single graph
    object **in-place**, skipping the ``is_home`` binary column.

    Input:
      - g (BaseData): A PyG graph whose ``x`` (node features) and
        ``edge_attr`` (edge features) will be modified.
      - e_scaler (StandardScaler): Fitted scaler for edge attributes.
      - x_scaler (StandardScaler): Fitted scaler for node features
        (applied only to ``non_home_cols``).
      - non_home_cols (list[int]): Column indices of non-binary node feature
        columns that should be standardised.

    Output:
      - None (modifies ``g`` in-place).
    """
    # Scale selected node feature columns (excluding the is_home binary flag)
    g.x[:, non_home_cols] = torch.tensor(x_scaler.transform(g.x[:, non_home_cols].numpy()), dtype=torch.float)
    # Scale all edge attributes
    g.edge_attr = torch.tensor(e_scaler.transform(g.edge_attr.numpy()), dtype=torch.float)
