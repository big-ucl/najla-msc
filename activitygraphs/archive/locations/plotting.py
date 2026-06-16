import base64
import itertools
from io import BytesIO
from typing import Protocol

import altair as alt
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import polars as pl
from archive.exploration.dataprocessing import Purpose
from archive.exploration import ActivityGraph
from archive.exploration.metrics import Metrics
from matplotlib.axes import Axes

# Ordered list of activity purposes from most to least "important" for choosing a node's display colour/label.
# When a node is associated with multiple purposes, the first matching purpose in this list is used.
PURPOSE_IMPORTANCE = [Purpose.HOME, Purpose.WORK, Purpose.EDUCATION]

alt.data_transformers.enable("vegafusion")


def line_styles_by_key(G: nx.MultiDiGraph, key: str = "person_id"):
    """
    Description: Assigns a distinct matplotlib line style (solid, dotted, dash-dot, dashed) to
    each unique value of the given edge attribute key (e.g. person_id). Returns a list of line
    styles in the same order as the graph edges. This allows trips by different people to be
    visually distinguished when overlaid on the same plot.

    Input:
      - G (nx.MultiDiGraph): The NetworkX multi-directed graph whose edges carry the key attribute.
      - key (str): The edge attribute to use for grouping. Defaults to 'person_id'.

    Output:
      - (list[str]): A list of matplotlib line style strings (e.g. '-', ':', '-.', '--'),
        one per edge, in graph edge order.
    """
    # Extract the key attribute value from each edge (u, v, data[key])
    person_ids = [person_id for _, _, person_id in G.edges.data(data=key)]
    person_id_set = set(person_ids)  # Unique values, used to assign one style per person

    # Cycle through line styles so each unique person gets a different style
    line_styles = itertools.cycle(["-", ":", "-.", "--"])
    # Map each unique person_id to a line style
    line_person_mapping = {k: ls for k, ls in zip(person_id_set, line_styles)}

    return [line_person_mapping[person_id] for person_id in person_ids]


def _find_main_value(values, incomplete_ordering):
    """
    Description: Given a set of values and a priority ordering, returns the most important value.
    If any value matches an element in the ordering, returns the first match (highest priority).
    If none of the values appear in the ordering, returns the first value not in the ordering.
    This is used to pick the "main purpose" of a node that serves multiple purposes.

    Input:
      - values: An iterable of values associated with a node (e.g. list of activity purposes).
      - incomplete_ordering: An ordered list of priority values (highest priority first).
        Does not need to cover all possible values.

    Output:
      - The highest-priority value from `values` according to `incomplete_ordering`.
    """
    values_set = set(values)  # Convert to set for O(1) membership testing

    # Return the first element of the ordering that appears in the values set (highest priority)
    for elem in incomplete_ordering:
        if elem in values_set:
            return elem

    # No match found in ordering: return any value not in the ordering (a "catch-all" fallback)
    return next(filter(lambda v: v not in incomplete_ordering, values))


def node_colours_by_purpose(G: nx.MultiDiGraph):
    """
    Description: Generates a list of colours for each node in the activity graph, based on the
    node's main activity purpose. Nodes serving multiple purposes are assigned the colour of the
    most important one (according to PURPOSE_IMPORTANCE). Colours visually distinguish activity
    types (e.g. home is purple, work is blue, shopping is red).

    Input:
      - G (nx.MultiDiGraph): The NetworkX graph whose nodes have a 'purposes' attribute
        containing a list of Purpose enum values.

    Output:
      - (list[str]): A list of hex colour strings, one per node, in graph node order.
    """
    node_colours = []  # Will hold one colour string per node

    for _, purposes in G.nodes.data(data="purposes"):
        main_purpose = _find_main_value(purposes, PURPOSE_IMPORTANCE)  # Pick the most important purpose
        colour = _map_purpose_to_colour(main_purpose)  # Map purpose to its display colour
        node_colours.append(colour)

    return node_colours


def node_short_labels_by_purpose(G: nx.MultiDiGraph) -> dict[str, str]:
    """
    Description: Generates a dictionary of short text labels for each node in the activity graph,
    based on the node's main activity purpose. These abbreviations (e.g. 'H' for home, 'W' for
    work) are displayed inside nodes when drawing the graph, allowing quick purpose identification.

    Input:
      - G (nx.MultiDiGraph): The NetworkX graph whose nodes have a 'purposes' attribute
        containing a list of Purpose enum values.

    Output:
      - (dict[str, str]): A mapping from node ID to its short label string (e.g. {'loc_A': 'H'}).
    """
    node_labels = {}  # Will map each node ID to its short display label

    for n, purposes in G.nodes.data(data="purposes"):
        main_purpose = _find_main_value(purposes, PURPOSE_IMPORTANCE)  # Pick the most important purpose
        short_label = _map_purpose_to_short_label(main_purpose)  # Map to short string (e.g. 'H', 'W')
        node_labels[n] = short_label

    return node_labels


def _map_purpose_to_colour(purpose: Purpose):
    """
    Description: Maps a single activity Purpose enum value to its corresponding hex colour string
    for visualisation. Each activity category has a distinct colour to aid interpretation.

    Input:
      - purpose (Purpose): An activity purpose enum value (e.g. Purpose.HOME, Purpose.WORK).

    Output:
      - (str): A hex colour string (e.g. '#6929c4') for use in matplotlib/networkx drawing.
    """
    match purpose:
        case Purpose.HOME:
            return "#6929c4"
        case Purpose.WORK | Purpose.EDUCATION:
            return "#1192e8"
        case Purpose.WORK_DELIVERY | Purpose.WORK_OTHER:
            return "#005d5d"
        case Purpose.ENTERTAINMENT | Purpose.SPORT | Purpose.LEISURE:
            return "#9f1853"
        case Purpose.SHOPPING_FOOD | Purpose.SHOPPING_OTHER:
            return "#fa4d56"
        case Purpose.PERSONAL_BUSINESS:
            return "#570408"
        case Purpose.HOTEL:
            return "#198038"
        case Purpose.ESCORT_WORK | Purpose.ESCORT_HEALTH | Purpose.ESCORT_SCHOOL | Purpose.ESCORT_OTHER:
            return "#002d9c"
        case Purpose.WORSHIP:
            return "#ee538b"
        case Purpose.OTHER:
            return "#b28600"
        case Purpose.HEALTH:
            return "#009d9a"
        case Purpose.SOCIAL_VISIT | Purpose.SOCIAL_OTHER:
            return "#012749"
        case _:
            raise ValueError(purpose)


def _map_purpose_to_short_label(purpose: Purpose):
    """
    Description: Maps a single activity Purpose enum value to a short text abbreviation for use
    as a node label in graph visualisations (e.g. 'H' for Home, 'W' for Work, 'Sh' for Shopping).

    Input:
      - purpose (Purpose): An activity purpose enum value.

    Output:
      - (str): A short abbreviation string (typically 1–2 characters) identifying the purpose.
    """
    match purpose:
        case Purpose.HOME:
            return "H"
        case Purpose.WORK:
            return "W"
        case Purpose.EDUCATION:
            return "Ed"
        case Purpose.WORK_DELIVERY:
            return "Wd"
        case Purpose.WORK_OTHER:
            return "Wo"
        case Purpose.ENTERTAINMENT | Purpose.SPORT | Purpose.LEISURE:
            return "L"
        case Purpose.SHOPPING_FOOD | Purpose.SHOPPING_OTHER:
            return "Sh"
        case Purpose.PERSONAL_BUSINESS | Purpose.HOTEL:
            return "P"
        case Purpose.ESCORT_WORK | Purpose.ESCORT_HEALTH | Purpose.ESCORT_SCHOOL | Purpose.ESCORT_OTHER:
            return "Es"
        case Purpose.WORSHIP:
            return "Wo"
        case Purpose.OTHER:
            return "O"
        case Purpose.HEALTH:
            return "H"
        case Purpose.SOCIAL_VISIT | Purpose.SOCIAL_OTHER:
            return "So"
        case _:
            raise ValueError(purpose)


def _ax_centered_text(text: str, ax: plt.Axes):
    """
    Description: Renders a text string centered in the middle of a matplotlib Axes. Used to
    display informational messages (e.g. 'No activities.') in an empty plot area.

    Input:
      - text (str): The string to display in the centre of the axes.
      - ax (plt.Axes): The matplotlib Axes on which to draw the text.

    Output:
      - None. Modifies the Axes object in place.
    """
    left, width = 0.25, 0.5  # Horizontal bounds of the text box in axes-relative coordinates
    bottom, height = 0.25, 0.5  # Vertical bounds of the text box in axes-relative coordinates
    right = left + width  # Right edge of the text box
    top = bottom + height  # Top edge of the text box

    ax.text(
        0.5 * (left + right),  # Horizontal centre of the text box
        0.5 * (bottom + top),  # Vertical centre of the text box
        text,
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,  # Use axes-relative (0–1) coordinate system
        color="black",
    )


def draw_hh_graph(
    G: nx.MultiDiGraph,
    hh_id=None,
    line_style_key="person_id",
    node_colours=None,
    node_labels=None,
    use_coords=False,
):
    """
    Description: Draws a household (HH) activity graph using NetworkX and matplotlib. Each node
    represents a visited location, coloured by its main activity purpose. Each directed edge
    represents a trip between locations, styled by person ID so different household members'
    trips are visually distinguishable. Optionally uses geographic coordinates for node positions.

    Input:
      - G (nx.MultiDiGraph): The household activity graph to draw.
      - hh_id: The household ID for the plot title. If None, uses 'Activity graph'. Defaults to None.
      - line_style_key (str): Edge attribute to use for assigning distinct line styles.
        Defaults to 'person_id'.
      - node_colours (list | None): Custom list of node colours. If None, colours are assigned
        automatically by purpose. Defaults to None.
      - node_labels (dict | None): Custom node label mapping. If None, short purpose labels are used.
        Defaults to None.
      - use_coords (bool): If True, uses 'lon'/'lat' node attributes as positions. If False, uses
        Kamada-Kawai layout. Defaults to False.

    Output:
      - (tuple[plt.Figure, plt.Axes]): The matplotlib Figure and Axes objects.
    """
    title = "Activity graph" if hh_id is None else f"Act. graph of household: {hh_id}"

    fig, ax = plt.subplots()
    fig.set_facecolor("white")
    fig.set_size_inches(8, 6)

    ax.set_title(title, loc="left", color="black")
    ax.axis("off")

    line_styles = line_styles_by_key(G, key=line_style_key)
    node_colours = node_colours_by_purpose(G) if node_colours is None else node_colours
    node_labels = node_short_labels_by_purpose(G) if node_labels is None else node_labels

    if len(G.nodes) == 0:
        _ax_centered_text("No activities.", ax)

    if use_coords:
        pos = {node: (data["lon"], data["lat"]) for node, data in G.nodes(data=True)}
    else:
        pos = nx.layout.kamada_kawai_layout(G, weight="distance")

    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=node_colours)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, labels=node_labels, font_color="white")
    nx.draw_networkx_edges(G, pos=pos, ax=ax, connectionstyle="arc3,rad=0.1", style=line_styles)

    return fig, ax


def _plot_metric_histogram(metric_col: str, results: pl.DataFrame, bin_count=20) -> alt.Chart:
    """
    Description: Creates a single Altair bar chart showing the distribution (histogram) of one
    metric column from the results DataFrame.

    Input:
      - metric_col (str): The column name to plot on the x-axis (e.g. 'roc_auc', 'precision').
      - results (pl.DataFrame): A DataFrame containing per-sample or per-run metric values.
      - bin_count (int): Maximum number of histogram bins. Defaults to 20.

    Output:
      - (alt.Chart): An Altair Chart object (histogram bar chart) for the given metric.
    """
    return results.plot.bar(alt.X(metric_col).bin(maxbins=bin_count), alt.Y("count()"))


def plot_metric_histograms(metrics: Metrics, results: pl.DataFrame, n_cols=2) -> alt.Chart:
    """
    Description: Creates a grid of Altair histogram charts, one per metric, arranged in rows of
    n_cols. Each histogram shows the distribution of that metric across all samples or runs.
    Axes scales are resolved independently so each chart uses its own range.

    Input:
      - metrics (Metrics): A Metrics object providing the list of metric names to plot.
      - results (pl.DataFrame): A DataFrame containing the metric values to plot.
      - n_cols (int): Number of histogram charts per row in the grid. Defaults to 2.

    Output:
      - (alt.Chart): A nested Altair chart (vconcat of hconcat rows) with all metric histograms.
    """
    chart = alt.vconcat()  # Outer vertical concatenation container

    for batch in itertools.batched(metrics.names(), n_cols):
        row = alt.hconcat()  # Inner horizontal row container
        for metric_col in batch:
            row |= _plot_metric_histogram(metric_col, results)  # Add one histogram per metric in the row
        chart &= row  # Append the row to the vertical stack

    return chart.resolve_scale("independent")  # Each sub-chart uses its own axis scale


def geo_plot_mean_stat_by_postcode(
    mean_stats_by_postcode: pl.DataFrame,
    geo_postcode_shapes: gpd.GeoDataFrame,
    postcode_split: str,
    stat: str,
):
    """
    Description: Creates a geographic choropleth plot of a summary statistic (e.g. average
    node count or graph density) aggregated by postcode area. For most postcode splits, uses
    Altair's geoshape mark for an interactive web chart. For 'sector'-level splits (which have
    many fine-grained polygons), falls back to a static matplotlib plot for performance.

    Input:
      - mean_stats_by_postcode (pl.DataFrame): A DataFrame with one row per postcode area,
        containing the postcode column named by `postcode_split` and the `stat` column.
      - geo_postcode_shapes (gpd.GeoDataFrame): A GeoDataFrame with postcode geometry.
        Must contain a 'name' column matching the postcode values.
      - postcode_split (str): The column name for the postcode level (e.g. 'area', 'district', 'sector').
      - stat (str): The column name of the statistic to plot (e.g. 'mean_nodes', 'n_samples').

    Output:
      - An Altair Chart (interactive, for non-sector splits) or a matplotlib Figure
        (static, for sector splits).
    """
    mean_gdf = geo_postcode_shapes.merge(
        mean_stats_by_postcode.to_pandas(),
        left_on="name",
        right_on=postcode_split,
    )

    title = f"Average HH Graph {stat} by uk postcode {postcode_split}"

    if postcode_split != "sector":
        return (
            alt
            .Chart(mean_gdf, title=title)
            .mark_geoshape()
            .encode(color=stat, tooltip=["name", stat, "n_samples"])
            .properties(width=500, height=500)
        )

    else:
        fig, ax = plt.subplots()

        ax = mean_gdf.plot(
            ax=ax,
            column=stat,
            legend=True,
            legend_kwds={"label": stat, "orientation": "horizontal"},
        )

        ax.set_title(title)

        return fig


def geo_plot_mean_stat_by_municipality(
    mean_stats_by_municipality: pl.DataFrame,
    geo_municipality_shapes: gpd.GeoDataFrame,
    stat: str,
):
    """
    Description: Creates an interactive Altair geographic choropleth map of a summary statistic
    aggregated by municipality (local authority district). The map is reprojected to WGS84
    (EPSG:4326) for web rendering. Hovering shows the municipality name, its ID, the stat value,
    and the sample count.

    Input:
      - mean_stats_by_municipality (pl.DataFrame): A DataFrame with one row per municipality,
        containing 'municipality_id', 'municipality_name', 'n_samples', and the `stat` column.
      - geo_municipality_shapes (gpd.GeoDataFrame): A GeoDataFrame with municipality geometry.
        Must contain 'LAD24CD' (Local Authority District code) for merging.
      - stat (str): The name of the column in mean_stats_by_municipality to visualise.

    Output:
      - (alt.Chart): An interactive Altair geoshape choropleth chart, width=500 × height=500.
    """
    mean_gdf = geo_municipality_shapes.merge(
        mean_stats_by_municipality.to_pandas(), left_on="LAD24CD", right_on="municipality_id", how="right"
    ).to_crs("EPSG:4326")

    title = f"Average HH Graph {stat} by municipality"

    return (
        alt
        .Chart(mean_gdf, title=title)
        .mark_geoshape()
        .encode(color=stat, tooltip=["municipality_id", "municipality_name", stat, "n_samples"])
        .properties(width=500, height=500)
    )


def build_dash_graph_scatter(results: pl.DataFrame, graph: ActivityGraph):
    """
    Description: Builds an interactive Dash web application that displays a 3D scatter plot of
    households or trips. When the user hovers over a data point, a tooltip appears showing a
    rendered 2D activity graph image for that household. Requires Dash and Plotly to be installed.

    Input:
      - results (pl.DataFrame): A DataFrame with columns 'x', 'y', 'z' (3D coordinates for each
        point, e.g. from a dimensionality reduction like UMAP or PCA), 'c' (colour values), and
        'hh_id' (household ID for looking up the graph).
      - graph (ActivityGraph): An ActivityGraph object that can render household graphs via
        `graph.to_nx(hh_id)`.

    Output:
      - (Dash): A Dash application object. Call `.run()` to start the interactive server.
    """
    import plotly.graph_objects as go
    from dash import Dash, Input, Output, callback, dcc, html, no_update

    _fig = go.Figure(
        go.Scatter3d(
            x=results["x"],
            y=results["y"],
            z=results["z"],
            mode="markers",
            marker=dict(
                colorscale="viridis",
                color=results["c"],
                line={"color": "#444"},
                reversescale=True,
                sizeref=45,
                sizemode="diameter",
                opacity=0.8,
            ),
        ),
        layout=dict(
            width=1500,
            height=1000,
        ),
    )

    _fig.update_traces(hoverinfo="none", hovertemplate=None)

    app = Dash()

    app.layout = html.Div([
        dcc.Graph(id="graph-basic-2", figure=_fig, clear_on_unhover=True),
        dcc.Tooltip(id="graph-tooltip"),
    ])

    @callback(
        Output("graph-tooltip", "show"),
        Output("graph-tooltip", "bbox"),
        Output("graph-tooltip", "children"),
        Input("graph-basic-2", "hoverData"),
    )
    def display_hover(hoverData):
        """
        Description: Dash callback function triggered when the user hovers over a data point
        in the 3D scatter plot. Retrieves the household ID for the hovered point, renders its
        activity graph as a PNG image, and displays it in a tooltip overlay.

        Input:
          - hoverData (dict | None): Plotly hover event data. Contains the point index and
            bounding box. None if no point is being hovered.

        Output:
          - (tuple): A 3-tuple of (show: bool, bbox: dict, children: list) for the Dash Tooltip.
            show=True makes the tooltip visible, bbox positions it near the hovered point,
            children contains the rendered HTML with the graph image.
        """
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        pt = hoverData["points"][0]
        bbox = pt["bbox"]
        num = pt["pointNumber"]

        df_row = results.row(num, named=True)
        hh_id = df_row["hh_id"]

        buf = BytesIO()
        fig, _ = draw_hh_graph(graph.to_nx(hh_id), hh_id)
        fig.savefig(buf, format="png")
        plt.close()
        buf.seek(0)

        # noinspection PyTypeChecker
        image_string = base64.b64encode(buf.getvalue()).decode()
        image_string = f"data:image/png;base64,{image_string}"

        children = [
            html.Div(
                [
                    html.H2(f"HH Graph #{hh_id}", style={"color": "darkblue", "overflow-wrap": "break-word"}),
                    # html.P(f"Cluster: {c}"),
                    html.Img(src=image_string, style={"width": "100%"}),
                ],
                style={"width": "400px", "white-space": "normal"},
            )
        ]

        return True, bbox, children

    return app


""" ============================================================================ """


def _default_axes(ax: Axes = None) -> Axes:
    """
    Description: Returns the provided Axes object, or creates and returns a new one if None is
    given. This is a convenience helper used throughout the plotting module to allow callers
    to either provide their own Axes or let the function create its own figure.

    Input:
      - ax (Axes | None): An existing matplotlib Axes to draw on, or None to create a new one.
        Defaults to None.

    Output:
      - (Axes): Either the provided Axes, or a freshly created Axes with a 10×5 inch figure size.
    """
    return ax if ax is not None else plt.subplots(figsize=(10, 5))[1]


def _default_positions(G: nx.Graph, weight_name: str) -> dict[str, tuple[float, float]]:
    """
    Description: Computes default 2D spring-layout positions for graph nodes, where edges
    are weighted by INVERSE distance. Nodes connected by shorter edges are pulled closer
    together in the layout, giving a spatial representation that reflects travel proximity.
    Uses a fixed random seed for reproducibility.

    Input:
      - G (nx.Graph): The NetworkX graph to lay out.
      - weight_name (str): The edge attribute name that stores distance values (e.g. 'distance').

    Output:
      - (dict[str, tuple[float, float]]): A mapping from node ID to (x, y) position coordinates.
    """
    G = G.copy()  # Avoid modifying the original graph
    # Compute inverse-distance weights: closer nodes get higher weight (pulled together)
    weights = [(u, v, 1 / d) for u, v, d in G.edges(data=weight_name)]
    nx.set_node_attributes(G, name="weight", values=weights)
    return nx.spring_layout(G, weight="weight", seed=42)  # Fixed seed for reproducibility


class Graph(Protocol):
    """Protocol class emulating SyntheticGraph, as argument to `draw_synthetic_network`"""

    WEIGHT_NAME: str

    @property
    def G(self) -> nx.Graph:
        """
        Description: Returns the primary (simplified or pruned) NetworkX graph for this
        synthetic network. This is the graph normally used during training and evaluation —
        it may omit certain edges that exist in the fully-connected version.

        Output:
          - (nx.Graph): The primary NetworkX graph, where nodes are locations and edges
                are travel links with distance attributes named by WEIGHT_NAME.
        """
        pass

    @property
    def G_full(self) -> nx.Graph:
        """
        Description: Returns the fully-connected version of the NetworkX graph for this
        synthetic network. Unlike `G`, this version includes all possible edges between
        locations, making it useful for visualising the complete network topology and
        comparing against the pruned graph used during training.

        Output:
          - (nx.Graph): The fully-connected NetworkX graph, where every pair of nodes
                has a directed edge with a distance attribute named by WEIGHT_NAME.
        """
        pass


def draw_synthetic_network(graph: Graph, full=False, ax: Axes = None):
    """Draws a SyntheticGraph with shopping and work nodes highlighted

    Args:
        graph (Graph): the graph to draw
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _node_colour(node_attrs: dict) -> str:
        """
        Description: Determines the display colour for a single graph node based on its
        semantic attributes. Nodes are coloured to visually communicate their role in the
        synthetic network: shopping nodes are orange, workplace nodes are tomato-red, both
        are orangered, home/residential nodes are blue, and unclassified nodes are gray.

        Input:
          - node_attrs (dict): A dictionary of node attributes from the NetworkX graph
                (e.g. {'is_shopping': True, 'is_workplace': False, 'is_home': True, ...}).
                Must contain at least 'is_shopping' and 'is_workplace' keys; if either is
                absent, the node is treated as unclassified and coloured gray.

        Output:
          - (str): A matplotlib colour string identifying the node's display colour:
                'tab:gray'  — node lacks classification attributes,
                'orangered' — node is both a shopping location AND a workplace,
                'orange'    — node is a shopping location only,
                'tomato'    — node is a workplace only,
                'tab:blue'  — node is a regular (home/residential) location.
        """
        if "is_shopping" not in node_attrs or "is_workplace" not in node_attrs:
            return "tab:gray"  # Missing classification attributes; render as neutral gray

        if node_attrs["is_shopping"] and node_attrs["is_workplace"]:
            return "orangered"  # Both shopping and work — mixed-use node
        if node_attrs["is_shopping"]:
            return "orange"     # Shopping-only node
        if node_attrs["is_workplace"]:
            return "tomato"     # Workplace-only node

        return "tab:blue"  # Regular home/residential node

    G = graph.G_full if full else graph.G
    ax = _default_axes(ax)

    pos = _default_positions(G, graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)

    colors = [_node_colour(attrs) for _, attrs in G.nodes(data=True)]

    nx.draw_networkx(G, pos, node_color=colors, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

    return ax


class Schedules(Protocol):
    """Protocol class emulating SyntheticGraph, as argument to `draw_synthetic_trip`"""

    graph: Graph
    trip_df: pl.DataFrame


def draw_synthetic_trip(schedules: Schedules, person_id: int, full=False, ax: Axes = None):
    """Draws the SyntheticGraph in schedules with a person's schedule and trips overlain.

    Args:
        schedules (Schedules): The schedules from which to draw from
        person_id (int): the ID of the person whose schedule you want to draw
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _activities_to_colors(types: pl.Series):
        """
        Description: Converts a Polars Series of activity type codes for a single node into
        a single matplotlib colour string. This inner function is applied per-node to assign
        a colour that summarises the dominant activity type at that location for the given
        person's schedule. The priority ordering reflects how "important" each activity is
        for visual display: home (H) takes precedence, then mixed work+shopping, then
        shopping alone, then work alone.

        Input:
          - types (pl.Series): A Polars Series of activity type strings for one node,
                e.g. ['H'], ['W', 'S1'], ['S2']. Activity codes used are:
                  'H'  — home visit,
                  'W'  — work visit,
                  'S1' — shopping type 1 visit,
                  'S2' — shopping type 2 visit.

        Output:
          - (str): A matplotlib colour string for the node:
                'tab:blue'  — node was used as a home location,
                'orangered' — node was used for both work and shopping,
                'orange'    — node was used for shopping only,
                'tomato'    — node was used for work only.

        Raises:
          - NotImplementedError: If the combination of activity types does not match any
                expected pattern (indicates unexpected data or a missing case).
        """
        if "H" in types:
            return "tab:blue"    # Home node — highest visual priority
        if "W" in types and ("S1" in types or "S2" in types):
            return "orangered"   # Both work and shopping activity at this node
        if "S1" in types or "S2" in types:
            return "orange"      # Shopping-only node
        if "W" in types:
            return "tomato"      # Work-only node

        raise NotImplementedError("Impossible")  # Unreachable if data is valid

    graph = schedules.graph
    G = graph.G_full if full else graph.G
    ax = _default_axes(ax)

    pos = _default_positions(G, graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)

    trips = schedules.trip_df.filter(pl.col("person_id") == person_id)
    edgelist = trips.select("from_loc_id", "to_loc_id").rows()
    node_colours = (
        pl
        .concat([
            trips.select("from_loc_id", "from_type").rename({"from_loc_id": "loc_id", "from_type": "type"}),
            trips.select("to_loc_id", "to_type").rename({"to_loc_id": "loc_id", "to_type": "type"}),
        ])
        .group_by("loc_id")
        .agg(pl.col("type").map_elements(_activities_to_colors, return_dtype=pl.String).first())
        .join(pl.DataFrame({"loc_id": list(G.nodes())}), on="loc_id", how="right")
        .with_columns(pl.col("type").fill_null("tab:gray"))
    )["type"].to_list()

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colours)
    nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edgelist,
        arrows=True,
        arrowstyle="-|>",
        style="--",
        connectionstyle="arc3,rad=0.2",
        edge_color="red",
        ax=ax,
    )

    return ax


class Results(Protocol):
    """Protocol class emulating experiment.Results, as argument to `plot_training_progress`"""

    n_epochs: int
    name: str

    def final_losses(self) -> pl.DataFrame:
        """
        Description: Returns the final train, validation, and test losses recorded at the
        end of model training. Used by plot_training_progress to draw the horizontal test
        loss reference line and by plot_model_comparisons to compare models.

        Output:
          - (pl.DataFrame): A DataFrame (or tuple) with the final train, validation, and
                test loss values. The third value (index 2) is always the test loss.
        """
        pass

    def train_losses(self) -> pl.DataFrame:
        """
        Description: Returns the per-epoch training loss values over the full training run.
        Used by plot_training_progress to draw the train loss curve across epochs.

        Output:
          - (pl.DataFrame | list[float]): A sequence of training loss values, one per epoch,
                of length n_epochs. Each value is the average training loss for that epoch.
        """
        pass

    def val_losses(self) -> pl.DataFrame:
        """
        Description: Returns the per-epoch validation loss values over the full training run.
        Used by plot_training_progress and _plot_model_comparisons_line to draw the
        validation loss curve. Only meaningful for models with training history.

        Output:
          - (pl.DataFrame | list[float]): A sequence of validation loss values, one per epoch,
                of length n_epochs. Each value is the average validation loss for that epoch.
        """
        pass

    def test_loss(self) -> float:
        """
        Description: Returns the final scalar test loss for this model. Evaluated once on
        the held-out test set after training is complete. Used as the bar height in
        _plot_model_comparisons_bar and as the horizontal dashed line in
        _plot_model_comparisons_line.

        Output:
          - (float): The scalar cross-entropy test loss for this model.
        """
        pass

    def has_training_history(self) -> bool:
        """
        Description: Returns True if this Results object has per-epoch training and
        validation loss histories (i.e. it is a trained model), or False if it is a
        benchmark/baseline that was never trained (and therefore only has a test loss).
        Used by plot_model_comparisons to decide how to render each model's result.

        Output:
          - (bool): True if per-epoch loss histories are available, False otherwise.
        """
        pass


def plot_training_progress(results: Results, ax: Axes = None):
    """Plots the training, validation and test losses w.r.t the epochs.

    Args:
        results (experiment.Results): the results to be plotted
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """
    ax = _default_axes(ax)

    epochs = list(range(1, results.n_epochs + 1))
    _, _, test_loss = results.final_losses()

    ax.grid()
    ax.plot(epochs, results.train_losses(), label="Train")
    ax.plot(epochs, results.val_losses(), label="Validation")
    ax.plot([1, results.n_epochs], [test_loss, test_loss], label="Test", linestyle="dashed", linewidth=1)
    ax.set_title(f"{results.name} losses")
    ax.set_xlabel("Epoch")
    ax.set_xlim((1, results.n_epochs))
    ax.set_ylabel("CE Loss")
    ax.legend()

    return ax


def draw_prediction(graph: Graph, x: list, y_prob: list, full=False, labels=True, ax: Axes = None):
    """Draws the predictions of an ML model over the graph.

    Args:
        graph (Graph): the graph on which to draw the predictions
        x (list): the input node indicators (each node truthy if already selected, of length N)
        y_prob (list): the probabilities for each node (of length N)
        full (bool, optional): draw the fully-connected version of the graph. Defaults to False.
        labels (bool, optional): add the prediction values as node labels. Defaults to True.
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """

    def _node_colour(x, y):
        """
        Description:
            Decides the colour for a single graph node when drawing a weight-based
            activity graph. Nodes that were actually visited are shown in blue;
            unvisited nodes are shown on a grey colour scale proportional to their
            edge weight.

        Input:
          - x (bool): True if this node was visited by the individual, False otherwise.
          - y (float): a normalised weight value in [0, 1] used to pick a grey shade
                when the node was not visited.

        Output:
          - (str or tuple): a Matplotlib colour value — either the string
                "tab:blue" for visited nodes, or an RGBA tuple from the grey
                colormap for unvisited nodes.
        """
        if x:
            return "tab:blue"

        return mpl.colormaps["grey_r"](y)

    ax = _default_axes(ax)

    G = graph.G_full if full else graph.G
    pos = _default_positions(G, graph.WEIGHT_NAME)
    edge_labels = nx.get_edge_attributes(G, graph.WEIGHT_NAME)
    colors = [_node_colour(_x, _y) for _x, _y in zip(x, y_prob)]

    nx.draw_networkx(G, pos, node_color=colors, ax=ax, edgecolors="gray", font_color="DimGray")
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax)

    if labels:
        label_pos = {n: (x, y + 0.05) for n, (x, y) in pos.items()}
        neg_label = {n: f"{y:.2f}" for n, y in zip(G.nodes(), y_prob) if y < 0.5}
        pos_labels = {n: f"{y:.2f}" for n, y in zip(G.nodes(), y_prob) if y >= 0.5}
        nx.draw_networkx_labels(G, label_pos, neg_label, font_color="red", font_size=10, ax=ax)
        nx.draw_networkx_labels(G, label_pos, pos_labels, font_color="green", font_size=10, ax=ax)

    return ax


def plot_model_comparisons(*results: Results, how="bar", ax: Axes = None):
    """Plots a comparison between results from different models as a bar chart or as a line chart with training history.

    Args:
        how (str, optional): "bar" or "line". Defaults to "bar".
        ax (Axes, optional): the Axes on which to draw. Creates a new Axes if None. Defaults to None.

    Returns:
        Axes: the drawn axes
    """
    ax = _default_axes(ax)

    if how == "bar":
        return _plot_model_comparisons_bar(results, ax)
    elif how == "line":
        return _plot_model_comparisons_line(results, ax)

    raise KeyError(f"Unknown plot '{how}'. Valid entries are 'bar' or 'line'")


def _plot_model_comparisons_bar(results: tuple[Results, ...], ax: Axes = None):
    """
    Description: Creates a bar chart comparing the final test CE (cross-entropy) loss of multiple
    models side-by-side. Trained models are shown in orange and benchmark/baseline models are shown
    in blue. Each bar is labelled with its loss value to 4 decimal places.

    Input:
      - results (tuple[Results, ...]): A tuple of Results objects, one per model to compare.
      - ax (Axes | None): The matplotlib Axes to draw on. Required (not optional here).

    Output:
      - (Axes): The matplotlib Axes with the bar chart drawn on it.
    """
    xs = [r.name for r in results]  # Model names for the x-axis tick labels
    heights = [r.test_loss() for r in results]  # Test loss values as bar heights
    labels = [f"{h:.4f}" for h in heights]  # Formatted loss strings for bar labels

    # Orange for models with training history, blue for benchmarks that were not trained
    colors = ["tab:orange" if r.has_training_history() else "tab:blue" for r in results]

    ax.set_axisbelow(True)  # Draw grid lines below the bars
    b = ax.bar(xs, heights, color=colors)  # Draw the bars
    ax.bar_label(b, labels)  # Add numerical labels on top of each bar
    ax.grid()
    ax.set_title("Model comparison (Test CE loss)")
    ax.set_xlabel("Model")
    ax.set_ylabel("CE Loss")

    return ax


def _plot_model_comparisons_line(results: tuple[Results, ...], ax: Axes = None):
    """
    Description: Creates a line chart overlaying the validation loss curves of all trained models
    and horizontal dashed lines for benchmark (untrained) models. This allows direct comparison of
    training dynamics and final test performance. Each trained model is shown with its validation
    loss trajectory and its final test loss as a horizontal dashed line of the same colour.

    Input:
      - results (tuple[Results, ...]): A tuple of Results objects, one per model.
      - ax (Axes | None): The matplotlib Axes to draw on. Required (not optional here).

    Output:
      - (Axes): The matplotlib Axes with all model loss curves drawn on it.
    """
    max_epochs = max(res.n_epochs for res in results)  # Longest training run (for x-axis range)
    # Separate results into trained models (with val history) and static benchmarks
    trained_results = [res for res in results if res.has_training_history()]
    benchmark_results = [res for res in results if not res.has_training_history()]

    ax.grid()

    for trained_res in trained_results:
        test = trained_res.test_loss()  # Final test loss for the horizontal dashed reference line

        epochs = list(range(1, trained_res.n_epochs + 1))  # Epoch numbers for the x-axis
        # Plot validation loss curve; save the line handle to reuse its colour for the test line
        line = ax.plot(epochs, trained_res.val_losses(), label=f"{trained_res.name} (val)")
        # Plot the final test loss as a horizontal dashed line using the same colour
        ax.plot(
            [1, trained_res.n_epochs],
            [test, test],
            linestyle="dashed",
            linewidth=1,
            color=line[0].get_color(),
            label=trained_res.name,
        )

    for benchmark_res in benchmark_results:
        test = benchmark_res.test_loss()  # Single scalar test loss for this benchmark
        # Plot benchmark as a horizontal dashed line across the full epoch range
        ax.plot([1, max_epochs], [test, test], label=benchmark_res.name, linestyle="dashed", linewidth=1)

    ax.set_title("Model losses")
    ax.set_xlabel("Epoch")
    ax.set_xlim((1, max_epochs))
    ax.set_ylabel("BCE Loss")
    ax.legend()

    return ax
