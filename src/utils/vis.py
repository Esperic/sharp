# <Copyright 2022, Argo AI, LLC. Released under the MIT license.>
"""Visualization utils for Argoverse MF scenarios."""

import math
from pathlib import Path
from typing import Final, Optional, Sequence, Set, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from av2.datasets.motion_forecasting.data_schema import ArgoverseScenario, ObjectType
from av2.map.map_api import ArgoverseStaticMap
from av2.utils.typing import NDArrayFloat, NDArrayInt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.patches import Rectangle

_PlotBounds = Tuple[float, float, float, float]

# Configure constants
_OBS_DURATION_TIMESTEPS: Final[int] = 50
_PRED_DURATION_TIMESTEPS: Final[int] = 60

_ESTIMATED_VEHICLE_LENGTH_M: Final[float] = 4.5
_ESTIMATED_VEHICLE_WIDTH_M: Final[float] = 2.0
_ESTIMATED_CYCLIST_LENGTH_M: Final[float] = 1.8
_ESTIMATED_CYCLIST_WIDTH_M: Final[float] = 0.6
_PLOT_BOUNDS_BUFFER_W: Final[float] = 48
_PLOT_BOUNDS_BUFFER_H: Final[float] = 44

_CANVAS_COLOR: Final[str] = "#FCFBF8"
_DRIVABLE_AREA_COLOR: Final[str] = "#F7F7F7"
_LANE_SEGMENT_COLOR: Final[str] = "#BFC8CC"
_CROSSWALK_COLOR: Final[str] = "#D7D0C4"
_DEFAULT_ACTOR_COLOR: Final[str] = "#9CB1BDDA"
_CYCLIST_COLOR: Final[str] = "#82A996"
_PEDESTRIAN_COLOR: Final[str] = "#C49362"
_CONTEXT_ACTOR_EDGE_COLOR: Final[str] = "#718794"
_FOCAL_AGENT_COLOR: Final[str] = "#253F59"
_FOCAL_AGENT_EDGE_COLOR: Final[str] = "#182E43"
_HISTORY_COLOR: Final[str] = "#426985E1"
_CONTEXT_HISTORY_COLOR: Final[str] = "#718794"
_BEST_PREDICTION_COLOR: Final[str] = "#C96A50"
_OTHER_PREDICTION_COLOR = "#7C83B9"
_OTHER_ENDPOINT_COLOR = "#5F679C"
_GT_RIBBON_COLOR: Final[str] = "#47BE9A"
_GT_CENTERLINE_COLOR: Final[str] = "#4A968E"
_BOUNDING_BOX_ZORDER: Final[int] = 100

_STATIC_OBJECT_TYPES: Set[ObjectType] = {
    ObjectType.STATIC,
    ObjectType.BACKGROUND,
    ObjectType.CONSTRUCTION,
    ObjectType.RIDERLESS_BICYCLE,
}


def visualize_scenario(
    scenario: ArgoverseScenario,
    scenario_static_map: ArgoverseStaticMap,
    prediction: np.ndarray = None,
    timestep: int = 50,
    save_path: Path = None,
    title: str = "",
    tight=False,
    create_fig=True,
    show_future=True,
    show_history=True,
    show_map=True,
    best_pred=-1,
) -> None:
    if create_fig:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        ax = plt.gca()
    ax.figure.set_facecolor(_CANVAS_COLOR)
    ax.set_facecolor(_CANVAS_COLOR)
    ax.set_axis_off()
    if title != "": plt.title(title)

    # Plot static map elements and actor tracks
    if show_map: _plot_static_map_elements(scenario_static_map, True)
    plot_bounds, focal_gt = _plot_actor_tracks(
        ax, scenario, timestep, show_history, show_future, show_map
    )
    if show_future and focal_gt is not None:
        _plot_fading_ribbon(
            ax,
            focal_gt,
            width_m=3.0,
            color=_GT_RIBBON_COLOR,
            alpha_start=0.73,
            alpha_end=0.25,
            zorder=30,
        )
        ax.plot(
            focal_gt[:, 0],
            focal_gt[:, 1],
            color=_GT_CENTERLINE_COLOR,
            linewidth=0.75,
            alpha=0.45,
            zorder=31,
            solid_capstyle="round",
        )

    other_modes = None
    best_endpoint = None
    if prediction is not None:
        other_modes = prediction if best_pred < 0 else np.delete(prediction, best_pred, axis=0)
        _scatter_polylines(
            other_modes,
            ax,
            color=_OTHER_PREDICTION_COLOR,
            grad_color=False,
            alpha=0.82,
            linewidth=3.4,
            zorder=1000,
            arrow=False,
        )
        if best_pred >= 0:
            _scatter_polylines(
                prediction[best_pred][None],
                ax,
                color=_BEST_PREDICTION_COLOR,
                grad_color=False,
                alpha=1.0,
                linewidth=4.2,
                linestyle="-",
                zorder=1010,
                arrow=False,
            )
            best_endpoint = prediction[best_pred, -1]

    if other_modes is not None and len(other_modes):
        ax.scatter(
            other_modes[:, -1, 0],
            other_modes[:, -1, 1],
            s=36,
            marker="o",
            facecolors=_OTHER_ENDPOINT_COLOR,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.95,
            zorder=1015,
        )

    if best_endpoint is not None:
        ax.scatter(
            best_endpoint[0],
            best_endpoint[1],
            s=46,
            marker="o",
            facecolors="white",
            edgecolors="none",
            zorder=1017,
        )
        ax.scatter(
            best_endpoint[0],
            best_endpoint[1],
            s=48,
            marker="o",
            facecolors=_BEST_PREDICTION_COLOR,
            edgecolors="#914936",
            linewidths=0.65,
            zorder=1018,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(
        plot_bounds[0] - _PLOT_BOUNDS_BUFFER_W, plot_bounds[0] + _PLOT_BOUNDS_BUFFER_W
    )
    ax.set_ylim(
        plot_bounds[1] - _PLOT_BOUNDS_BUFFER_H, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_H
    )
    if tight: plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, pad_inches=0)
        plt.close()
    if create_fig: return fig


def _plot_fading_ribbon(
    ax,
    trajectory,
    width_m=1.7,
    color="#69B6AC",
    alpha_start=0.34,
    alpha_end=0.05,
    zorder=30,
):
    """Draw a vehicle-width GT ribbon fading from current to future."""
    xy = np.asarray(trajectory, dtype=float)[:, :2]
    if len(xy) < 2:
        return

    tangent = np.gradient(xy, axis=0)
    tangent /= np.clip(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-6, None)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    half_width = width_m / 2.0
    left, right = xy + half_width * normal, xy - half_width * normal
    quads = np.stack([left[:-1], left[1:], right[1:], right[:-1]], axis=1)
    alphas = np.linspace(alpha_start, alpha_end, len(quads))
    ax.add_collection(
        PolyCollection(
            quads,
            facecolors=[to_rgba(color, alpha) for alpha in alphas],
            edgecolors="none",
            antialiased=True,
            zorder=zorder,
        )
    )


def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    _plot_polygons(
        [area.xyz for area in static_map.vector_drivable_areas.values()],
        alpha=1.0,
        color=_DRIVABLE_AREA_COLOR,
    )
 
    for lane_segment in static_map.vector_lane_segments.values():
        centerline = static_map.get_lane_segment_centerline(lane_segment.id)
        _plot_polylines(
            [centerline],
            line_width=1.5,
            color=_LANE_SEGMENT_COLOR,
            alpha=0.88,
            zorder=5,
        )

    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                line_width=0.55,
                alpha=0.55,
                color=_CROSSWALK_COLOR,
                zorder=6,
            )


def _plot_actor_tracks(
    ax: plt.Axes,
    scenario: ArgoverseScenario,
    timestep: int,
    show_history: bool,
    show_future: bool,
    show_map: bool
) -> Tuple[Optional[NDArrayFloat], Optional[NDArrayFloat]]:
    """Plot all actor tracks (up to a particular time step) associated with an Argoverse scenario.

    Args:
        ax: Axes on which actor tracks should be plotted.
        scenario: Argoverse scenario for which to plot actor tracks.
        timestep: Tracks are plotted for all actor data up to the specified time step.

    Returns:
        track_bounds: (x_min, x_max, y_min, y_max) bounds for the extent of actor tracks.
    """
    track_bounds = None
    focal_future = None
    focal_id = scenario.focal_track_id

    for track in scenario.tracks:
        if track.track_id != focal_id and not show_map: continue
        # Get timesteps for which actor data is valid
        actor_timesteps: NDArrayInt = np.array(
            [
                object_state.timestep
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            continue

        future_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep > timestep
            ]
        )
        if len(future_trajectory.shape) > 1:
            future_trajectory = future_trajectory[:60, :]

        # Get actor trajectory and heading history
        history_trajectory: NDArrayFloat = np.array(
            [
                list(object_state.position)
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )
        actor_headings: NDArrayFloat = np.array(
            [
                object_state.heading
                for object_state in track.object_states
                if object_state.timestep <= timestep
            ]
        )

        is_focal = track.track_id == focal_id
        if is_focal:
            if len(future_trajectory) > 0:
                focal_future = future_trajectory
            track_bounds = history_trajectory[-1]

        if track.object_type in _STATIC_OBJECT_TYPES:
            continue

        if is_focal and show_history:
            _scatter_polylines(
                [history_trajectory],
                color=_HISTORY_COLOR,
                grad_color=False,
                linewidth=4.1,
                arrow=False,
                alpha=0.90,
                zorder=998,
            )
        elif (
            show_history
            and track.object_type == ObjectType.VEHICLE
            and len(history_trajectory) > 1
        ):
            _scatter_polylines(
                [history_trajectory[-20:]],
                color=_CONTEXT_HISTORY_COLOR,
                grad_color=False,
                linewidth=3.8,
                arrow=False,
                alpha=0.68,
                zorder=190,
            )

        if is_focal:
            track_color = _FOCAL_AGENT_COLOR
        elif track.object_type == ObjectType.VEHICLE:
            track_color = _DEFAULT_ACTOR_COLOR
        elif track.object_type in (ObjectType.CYCLIST, ObjectType.MOTORCYCLIST):
            track_color = _CYCLIST_COLOR
        else:
            track_color = _PEDESTRIAN_COLOR
        if track.object_type == ObjectType.VEHICLE:
            _plot_actor_bounding_box(
                ax,
                history_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M),
                is_focal,
            )
        elif (
            track.object_type == ObjectType.CYCLIST
            or track.object_type == ObjectType.MOTORCYCLIST
        ):
            _plot_actor_bounding_box(
                ax,
                history_trajectory[-1],
                actor_headings[-1],
                track_color,
                (_ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M),
                is_focal,
            )
        else:
            plt.plot(
                history_trajectory[-1, 0],
                history_trajectory[-1, 1],
                "o",
                color=track_color,
                markeredgecolor=(
                    _CONTEXT_ACTOR_EDGE_COLOR if not is_focal else "white"
                ),
                markeredgewidth=0.45 if not is_focal else 1.1,
                markersize=5.0 if not is_focal else 7.0,
                alpha=0.82 if not is_focal else 1.0,
                zorder=1030 if is_focal else 200,
            )

    return track_bounds, focal_future


class HandlerColorLineCollection(HandlerLineCollection):
    def __init__(
        self,
        reverse: bool = False,
        marker_pad: float = ...,
        numpoints: None = ...,
        **kwargs,
    ) -> None:
        super().__init__(marker_pad, numpoints, **kwargs)
        self.reverse = reverse

    def create_artists(
        self, legend, artist, xdescent, ydescent, width, height, fontsize, trans
    ):
        x = np.linspace(0, width, self.get_numpoints(legend) + 1)
        y = np.zeros(self.get_numpoints(legend) + 1) + height / 2.0 - ydescent
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=artist.cmap, transform=trans)
        lc.set_array(x if not self.reverse else x[::-1])
        lc.set_linewidth(artist.get_linewidth())
        return [lc]


def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
    endpoint: bool = False,
    **kwargs,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
            **kwargs,
        )
        if endpoint:
            plt.scatter(polyline[0, 0], polyline[0, 1], color=color, s=15, **kwargs)


def get_polyline_arc_length(xy: np.ndarray) -> np.ndarray:
    """Get the arc length of each point in a polyline"""
    diff = xy[1:] - xy[:-1]
    displacement = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    arc_length = np.cumsum(displacement)
    return np.concatenate((np.zeros(1), arc_length), axis=0)


def interpolate_lane(xy: np.ndarray, arc_length: np.ndarray, steps: np.ndarray):
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def interpolate_centerline(xy: np.ndarray, n_points: int):
    arc_length = get_polyline_arc_length(xy)
    steps = np.linspace(0, arc_length[-1], n_points)
    xy_inter = np.empty((steps.shape[0], 2), dtype=xy.dtype)
    xy_inter[:, 0] = np.interp(steps, xp=arc_length, fp=xy[:, 0])
    xy_inter[:, 1] = np.interp(steps, xp=arc_length, fp=xy[:, 1])
    return xy_inter


def _scatter_polylines(
    polylines: Sequence[NDArrayFloat],
    cmap="spring",
    linewidth=3,
    arrow: bool = True,
    reverse: bool = False,
    alpha=0.5,
    zorder=100,
    grad_color: bool = True,
    color=None,
    linestyle="-",
    halo=False,
    halo_width=1.0,
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    ax = plt.gca()
    for polyline in polylines:
        inter_poly = interpolate_centerline(polyline, 60)

        if arrow:
            point = inter_poly[-1]
            diff = inter_poly[-1] - inter_poly[-2]
            diff = diff / np.linalg.norm(diff)
            if grad_color:
                c = plt.cm.get_cmap(cmap)(0)
            else:
                c = color
            arrow = ax.quiver(
                point[0],
                point[1],
                diff[0],
                diff[1],
                alpha=alpha,
                scale_units="xy",
                scale=0.25,
                minlength=0.5,
                zorder=zorder - 1,
                color=c,
            )

        if grad_color:
            arc = get_polyline_arc_length(inter_poly)
            polyline = inter_poly.reshape(-1, 1, 2)
            segment = np.concatenate([polyline[:-1], polyline[1:]], axis=1)
            norm = plt.Normalize(arc.min(), arc.max())
            lc = LineCollection(
                segment, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha
            )
            lc.set_array(arc if not reverse else arc[::-1])
            lc.set_linewidth(linewidth)
            ax.add_collection(lc)
        else:
            line, = ax.plot(
                inter_poly[:, 0],
                inter_poly[:, 1],
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                alpha=alpha,
                linestyle=linestyle,
                solid_capstyle="round",
                solid_joinstyle="round",
                dash_capstyle="round",
            )
            if halo:
                line.set_path_effects(
                    [
                        pe.Stroke(
                            linewidth=linewidth + halo_width,
                            foreground="white",
                            alpha=0.95,
                        ),
                        pe.Normal(),
                    ]
                )


def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(
            polygon[:, 0],
            polygon[:, 1],
            fc=to_rgba(color, alpha),
            ec="none",
            linewidth=0,
            zorder=2,
        )


def _plot_actor_bounding_box(
    ax: plt.Axes,
    cur_location: NDArrayFloat,
    heading: float,
    color: str,
    bbox_size: Tuple[float, float],
    is_focal: bool,
) -> None:
    """Plot an actor bounding box centered on the actor's current location.

    Args:
        ax: Axes on which actor bounding box should be plotted.
        cur_location: Current location of the actor (2,).
        heading: Current heading of the actor (in radians).
        color: Desired color for the bounding box.
        bbox_size: Desired size for the bounding box (length, width).
    """
    (bbox_length, bbox_width) = bbox_size

    # Compute coordinate for pivot point of bounding box
    d = np.hypot(bbox_length, bbox_width)
    theta_2 = math.atan2(bbox_width, bbox_length)
    pivot_x = cur_location[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = cur_location[1] - (d / 2) * math.sin(heading + theta_2)

    if is_focal:
        ax.add_patch(
            Rectangle(
                (pivot_x, pivot_y),
                bbox_length,
                bbox_width,
                angle=np.degrees(heading),
                zorder=1029,
                fc="none",
                ec="white",
                linewidth=1.4,
            )
        )

    vehicle_bounding_box = Rectangle(
        (pivot_x, pivot_y),
        bbox_length,
        bbox_width,
        angle=np.degrees(heading),
        zorder=1030 if is_focal else _BOUNDING_BOX_ZORDER + 100,
        fc=color,
        ec=_FOCAL_AGENT_EDGE_COLOR if is_focal else _CONTEXT_ACTOR_EDGE_COLOR,
        linewidth=0.85 if is_focal else 0.55,
        alpha=1.0 if is_focal else 0.86,
    )
    ax.add_patch(vehicle_bounding_box)
