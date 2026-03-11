import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


class RoadSceneRenderer:
    """
    Renders the road scene with lane markings and vehicles.

    Design:
      - render_road(ax, xlims): draw static road background (ideally once per Axes)
      - render_vehicles(ax, frame_df, test_vehicle_id): update cached Rectangle patches (fast)
    """

    def __init__(
        self,
        recording_meta,
        tracks_meta_df: pd.DataFrame,
        lane_color: str = "white",
        road_color: str = "#807D7D",
        median_color: str = "#BDDA90"
    ):
        self.recording_meta = recording_meta or {}
        self.tracks_meta_df = tracks_meta_df
        self.lane_color = lane_color
        self.road_color = road_color
        self.median_color = median_color

        # meta cache: id -> (width, height, drivingDirection)
        self._meta_by_id = {}
        try:
            for row in self.tracks_meta_df.itertuples(index=False):
                self._meta_by_id[int(row.id)] = (float(row.width), float(row.height), int(row.drivingDirection))
        except Exception:
            print(
                "[RoadSceneRenderer] Warning: could not cache tracksMeta. "
                "Will fall back to on-the-fly lookup (slower)."
            )

        # vehicle rectangle cache (per renderer instance / per Axes usage)
        self._rect_by_id = {}       # id -> Rectangle
        self._visible_last = set()  # ids visible last frame

        # static-road guard: avoid re-drawing road repeatedly on same Axes
        self._road_drawn_axes = set()

    # -------------------------
    # Road
    # -------------------------
    def render_road(self, ax: plt.Axes, xlims, window_height=50.0):
        """
        Draw road background + lane markings.

        Notes:
          - Intended to be called once per Axes (for performance).
          - If called again on the same Axes, it only updates limits.
        """
        ax_id = id(ax)
        if ax_id in self._road_drawn_axes:
            ax.set_xlim(xlims)
            return

        upper = self.recording_meta.get("upperLaneMarkings", []) or []
        lower = self.recording_meta.get("lowerLaneMarkings", []) or []

        if not upper or not lower:
            # Fallback: draw a blank-ish axis with sane limits
            ax.set_xlim(xlims)
            ax.set_ylim(0.0, window_height)
            ax.invert_yaxis()
            ax.set_aspect("equal")
            ax.set_xlabel("x (meters)")
            ax.set_ylabel("y (meters)")
            ax.set_title("Road Scene")
            self._road_drawn_axes.add(ax_id)
            return

        # road bounds
        road_top = float(upper[0])
        road_bottom = float(lower[-1])
        y_center = 0.5 * (road_top + road_bottom)
        ylims = (y_center - 0.5 * window_height, y_center + 0.5 * window_height)

        # median strip bounds between upper and lower lanes
        median_top = float(upper[-1])
        median_bottom = float(lower[0])

        offset = 0.5

        # draw background bands
        ax.fill_between(xlims, ylims[0], road_top - offset, color=self.median_color, zorder=0)
        ax.fill_between(xlims, road_bottom + offset, ylims[1], color=self.median_color, zorder=0)
        ax.fill_between(xlims, road_top - offset, road_bottom + offset, color=self.road_color, zorder=1)
        ax.fill_between(xlims, median_top + offset, median_bottom - offset, color=self.median_color, zorder=2)

        # lane markings (outer solid, inner dashed)
        for i, y in enumerate(upper):
            ls = "-" if (i == 0 or i == len(upper) - 1) else "--"
            ax.axhline(float(y), color=self.lane_color, linestyle=ls, linewidth=1, zorder=3)

        for i, y in enumerate(lower):
            ls = "-" if (i == 0 or i == len(lower) - 1) else "--"
            ax.axhline(float(y), color=self.lane_color, linestyle=ls, linewidth=1, zorder=3)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_xlabel("x (meters)")
        ax.set_ylabel("y (meters)")
        ax.set_title("Road Scene")

        self._road_drawn_axes.add(ax_id)

    # -------------------------
    # Vehicles
    # -------------------------
    def render_vehicles(self, ax: plt.Axes, frame_df: pd.DataFrame, test_vehicle_id=None):
        """
        Update cached rectangles for all vehicles in a single frame.

        frame_df must contain at least:
          - id or vehicle_id
          - x, y

        Performance:
          - reuses Rectangle patches (no re-add each frame)
          - hides rectangles that are not present in the current frame
        """
        if frame_df is None or frame_df.empty:
            # hide all currently visible rects
            for vid in list(self._visible_last):
                r = self._rect_by_id.get(vid)
                if r is not None:
                    r.set_visible(False)
            self._visible_last = set()
            return

        id_col = "id" if "id" in frame_df.columns else ("vehicle_id" if "vehicle_id" in frame_df.columns else None)
        if id_col is None:
            return

        test_vehicle_id = int(test_vehicle_id) if test_vehicle_id is not None else None
        seen = set()

        # itertuples is fast; attributes match column names
        for row in frame_df.itertuples(index=False):
            try:
                vid = int(getattr(row, id_col))
                x0 = float(getattr(row, "x"))
                y0 = float(getattr(row, "y"))
            except Exception:
                continue

            meta = self._meta_by_id.get(vid)
            if meta is None:
                # rare fallback
                try:
                    m = self.tracks_meta_df[self.tracks_meta_df["id"] == vid].iloc[0]
                    meta = (float(m["width"]), float(m["height"]), int(m["drivingDirection"]))
                    self._meta_by_id[vid] = meta
                except Exception:
                    continue

            width, height, direction = meta

            if test_vehicle_id is not None and vid == test_vehicle_id:
                color = "red"
            elif direction == 1:
                color = "blue"
            else:
                color = "green"

            rect = self._rect_by_id.get(vid)
            if rect is None:
                rect = Rectangle(
                    (x0, y0),
                    width,
                    height,
                    edgecolor=color,
                    facecolor="none",
                    lw=2,
                    zorder=4,
                )
                ax.add_patch(rect)
                self._rect_by_id[vid] = rect
            else:
                rect.set_xy((x0, y0))
                rect.set_visible(True)
                # important: test_vehicle_id differs per Axes in multi-vehicle mode
                rect.set_edgecolor(color)

            seen.add(vid)

        # hide rectangles that were visible last frame but are not present now
        for vid in (self._visible_last - seen):
            r = self._rect_by_id.get(vid)
            if r is not None:
                r.set_visible(False)

        self._visible_last = seen