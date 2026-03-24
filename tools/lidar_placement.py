"""
LiDAR Sensor Placement Tool
Based on: "Optimization and Validation of a 2D-LiDAR-Coverage Algorithm"
Noel D'Avis & Silvia Faquiri, PETRA '25

Coverage rule (Section 3):
  Each grid cell is divided into 4 triangles (TOP / RIGHT / BOTTOM / LEFT).
  Every triangle that is NOT wall- or obstacle-adjacent must be covered
  by at least one sensor from the correct direction.
  Wall/obstacle-adjacent triangle faces are pre-marked as covered.

Safety margin:
  1 grid-cell margin around walls and obstacles is excluded from the
  coverage requirement.

Obstacle categories:
  Category A: Signal-blocking obstacles (e.g. furniture).
              LiDAR rays are blocked. Cells behind them may be blind spots
              → coverage required. Safety margin applied around them.
              Drawn by click+drag (default).
  Category B: Unreachable areas (e.g. built-in wardrobes, pillars).
              A person cannot be there → excluded from coverage metric
              (neither numerator nor denominator). LiDAR rays pass through
              them transparently. No safety margin around them.
              Drawn by click+drag while holding Shift.

Requirements:
    pip install matplotlib numpy
"""

import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox
import json

# Colours
BG            = '#1e1e2e'
ROOM_FACE     = '#2a2a3e'
OBS_A_FACE    = '#555577'   # Category A obstacle fill
OBS_A_EDGE    = '#8888aa'
OBS_B_FACE    = '#335577'   # Category B obstacle fill (blue tint)
OBS_B_EDGE    = '#55aacc'
GRID_COLOR    = '#3a3a4e'
WALL_COLOR    = '#aaaacc'
SENSOR_COLORS = ['#ff4444', '#44ff88', '#4488ff', '#ffaa00', '#cc44ff', '#00ffff']

SENSOR_RANGE = 6.0   # metres (RPLidar A1M8)
MARGIN_CELLS = 1     # 1 grid-cell safety margin around walls and obstacles

TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
DIRS = [TOP, RIGHT, BOTTOM, LEFT]


# =============================================================================
# ROOM
# =============================================================================

class Room:
    def __init__(self, width, height, cell_size=0.10):
        self.width      = width
        self.height     = height
        self.cell_size  = cell_size
        self.obstacles  = []   # list of (x, y, w, h) in metres  – Category A
        self.obstacles_b = []  # list of (x, y, w, h) in metres  – Category B

    def grid_shape(self):
        c = int(round(self.width  / self.cell_size))
        r = int(round(self.height / self.cell_size))
        return c, r

    def cell_is_obstacle(self, ci, cj):
        """Returns True if cell is a Category-A obstacle (blocks LiDAR rays).
        Category-B cells are transparent to rays and are NOT included here."""
        cx = (ci + 0.5) * self.cell_size
        cy = (cj + 0.5) * self.cell_size
        for (ox, oy, ow, oh) in self.obstacles:   # Category A only
            if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
                return True
        return False

    def cell_is_obstacle_b(self, ci, cj):
        """Returns True only for Category-B (unreachable) cells."""
        cx = (ci + 0.5) * self.cell_size
        cy = (cj + 0.5) * self.cell_size
        for (ox, oy, ow, oh) in self.obstacles_b:
            if ox <= cx <= ox + ow and oy <= cy <= oy + oh:
                return True
        return False

    def potential_sensor_positions(self):
        """
        Candidate positions:
          1. The 4 room corners.
          2. The 4 corners of each obstacle (one cell outside each corner).
          3. Projections of each obstacle edge onto all 4 room walls.
        Only Category-A obstacles generate candidate positions (sensors are
        placed to resolve their blind spots). Category-B obstacles are
        unreachable, so placing a sensor there is pointless; they do however
        still block rays and are therefore excluded from candidate positions.
        """
        cols, rows = self.grid_shape()
        cs = self.cell_size
        pos = set()

        # 1. Room corners
        pos.add((0, 0))
        pos.add((cols - 1, 0))
        pos.add((0, rows - 1))
        pos.add((cols - 1, rows - 1))

        for (ox, oy, ow, oh) in self.obstacles:   # Category A only
            ci_left  = int(round(ox        / cs))
            ci_right = int(round((ox + ow) / cs))
            cj_bot   = int(round(oy        / cs))
            cj_top   = int(round((oy + oh) / cs))

            # 2. Obstacle corners: one cell diagonally outside
            for ci, cj in [
                (ci_left  - 1, cj_bot - 1),
                (ci_right,     cj_bot - 1),
                (ci_left  - 1, cj_top    ),
                (ci_right,     cj_top    ),
            ]:
                ci = max(0, min(cols - 1, ci))
                cj = max(0, min(rows - 1, cj))
                pos.add((ci, cj))

            # 3. Project obstacle x-edges onto bottom and top wall
            for ci in [ci_left - 1, ci_right]:
                ci = max(0, min(cols - 1, ci))
                pos.add((ci, 0))
                pos.add((ci, rows - 1))

            # 3. Project obstacle y-edges onto left and right wall
            for cj in [cj_bot - 1, cj_top]:
                cj = max(0, min(rows - 1, cj))
                pos.add((0, cj))
                pos.add((cols - 1, cj))

        # Exclude positions that fall inside Category-A obstacles only
        # (Cat-B is transparent, but a sensor placed there would monitor
        #  an unreachable area – exclude for cleanliness)
        return [p for p in pos
                if not self.cell_is_obstacle(*p)
                and not self.cell_is_obstacle_b(*p)]


# =============================================================================
# ALGORITHM HELPERS
# =============================================================================

def cell_in_margin(room, ci, cj):
    cols, rows = room.grid_shape()
    m = MARGIN_CELLS
    if ci < m or ci >= cols - m or cj < m or cj >= rows - m:
        return True
    # Margin only around Category-A obstacles (they block signal)
    # Category-B obstacles are transparent → no margin needed
    for (ox, oy, ow, oh) in room.obstacles:
        ci0 = int(ox        / room.cell_size)
        ci1 = int((ox + ow) / room.cell_size)
        cj0 = int(oy        / room.cell_size)
        cj1 = int((oy + oh) / room.cell_size)
        if (ci0 - m) <= ci <= (ci1 + m) and (cj0 - m) <= cj <= (cj1 + m):
            return True
    return False


def has_los(room, sx, sy, cx, cy):
    steps = int(max(abs(cx - sx), abs(cy - sy)) * 4) + 2
    cols, rows = room.grid_shape()
    for t in np.linspace(0, 1, steps):
        ri = int(sx + t * (cx - sx))
        rj = int(sy + t * (cy - sy))
        if 0 <= ri < cols and 0 <= rj < rows:
            if room.cell_is_obstacle(ri, rj):   # only Category A blocks rays
                return False
    return True


def get_covered_directions(sx, sy, cx, cy):
    dirs = []
    if sy > cy: dirs.append(TOP)
    if sx > cx: dirs.append(RIGHT)
    if sy < cy: dirs.append(BOTTOM)
    if sx < cx: dirs.append(LEFT)
    return dirs


def compute_coverage(room, pos):
    """Returns set of (ci, cj, direction) triangles visible from sensor at pos.
    Category-B cells are skipped (unreachable, not part of coverage metric).
    Rays pass through them transparently."""
    cols, rows = room.grid_shape()
    si, sj = pos
    covered = set()
    rc = SENSOR_RANGE / room.cell_size
    for ci in range(cols):
        for cj in range(rows):
            if room.cell_is_obstacle(ci, cj):     # skip Cat-A (solid)
                continue
            if room.cell_is_obstacle_b(ci, cj):   # skip Cat-B (unreachable)
                continue
            if (ci - si) ** 2 + (cj - sj) ** 2 > rc ** 2:
                continue
            cx, cy = ci + 0.5, cj + 0.5
            if has_los(room, si, sj, cx, cy):
                for d in get_covered_directions(si, sj, cx, cy):
                    covered.add((ci, cj, d))
    return covered


def build_required_triangles(room):
    """
    Returns dict { (ci, cj, direction): False } for every triangle face
    that needs sensor coverage.

    Excluded from the requirement:
      - Category-A obstacle cells  (solid, signal-blocking)
      - Category-B obstacle cells  (unreachable → excluded from coverage metric)
      - cells within MARGIN_CELLS of walls or Category-A obstacles
      - faces that directly touch a wall or Category-A obstacle
    """
    cols, rows = room.grid_shape()
    required = {}

    for ci in range(cols):
        for cj in range(rows):
            # Skip Category-A obstacle cells (solid)
            if room.cell_is_obstacle(ci, cj):
                continue
            # Skip Category-B obstacle cells (unreachable, not in metric)
            if room.cell_is_obstacle_b(ci, cj):
                continue
            # Skip margin cells (only Cat-A margins apply)
            if cell_in_margin(room, ci, cj):
                continue
            for d in DIRS:
                # Skip wall-touching faces
                if d == BOTTOM and cj == 0:        continue
                if d == TOP    and cj == rows - 1: continue
                if d == LEFT   and ci == 0:        continue
                if d == RIGHT  and ci == cols - 1: continue
                # Skip obstacle-touching faces (only Cat-A blocks)
                ni = ci + (1 if d == RIGHT else -1 if d == LEFT  else 0)
                nj = cj + (1 if d == TOP   else -1 if d == BOTTOM else 0)
                if 0 <= ni < cols and 0 <= nj < rows:
                    if room.cell_is_obstacle(ni, nj):   # Cat-A only
                        continue
                required[(ci, cj, d)] = False

    return required


# =============================================================================
# ALGORITHM
# =============================================================================

def run_algorithm(room):
    """
    Greedy sensor placement following D'Avis & Faquiri, PETRA '25.
    Category-B cells are excluded from coverage requirements but still
    block LiDAR rays (handled transparently via cell_is_obstacle / has_los).
    """
    pot      = room.potential_sensor_positions()
    sens_cov = {p: compute_coverage(room, p) for p in pot}
    covered  = build_required_triangles(room)

    total  = len(covered)
    cols, rows = room.grid_shape()
    print(f"Grid {cols}x{rows} | Required triangles: {total} | Candidates: {len(pot)}")

    def uncov_count():
        return sum(1 for v in covered.values() if not v)

    def score(p):
        cols, rows = room.grid_shape()
        n = sum(1 for tri in sens_cov[p] if tri in covered and not covered[tri])
        si, sj = p
        on_horiz = (sj == 0 or sj == rows - 1)
        on_vert  = (si == 0 or si == cols - 1)
        is_corner = int(on_horiz and on_vert)
        return (n, is_corner)

    def place(p):
        essential.append(p)
        eset.add(p)
        for tri in sens_cov[p]:
            if tri in covered:
                covered[tri] = True

    essential, eset = [], set()

    # Step 1: essential sensors (iterative)
    changed = True
    while changed:
        changed = False
        for tri, done in list(covered.items()):
            if done:
                continue
            covering = [p for p in pot if p not in eset and tri in sens_cov[p]]
            if len(covering) == 1:
                place(covering[0])
                changed = True

    print(f"Essential sensors: {len(essential)}")

    # Step 2: greedy fill
    iteration = 0
    while uncov_count() > 0:
        iteration += 1
        best, best_n = None, (0, 0)
        for p in pot:
            if p in eset:
                continue
            n = score(p)
            if n > best_n:
                best_n, best = n, p

        if best is None or best_n[0] == 0:
            print("Warning: full coverage not achievable with given candidates.")
            break

        place(best)
        pct = 100 * (1 - uncov_count() / total)
        print(f"  Iter {iteration}: sensor at {best}  coverage {pct:.1f}%")

    # Step 3: redundancy check
    removed = True
    while removed:
        removed = False
        for p in list(essential):
            others = [q for q in essential if q != p]
            cov_without = {k: False for k in covered}
            for q in others:
                for tri in sens_cov[q]:
                    if tri in cov_without:
                        cov_without[tri] = True
            if all(cov_without[k] for k in covered):
                essential.remove(p)
                eset.discard(p)
                covered.update(cov_without)
                print("  Redundancy: removed " + str(p))
                removed = True
                break

    print(f"Done: {len(essential)} sensor(s) placed.")
    return essential, covered, sens_cov


# =============================================================================
# GUI
# =============================================================================

class PlacementGUI:

    def __init__(self):
        self.room_w    = 4.0
        self.room_h    = 3.0
        self.cell_size = 0.10
        self.obstacles   = []    # Category A
        self.obstacles_b = []    # Category B

        self._drag_start    = None
        self._drag_rect     = None
        self._placing_cat_b = False   # True while Shift is held

        self.sensors  = []
        self.covered  = {}
        self.sens_cov = {}

        self._build_figure()
        self._draw_room()
        plt.show()

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor=BG)
        self.fig.canvas.manager.set_window_title(
            "LiDAR Placement Tool  -  D'Avis & Faquiri, PETRA '25")

        self.ax_panel = self.fig.add_axes([0.00, 0.00, 0.215, 1.00])
        self.ax_panel.set_facecolor('#13131f')
        self.ax_panel.axis('off')

        self.ax = self.fig.add_axes([0.225, 0.06, 0.755, 0.90])
        self.ax.set_facecolor(BG)
        self.ax.set_aspect('equal')

        def lbl(y, text, bold=False, color='#cccccc', size=9):
            self.ax_panel.text(0.08, y, text,
                               transform=self.ax_panel.transAxes,
                               color=color, fontsize=size, va='top',
                               fontweight='bold' if bold else 'normal')

        lbl(0.975, "LiDAR Placement Tool", bold=True, color='white', size=11)
        lbl(0.940, "D'Avis & Faquiri  PETRA '25", color='#666688', size=7)
        self._hline(0.915)

        lbl(0.895, "ROOM DIMENSIONS", bold=True, color='#8888bb', size=8)
        lbl(0.865, "Width (m)")
        lbl(0.800, "Height (m)")
        lbl(0.735, "Grid cell (m)")

        ax_w = self.fig.add_axes([0.01, 0.830, 0.185, 0.038])
        ax_h = self.fig.add_axes([0.01, 0.765, 0.185, 0.038])
        ax_g = self.fig.add_axes([0.01, 0.700, 0.185, 0.038])
        self.tb_w = TextBox(ax_w, '', initial=str(self.room_w),
                            color='#22223a', hovercolor='#33334a')
        self.tb_h = TextBox(ax_h, '', initial=str(self.room_h),
                            color='#22223a', hovercolor='#33334a')
        self.tb_g = TextBox(ax_g, '', initial=str(self.cell_size),
                            color='#22223a', hovercolor='#33334a')
        for tb in (self.tb_w, self.tb_h, self.tb_g):
            tb.text_disp.set_color('white')

        ax_apply = self.fig.add_axes([0.01, 0.648, 0.185, 0.042])
        self.btn_apply = self._btn(ax_apply, 'Apply Room', '#1a4a8a')
        self.btn_apply.on_clicked(self._on_apply)

        self._hline(0.630)
        lbl(0.615, "OBSTACLES", bold=True, color='#8888bb', size=8)
        lbl(0.585, "Drag → Cat. A (furniture)\nShift+Drag → Cat. B\n(unreachable areas)",
            size=8, color='#aaaaaa')

        # Category indicator label
        self.lbl_cat = self.ax_panel.text(
            0.08, 0.540,
            "Mode: Category A",
            transform=self.ax_panel.transAxes,
            color='#ff9999', fontsize=8, va='top', fontweight='bold')

        ax_undo  = self.fig.add_axes([0.01, 0.475, 0.185, 0.038])
        ax_clear = self.fig.add_axes([0.01, 0.427, 0.185, 0.038])
        self.btn_undo  = self._btn(ax_undo,  'Undo Last', '#5a2222')
        self.btn_clear = self._btn(ax_clear, 'Clear All', '#5a2222')
        self.btn_undo.on_clicked(self._on_undo)
        self.btn_clear.on_clicked(self._on_clear)

        self._hline(0.410)
        ax_run = self.fig.add_axes([0.01, 0.345, 0.185, 0.052])
        self.btn_run = self._btn(ax_run, 'Run Algorithm', '#1a6a3a', size=10)
        self.btn_run.on_clicked(self._on_run)

        ax_exp = self.fig.add_axes([0.01, 0.285, 0.185, 0.044])
        self.btn_exp = self._btn(ax_exp, 'Export JSON', '#2a4a2a')
        self.btn_exp.on_clicked(self._on_export)

        self._hline(0.268)
        lbl(0.253, "STATUS", bold=True, color='#8888bb', size=8)
        self.lbl_status = self.ax_panel.text(
            0.08, 0.225,
            "Enter room dimensions\nand click Apply Room.",
            transform=self.ax_panel.transAxes,
            color='#88ff88', fontsize=8, va='top')

        self._hline(0.120)
        lbl(0.108, "SENSORS", bold=True, color='#8888bb', size=8)
        self.lbl_sensors = self.ax_panel.text(
            0.05, 0.088, "",
            transform=self.ax_panel.transAxes,
            color='#aaaaff', fontsize=7, va='top', family='monospace')

        self.fig.canvas.mpl_connect('button_press_event',   self._on_press)
        self.fig.canvas.mpl_connect('motion_notify_event',  self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('key_press_event',      self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event',    self._on_key_release)

    def _btn(self, ax, label, color, size=9):
        b = Button(ax, label, color=color, hovercolor=self._lighten(color))
        b.label.set_color('white')
        b.label.set_fontsize(size)
        return b

    @staticmethod
    def _lighten(hex_color, factor=1.4):
        import matplotlib.colors as mc
        return tuple(min(1.0, c * factor) for c in mc.to_rgb(hex_color))

    def _hline(self, y):
        self.ax_panel.axhline(y, color='#333355', linewidth=0.8)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _setup_axes(self):
        pad = max(self.room_w, self.room_h) * 0.06
        self.ax.set_xlim(-pad, self.room_w + pad)
        self.ax.set_ylim(-pad, self.room_h + pad)
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors='#666688')
        self.ax.set_xlabel("Meters", color='#666688', fontsize=9)
        self.ax.set_ylabel("Meters", color='#666688', fontsize=9)
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#2a2a4a')
        self.ax.set_title(
            "Room Editor  –  drag: Cat. A obstacle  |  Shift+drag: Cat. B obstacle",
            color='#aaaacc', fontsize=10, pad=6)

    def _draw_room(self):
        self.ax.cla()
        self._setup_axes()
        cs = self.cell_size

        for x in np.arange(0, self.room_w + cs * 0.5, cs):
            self.ax.axvline(x, color=GRID_COLOR, lw=0.25, zorder=0)
        for y in np.arange(0, self.room_h + cs * 0.5, cs):
            self.ax.axhline(y, color=GRID_COLOR, lw=0.25, zorder=0)

        self.ax.add_patch(mpatches.Rectangle(
            (0, 0), self.room_w, self.room_h,
            lw=2, edgecolor=WALL_COLOR, facecolor=ROOM_FACE, zorder=1))

        # Draw Category-A obstacles
        for (ox, oy, ow, oh) in self.obstacles:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh,
                lw=1.5, edgecolor=OBS_A_EDGE, facecolor=OBS_A_FACE, zorder=2))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow/2, oy + oh/2, 'Cat. A',
                             ha='center', va='center',
                             fontsize=7, color='#ccccdd', zorder=3)

        # Draw Category-B obstacles (distinct blue tint + hatching)
        for (ox, oy, ow, oh) in self.obstacles_b:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh,
                lw=1.5, edgecolor=OBS_B_EDGE, facecolor=OBS_B_FACE,
                hatch='///', zorder=2))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow/2, oy + oh/2, 'Cat. B',
                             ha='center', va='center',
                             fontsize=7, color='#aaddff', zorder=3)

        # Legend
        legend_patches = [
            mpatches.Patch(facecolor=OBS_A_FACE, edgecolor=OBS_A_EDGE,
                           label='Cat. A – signal-blocking (furniture)'),
            mpatches.Patch(facecolor=OBS_B_FACE, edgecolor=OBS_B_EDGE,
                           hatch='///', label='Cat. B – unreachable (built-in)'),
        ]
        self.ax.legend(handles=legend_patches, loc='upper right',
                       fontsize=7, facecolor='#22223a',
                       labelcolor='white', edgecolor='#555577')

        if self.sensors:
            self._draw_coverage()
            self._draw_sensors()

        self.fig.canvas.draw_idle()

    def _draw_coverage(self):
        from matplotlib.patches import Polygon as MPoly
        cs   = self.cell_size
        room = self._make_room()
        cols, rows = room.grid_shape()
        sensor_idx = {p: i for i, p in enumerate(self.sensors)}

        def tri_verts(d, x0, y0):
            m = x0 + cs/2, y0 + cs/2
            if d == TOP:    return [(x0, y0+cs), (x0+cs, y0+cs), m]
            if d == BOTTOM: return [(x0, y0),    (x0+cs, y0),    m]
            if d == LEFT:   return [(x0, y0),    (x0, y0+cs),    m]
            if d == RIGHT:  return [(x0+cs, y0), (x0+cs, y0+cs), m]

        for ci in range(cols):
            for cj in range(rows):
                if room.cell_is_obstacle(ci, cj):    # Cat-A: solid, skip
                    continue
                if room.cell_is_obstacle_b(ci, cj):  # Cat-B: drawn separately, skip
                    continue
                x0, y0 = ci * cs, cj * cs

                # Margin cells: subtle dark overlay
                if cell_in_margin(room, ci, cj):
                    self.ax.add_patch(mpatches.Rectangle(
                        (x0, y0), cs, cs,
                        facecolor='#333344', edgecolor='none',
                        alpha=0.5, zorder=2))
                    continue

                for d in DIRS:
                    tri = (ci, cj, d)
                    if self.covered.get(tri) is True:
                        color = '#334433'
                        for spos in self.sensors:
                            if tri in self.sens_cov.get(spos, set()):
                                color = SENSOR_COLORS[sensor_idx[spos] % len(SENSOR_COLORS)]
                                break
                        self.ax.add_patch(MPoly(
                            tri_verts(d, x0, y0), closed=True,
                            facecolor=color, edgecolor='none',
                            alpha=0.45, zorder=2))
                    elif tri in self.covered:
                        # Required but not covered: red
                        self.ax.add_patch(MPoly(
                            tri_verts(d, x0, y0), closed=True,
                            facecolor='#660000', edgecolor='none',
                            alpha=0.4, zorder=2))

    def _draw_sensors(self):
        cs = self.cell_size
        for i, (si, sj) in enumerate(self.sensors):
            sx = (si + 0.5) * cs
            sy = (sj + 0.5) * cs
            col = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            self.ax.add_patch(plt.Circle(
                (sx, sy), SENSOR_RANGE,
                color=col, fill=False, alpha=0.10,
                ls='--', lw=1, zorder=3))
            self.ax.plot(sx, sy, 'o', color=col,
                         ms=13, zorder=5, mec='white', mew=1.2)
            self.ax.text(sx, sy + cs * 1.6, f'S{i+1}',
                         ha='center', fontsize=8,
                         fontweight='bold', color=col, zorder=6)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_room(self):
        r = Room(self.room_w, self.room_h, self.cell_size)
        r.obstacles   = list(self.obstacles)
        r.obstacles_b = list(self.obstacles_b)
        return r

    def _status(self, msg, color='#88ff88'):
        self.lbl_status.set_text(msg)
        self.lbl_status.set_color(color)
        self.fig.canvas.draw_idle()

    def _update_sensor_list(self):
        if not self.sensors:
            self.lbl_sensors.set_text('')
            return
        cs = self.cell_size
        lines = ['S   grid        x       y']
        lines.append('-' * 26)
        for i, (si, sj) in enumerate(self.sensors):
            lines.append(
                f"S{i+1}  ({si:>3},{sj:>3})  "
                f"{(si+0.5)*cs:5.2f}m  {(sj+0.5)*cs:5.2f}m")
        self.lbl_sensors.set_text('\n'.join(lines))
        self.fig.canvas.draw_idle()

    def _reset_results(self):
        self.sensors  = []
        self.covered  = {}
        self.sens_cov = {}

    def _update_cat_label(self):
        if self._placing_cat_b:
            self.lbl_cat.set_text("Mode: Category B")
            self.lbl_cat.set_color('#88ccff')
        else:
            self.lbl_cat.set_text("Mode: Category A")
            self.lbl_cat.set_color('#ff9999')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _on_apply(self, _):
        try:
            self.room_w    = float(self.tb_w.text)
            self.room_h    = float(self.tb_h.text)
            self.cell_size = float(self.tb_g.text)
        except ValueError:
            self._status("Invalid value!", '#ff6666')
            return
        self.obstacles   = []
        self.obstacles_b = []
        self._reset_results()
        self._draw_room()
        self._status(
            f"Room {self.room_w} x {self.room_h} m\n"
            f"Grid {self.cell_size*100:.0f} cm\n\n"
            f"Draw obstacles,\nthen Run Algorithm.\n\n"
            f"Shift+drag for\nCat. B obstacles.")
        self._update_sensor_list()

    def _on_undo(self, _):
        # Remove last obstacle from whichever list is non-empty (B first)
        if self.obstacles_b:
            self.obstacles_b.pop()
        elif self.obstacles:
            self.obstacles.pop()
        else:
            return
        self._reset_results()
        self._draw_room()
        total = len(self.obstacles) + len(self.obstacles_b)
        self._status(f"{total} obstacle(s) remain.")
        self._update_sensor_list()

    def _on_clear(self, _):
        self.obstacles   = []
        self.obstacles_b = []
        self._reset_results()
        self._draw_room()
        self._status("Obstacles cleared.")
        self._update_sensor_list()

    def _on_run(self, _):
        self._status("Running algorithm...", '#ffff88')
        self.fig.canvas.flush_events()
        plt.pause(0.05)
        try:
            room = self._make_room()
            self.sensors, self.covered, self.sens_cov = run_algorithm(room)
        except Exception as e:
            import traceback; traceback.print_exc()
            self._status(f"Error:\n{e}", '#ff6666')
            return

        total     = len(self.covered)
        covered_n = sum(self.covered.values())
        pct       = 100 * covered_n / total if total else 0

        self._draw_room()
        self._status(
            f"{len(self.sensors)} sensor(s)\n"
            f"Triangles: {covered_n}/{total}\n"
            f"Coverage: {pct:.1f}%\n\n"
            f"Export JSON for\nlive viewer.")
        self._update_sensor_list()

    def _on_export(self, _):
        if not self.sensors:
            self._status("Run algorithm first!", '#ff6666')
            return
        cs = self.cell_size
        data = {
            "room": {
                "width":       self.room_w,
                "height":      self.room_h,
                "cell_size":   cs,
                "obstacles_a": list(self.obstacles),
                "obstacles_b": list(self.obstacles_b)
            },
            "sensors": [
                {"id":     i + 1,
                 "grid":   [si, sj],
                 "meters": [round((si + 0.5) * cs, 3),
                            round((sj + 0.5) * cs, 3)],
                 "port":   ""}
                for i, (si, sj) in enumerate(self.sensors)
            ]
        }
        with open("sensor_positions.json", "w") as f:
            json.dump(data, f, indent=2)
        self._status("Exported to\nsensor_positions.json")
        print(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Mouse / keyboard events
    # ------------------------------------------------------------------

    def _on_key_press(self, ev):
        if ev.key == 'shift':
            self._placing_cat_b = True
            self._update_cat_label()

    def _on_key_release(self, ev):
        if ev.key == 'shift':
            self._placing_cat_b = False
            self._update_cat_label()

    def _clamp(self, x, y):
        return np.clip(x, 0, self.room_w), np.clip(y, 0, self.room_h)

    def _on_press(self, ev):
        if ev.inaxes is not self.ax or ev.button != 1:
            return
        x, y = ev.xdata, ev.ydata
        if x is None or y is None:
            return
        if not (0 <= x <= self.room_w and 0 <= y <= self.room_h):
            return
        self._drag_start = (x, y)
        edge_color = '#55aaff' if self._placing_cat_b else '#ffff44'
        self._drag_rect = mpatches.Rectangle(
            (x, y), 0, 0, lw=1.5,
            edgecolor=edge_color, facecolor=edge_color + '15', zorder=10)
        self.ax.add_patch(self._drag_rect)

    def _on_motion(self, ev):
        if self._drag_start is None or ev.inaxes is not self.ax:
            return
        x, y = ev.xdata, ev.ydata
        if x is None or y is None:
            return
        x, y   = self._clamp(x, y)
        x0, y0 = self._drag_start
        self._drag_rect.set_xy((min(x, x0), min(y, y0)))
        self._drag_rect.set_width(abs(x - x0))
        self._drag_rect.set_height(abs(y - y0))
        self.fig.canvas.draw_idle()

    def _on_release(self, ev):
        if self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect:
            self._drag_rect.remove()
            self._drag_rect = None

        if ev.inaxes is not self.ax:
            self._draw_room(); return
        x1, y1 = ev.xdata, ev.ydata
        if x1 is None or y1 is None:
            self._draw_room(); return
        x1, y1 = self._clamp(x1, y1)
        ow, oh = abs(x1 - x0), abs(y1 - y0)
        if ow < self.cell_size or oh < self.cell_size:
            self._draw_room(); return

        obs = (min(x0, x1), min(y0, y1), ow, oh)
        if self._placing_cat_b:
            self.obstacles_b.append(obs)
            cat_label = "Cat. B"
        else:
            self.obstacles.append(obs)
            cat_label = "Cat. A"

        self._reset_results()
        self._draw_room()
        total = len(self.obstacles) + len(self.obstacles_b)
        self._status(
            f"{total} obstacle(s)\n"
            f"  A: {len(self.obstacles)}\n"
            f"  B: {len(self.obstacles_b)}\n\n"
            f"Last added: {cat_label}\n\n"
            f"Add more or click\n'Run Algorithm'.")
        self._update_sensor_list()


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == '__main__':
    PlacementGUI()