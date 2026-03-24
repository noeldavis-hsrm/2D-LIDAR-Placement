"""
LiDAR Validation Viewer
=======================
Viewer 2: Misst die *reale* Coverage der platzierten Sensoren.

Für jede Grid-Zelle wird akkumuliert, von wie vielen verschiedenen Sensoren
sie im Laufe der Messung mindestens einmal beobachtet wurde.

Ergebnis nach Stop:
  - Heatmap: 0 Sensoren (rot) / 1 Sensor (orange) / ≥2 Sensoren (grün)
  - Statistik-Zusammenfassung in der Konsole und als CSV-Export

Vergleich mit Viewer 1 (theoretische Coverage) zeigt, wie gut das
geometrische Modell die Realität abbildet.

Voraussetzungen:
    pip install rplidar-roboticia matplotlib numpy pyserial
"""

import sys, json, threading, time, math, csv, datetime
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MPoly
from matplotlib.widgets import Button as MplButton

# ── Farben ────────────────────────────────────────────────────────────────────
BG            = '#1e1e2e'
ROOM_FACE     = '#2a2a3e'
OBS_A_FACE    = '#555577'
OBS_A_EDGE    = '#8888aa'
OBS_B_FACE    = '#335577'
OBS_B_EDGE    = '#55aacc'
WALL_COLOR    = '#aaaacc'
SENSOR_COLORS = ['#ff4444', '#44ff88', '#4488ff', '#ffaa00', '#cc44ff', '#00ffff']
SENSOR_RANGE  = 6.0
REDRAW_INTERVAL = 200   # ms – etwas langsamer als Viewer 1, da Heatmap teurer


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def cell_is_obstacle_a(ci, cj, obstacles_a, cs):
    cx = (ci + 0.5) * cs
    cy = (cj + 0.5) * cs
    return any(ox <= cx <= ox + ow and oy <= cy <= oy + oh
               for ox, oy, ow, oh in obstacles_a)


def cell_is_obstacle_b(ci, cj, obstacles_b, cs):
    cx = (ci + 0.5) * cs
    cy = (cj + 0.5) * cs
    return any(ox <= cx <= ox + ow and oy <= cy <= oy + oh
               for ox, oy, ow, oh in obstacles_b)


def point_to_cell(x, y, cs):
    """Convert metric coordinates to grid cell indices."""
    return int(x / cs), int(y / cs)


def cell_center(ci, cj, cs):
    return (ci + 0.5) * cs, (cj + 0.5) * cs


def has_los_to_point(sx, sy, px, py, cols, rows, obstacles_a, cs):
    """
    Check line-of-sight from sensor (sx,sy) in grid coords
    to point (px,py) in grid coords.
    Only Category-A blocks rays.
    """
    steps = int(max(abs(px - sx), abs(py - sy)) * 4) + 2
    for t in np.linspace(0, 1, steps):
        ri = int(sx + t * (px - sx))
        rj = int(sy + t * (py - sy))
        if 0 <= ri < cols and 0 <= rj < rows:
            if cell_is_obstacle_a(ri, rj, obstacles_a, cs):
                return False
    return True


# =============================================================================
# OBSERVATION ACCUMULATOR
# =============================================================================

TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
DIRS = [TOP, RIGHT, BOTTOM, LEFT]


def get_covered_directions(sx, sy, cx, cy):
    """Which triangle faces of cell (cx,cy) does sensor at (sx,sy) cover?"""
    dirs = []
    if sy > cy: dirs.append(TOP)
    if sx > cx: dirs.append(RIGHT)
    if sy < cy: dirs.append(BOTTOM)
    if sx < cx: dirs.append(LEFT)
    return dirs


class ObservationMap:
    """
    Thread-safe accumulator – tracked at triangle level.

    triangles[(ci, cj, d)] = first sensor index that covered this triangle.
    cell_sensors[ci][cj]   = set of sensor indices that have seen this cell
                             (used for the ≥2-sensor statistics).
    """

    def __init__(self, cols, rows):
        self.cols         = cols
        self.rows         = rows
        self.lock         = threading.Lock()
        self.triangles    = {}                          # (ci,cj,d) → sensor_idx
        self.cell_sensors = [[set() for _ in range(rows)]
                             for _ in range(cols)]

    def record(self, ci, cj, direction, sensor_idx):
        """Mark triangle (ci, cj, direction) as seen by sensor_idx."""
        if not (0 <= ci < self.cols and 0 <= cj < self.rows):
            return
        with self.lock:
            tri = (ci, cj, direction)
            if tri not in self.triangles:
                self.triangles[tri] = sensor_idx
            self.cell_sensors[ci][cj].add(sensor_idx)

    def snapshot_triangles(self):
        """Return a copy of the triangle dict."""
        with self.lock:
            return dict(self.triangles)

    def snapshot_cell_counts(self):
        """Return (cols x rows) array of how many sensors saw each cell."""
        with self.lock:
            return np.array([[len(self.cell_sensors[ci][cj])
                              for cj in range(self.rows)]
                             for ci in range(self.cols)], dtype=np.int32)


# =============================================================================
# SENSOR THREAD
# =============================================================================

class SensorThread(threading.Thread):

    def __init__(self, sensor_info, room, obs_map, sensor_idx):
        super().__init__(daemon=True)
        self.info       = sensor_info
        self.room       = room
        self.obs_map    = obs_map
        self.sensor_idx = sensor_idx
        self.port       = sensor_info.get('port', '').strip()
        self.sx_m       = sensor_info['meters'][0]   # metres
        self.sy_m       = sensor_info['meters'][1]
        self.points     = []          # latest scan points (metres) for display
        self.lock       = threading.Lock()
        self.running    = True
        self.status     = 'init'
        self._lidar     = None
        self.empty      = (self.port == '<empty>')
        if self.empty:
            self.port = ''

        cs = room['cell_size']
        self.sx_g = self.sx_m / cs   # grid coords (float)
        self.sy_g = self.sy_m / cs

    def run(self):
        if self.empty:
            self.status = 'no sensor'
            return
        if self.port:
            self._run_real()
        else:
            self.status = 'simulated'
            self._run_simulated()

    # ------------------------------------------------------------------
    def _process_scan(self, pts_metres):
        """
        Walk each ray from sensor to hit point.
        For every free cell along the ray, record the triangle directions
        that the sensor covers from its position.
        """
        cs    = self.room['cell_size']
        obs_a = self.room.get('obstacles_a', self.room.get('obstacles', []))
        obs_b = self.room.get('obstacles_b', [])
        cols  = int(round(self.room['width']  / cs))
        rows  = int(round(self.room['height'] / cs))
        rc_sq = (SENSOR_RANGE / cs) ** 2

        for px_m, py_m in pts_metres:
            px_g = px_m / cs
            py_g = py_m / cs

            if (px_g - self.sx_g) ** 2 + (py_g - self.sy_g) ** 2 > rc_sq:
                continue

            steps = int(max(abs(px_g - self.sx_g),
                            abs(py_g - self.sy_g)) * 4) + 2
            for t in np.linspace(0, 1, steps):
                ci = int(self.sx_g + t * (px_g - self.sx_g))
                cj = int(self.sy_g + t * (py_g - self.sy_g))
                if not (0 <= ci < cols and 0 <= cj < rows):
                    continue
                if cell_is_obstacle_a(ci, cj, obs_a, cs):
                    break              # ray blocked by Cat-A
                if cell_is_obstacle_b(ci, cj, obs_b, cs):
                    continue           # transparent Cat-B, keep going
                cx, cy = ci + 0.5, cj + 0.5
                for d in get_covered_directions(self.sx_g, self.sy_g, cx, cy):
                    self.obs_map.record(ci, cj, d, self.sensor_idx)

    # ------------------------------------------------------------------
    def _run_real(self):
        try:
            from rplidar import RPLidar
        except ImportError:
            self.status = 'no_lib->sim'
            self._run_simulated()
            return

        MAX_RETRIES = 5
        retry = 0

        while self.running and retry < MAX_RETRIES:
            lidar = None
            try:
                lidar = RPLidar(self.port, baudrate=115200, timeout=3)
                self._lidar = lidar
                lidar.stop(); lidar.stop_motor(); time.sleep(0.3)
                for attr in ('_serial', 'serial'):
                    s = getattr(lidar, attr, None)
                    if s:
                        try: s.reset_input_buffer()
                        except Exception: pass
                        break
                lidar.start_motor(); time.sleep(1.0)
                self.status = 'connected'
                retry = 0

                for scan in lidar.iter_scans(min_len=3):
                    if not self.running:
                        break
                    pts = []
                    for _, angle_deg, dist_mm in scan:
                        if dist_mm < 100:
                            continue
                        r = dist_mm / 1000.0
                        a = math.radians(angle_deg)
                        pts.append((self.sx_m + r * math.sin(a),
                                    self.sy_m + r * math.cos(a)))
                    with self.lock:
                        self.points = pts
                    self._process_scan(pts)

            except Exception as e:
                retry += 1
                self.status = f'retry {retry}/{MAX_RETRIES}'
                print(f"S{self.info['id']} error (retry {retry}): {e}")
                time.sleep(1.0)
            finally:
                self._lidar = None
                if lidar is not None:
                    try:
                        lidar.stop(); lidar.stop_motor()
                        time.sleep(0.1); lidar.disconnect()
                    except Exception: pass

        if retry >= MAX_RETRIES:
            self.status = 'failed->sim'
            self._run_simulated()

    # ------------------------------------------------------------------
    def _run_simulated(self):
        w     = self.room['width']
        h     = self.room['height']
        obs_a = self.room.get('obstacles_a', self.room.get('obstacles', []))
        angle = 0.0
        while self.running:
            pts = []
            for i in range(360):
                a = math.radians(angle + i)
                for d in np.arange(0.05, SENSOR_RANGE, 0.02):
                    rx = self.sx_m + d * math.sin(a)
                    ry = self.sy_m + d * math.cos(a)
                    hit = False
                    if not (0 <= rx <= w and 0 <= ry <= h):
                        pts.append((rx, ry)); hit = True
                    elif any(ox <= rx <= ox + ow and oy <= ry <= oy + oh
                             for ox, oy, ow, oh in obs_a):
                        pts.append((rx, ry)); hit = True
                    if hit:
                        break
            with self.lock:
                self.points = pts
            self._process_scan(pts)
            angle = (angle + 5) % 360
            time.sleep(0.05)

    def get_points(self):
        with self.lock: return list(self.points)

    def stop(self):
        self.running = False


# =============================================================================
# SETUP DIALOG
# =============================================================================

def list_serial_ports():
    BLACKLIST = {'/dev/cu.debug-console', '/dev/cu.Bluetooth-Incoming-Port'}
    ports = ['(simulated)', '<empty>']
    try:
        import serial.tools.list_ports
        ports += [p.device for p in serial.tools.list_ports.comports()
                  if p.device not in BLACKLIST]
    except ImportError:
        pass
    return ports


class SetupDialog:

    def __init__(self):
        self.result     = None
        self._data      = None
        self._port_vars = []

        self.root = tk.Tk()
        self.root.title('LiDAR Validator – Setup')
        self.root.resizable(False, False)
        self.root.configure(bg='#1e1e2e')
        self._apply_style()
        self._build_ui()
        self.root.mainloop()

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.', background='#1e1e2e', foreground='#ccccdd',
                    font=('Helvetica', 10))
        s.configure('TLabel',      background='#1e1e2e', foreground='#ccccdd')
        s.configure('TFrame',      background='#1e1e2e')
        s.configure('TLabelframe', background='#1e1e2e', foreground='#8888bb')
        s.configure('TLabelframe.Label', background='#1e1e2e',
                    foreground='#8888bb', font=('Helvetica', 9, 'bold'))
        s.configure('TButton', background='#2a4a8a', foreground='white',
                    borderwidth=0, padding=6)
        s.map('TButton', background=[('active', '#3a5a9a')])
        s.configure('TCombobox', fieldbackground='#22223a', foreground='white',
                    background='#22223a', selectbackground='#22223a')
        s.configure('Start.TButton', background='#1a6a3a', foreground='white',
                    font=('Helvetica', 11, 'bold'), padding=8)
        s.map('Start.TButton', background=[('active', '#2a8a4a')])

    def _build_ui(self):
        r = self.root
        pad = dict(padx=14, pady=6)

        tk.Label(r, text='LiDAR Validation Viewer',
                 font=('Helvetica', 15, 'bold'),
                 bg='#1e1e2e', fg='white').pack(pady=(16, 2))
        tk.Label(r, text='Misst reale Coverage – Viewer 2',
                 bg='#1e1e2e', fg='#666688',
                 font=('Helvetica', 9)).pack(pady=(0, 6))

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        ff = ttk.Frame(r); ff.pack(fill='x', **pad)
        ttk.Label(ff, text='JSON-Datei:').pack(side='left')
        self._file_var = tk.StringVar(value='sensor_positions.json')
        ttk.Entry(ff, textvariable=self._file_var, width=34).pack(
            side='left', padx=(8, 4))
        ttk.Button(ff, text='Browse…', command=self._browse).pack(side='left')
        ttk.Button(r, text='JSON laden', command=self._load).pack(pady=(2, 6))

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        self._sf = ttk.LabelFrame(r, text='Sensor-Ports')
        self._sf.pack(fill='x', padx=14, pady=8)
        ttk.Label(self._sf, text='Lade zuerst eine JSON-Datei.',
                  foreground='#666688').pack(pady=10)

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        self._info_var = tk.StringVar(value='')
        ttk.Label(r, textvariable=self._info_var,
                  foreground='#88aaff').pack(**pad)

        bf = ttk.Frame(r); bf.pack(pady=(0, 16))
        ttk.Button(bf, text='Abbrechen', command=r.destroy).pack(
            side='left', padx=8)
        self._start_btn = ttk.Button(
            bf, text='▶  Messung starten',
            style='Start.TButton', command=self._start, state='disabled')
        self._start_btn.pack(side='left', padx=8)

    def _browse(self):
        p = filedialog.askopenfilename(
            title='sensor_positions.json auswählen',
            filetypes=[('JSON', '*.json'), ('Alle Dateien', '*.*')])
        if p: self._file_var.set(p)

    def _load(self):
        path = self._file_var.get().strip()
        try:
            with open(path) as f:
                self._data = json.load(f)
        except FileNotFoundError:
            messagebox.showerror('Fehler', f'Datei nicht gefunden:\n{path}')
            return
        except json.JSONDecodeError as e:
            messagebox.showerror('Fehler', f'Ungültiges JSON:\n{e}')
            return

        room    = self._data['room']
        sensors = self._data['sensors']
        obs_a   = room.get('obstacles_a', room.get('obstacles', []))
        obs_b   = room.get('obstacles_b', [])
        self._info_var.set(
            f"Raum {room['width']} × {room['height']} m  |  "
            f"{len(sensors)} Sensor(en)  |  "
            f"Zellgröße {room['cell_size']*100:.0f} cm  |  "
            f"Hind. A: {len(obs_a)}  B: {len(obs_b)}")

        for w in self._sf.winfo_children(): w.destroy()
        avail = list_serial_ports()
        self._port_vars = []

        for col, (txt, width) in enumerate(
                [('Sensor', 8), ('Position (m)', 18), ('COM-Port', 24)]):
            tk.Label(self._sf, text=txt, bg='#13131f', fg='#8888bb',
                     font=('Helvetica', 9, 'bold'),
                     width=width, anchor='w').grid(
                row=0, column=col, padx=(10, 4), pady=(6, 2), sticky='w')

        for i, s in enumerate(sensors, start=1):
            color = SENSOR_COLORS[(s['id'] - 1) % len(SENSOR_COLORS)]
            tk.Label(self._sf, text=f"  S{s['id']}",
                     bg='#1e1e2e', fg=color,
                     font=('Helvetica', 10, 'bold')).grid(
                row=i, column=0, padx=10, pady=4, sticky='w')
            tk.Label(self._sf,
                     text=f"({s['meters'][0]:.2f},  {s['meters'][1]:.2f})",
                     bg='#1e1e2e', fg='#aaaacc').grid(
                row=i, column=1, padx=4, pady=4, sticky='w')
            var = tk.StringVar(
                value=s.get('port', '').strip() or '(simulated)')
            ttk.Combobox(self._sf, textvariable=var,
                         values=avail, width=22, state='normal').grid(
                row=i, column=2, padx=(4, 4), pady=4, sticky='w')
            self._port_vars.append(var)

            # ── Info-Button (Seriennummer) ─────────────────────────────────
            btn = tk.Button(self._sf, text='ℹ',
                            bg='#2a2a4a', fg='#88aaff',
                            activebackground='#3a3a6a',
                            font=('Helvetica', 12),
                            relief='flat', bd=0, padx=4,
                            command=lambda v=var, sid=s['id']: self._show_serial(v, sid))
            btn.grid(row=i, column=3, padx=(0, 8), pady=4)

        self._start_btn.config(state='normal')

    def _show_serial(self, port_var, sensor_id):
        """Liest die Seriennummer des Sensors und zeigt sie in einem Popup."""
        port = port_var.get().strip()
        if port in ('(simulated)', '<empty>', ''):
            messagebox.showinfo(
                f'S{sensor_id} – Seriennummer',
                'Kein physischer Port ausgewählt.\nSeriennummer nicht verfügbar.')
            return

        def _fetch():
            try:
                from rplidar import RPLidar
                lidar = RPLidar(port=port, baudrate=115200, timeout=3)
                info  = lidar.get_info()
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                sn       = str(info.get('serialnumber', 'N/A'))
                sn_short = '...' + sn[-8:]
                self.root.after(0, lambda: messagebox.showinfo(
                    f'S{sensor_id} – Seriennummer', sn_short))
            except ImportError:
                self.root.after(0, lambda: messagebox.showerror(
                    'Fehler', 'rplidar-Bibliothek nicht installiert.'))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    f'S{sensor_id} – Fehler', str(e)))

        threading.Thread(target=_fetch, daemon=True).start()

    def _start(self):
        if self._data is None: return
        for i, var in enumerate(self._port_vars):
            val = var.get().strip()
            self._data['sensors'][i]['port'] = \
                '' if val == '(simulated)' else val
        self.result = self._data
        self.root.destroy()


# =============================================================================
# VALIDATION VIEWER
# =============================================================================

class ValidationViewer:

    def __init__(self, data):
        self.room    = data['room']
        self.sensors = data['sensors']

        # Normalize obstacle keys
        if 'obstacles_a' not in self.room and 'obstacles' in self.room:
            self.room['obstacles_a'] = self.room['obstacles']
        if 'obstacles_b' not in self.room:
            self.room['obstacles_b'] = []

        self.cs   = self.room['cell_size']
        self.cols = int(round(self.room['width']  / self.cs))
        self.rows = int(round(self.room['height'] / self.cs))
        self.obs_a = self.room.get('obstacles_a', [])
        self.obs_b = self.room.get('obstacles_b', [])

        # Build set of cells that count for validation
        # (begehbar, kein Cat-A, kein Cat-B, kein Margin)
        self.required_triangles = self._build_required_triangles()

        self.obs_map  = ObservationMap(self.cols, self.rows)
        self.running  = False
        self.stopped  = False
        self.start_ts = None

        self.threads = [
            SensorThread(s, self.room, self.obs_map, i)
            for i, s in enumerate(self.sensors)]

        self._build_figure()
        self._draw_static()
        self._start_measurement()
        plt.show()

    # ------------------------------------------------------------------
    def _build_required_triangles(self):
        """
        All (ci, cj, d) triangles that must be covered for validation.
        Mirrors the logic of build_required_triangles() in lidar_tool.py:
          - Skip Cat-A cells (solid obstacle)
          - Skip Cat-B cells (unreachable)
          - Skip margin cells (1 cell around walls and Cat-A obstacles)
        Returns a set of (ci, cj, d) tuples.
        """
        m = 1   # MARGIN_CELLS
        required = set()
        for ci in range(self.cols):
            for cj in range(self.rows):
                if cell_is_obstacle_a(ci, cj, self.obs_a, self.cs):
                    continue
                if cell_is_obstacle_b(ci, cj, self.obs_b, self.cs):
                    continue
                if ci < m or ci >= self.cols - m:
                    continue
                if cj < m or cj >= self.rows - m:
                    continue
                in_margin = False
                for ox, oy, ow, oh in self.obs_a:
                    ci0 = int(ox        / self.cs)
                    ci1 = int((ox + ow) / self.cs)
                    cj0 = int(oy        / self.cs)
                    cj1 = int((oy + oh) / self.cs)
                    if (ci0 - m) <= ci <= (ci1 + m) and \
                       (cj0 - m) <= cj <= (cj1 + m):
                        in_margin = True
                        break
                if not in_margin:
                    for d in DIRS:
                        required.add((ci, cj, d))
        return required

    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(15, 8))
        self.fig.subplots_adjust(right=0.78)
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(ROOM_FACE)
        self.fig.canvas.manager.set_window_title('LiDAR Validation Viewer')
        self.ax.set_aspect('equal')

        w = self.room['width']; h = self.room['height']
        pad = max(w, h) * 0.05
        self.ax.set_xlim(-pad, w + pad)
        self.ax.set_ylim(-pad, h + pad)
        self.ax.tick_params(colors='#666688')
        self.ax.set_xlabel('Meter', color='#666688')
        self.ax.set_ylabel('Meter', color='#666688')
        for sp in self.ax.spines.values():
            sp.set_edgecolor('#2a2a4a')

        # Buttons
        ax_stop = self.fig.add_axes([0.01, 0.01, 0.10, 0.045])
        ax_csv  = self.fig.add_axes([0.13, 0.01, 0.10, 0.045])

        self.btn_stop = MplButton(ax_stop, 'Stop',
                                  color='#5a2222', hovercolor='#8a3333')
        self.btn_stop.label.set_color('white')
        self.btn_stop.on_clicked(self._on_stop)

        self.btn_csv = MplButton(ax_csv, 'CSV Export',
                                 color='#1a4a2a', hovercolor='#2a6a3a')
        self.btn_csv.label.set_color('white')
        self.btn_csv.on_clicked(self._on_export)

        # Status text (top left of plot)
        self.txt_status = self.ax.text(
            0.01, 0.99, 'Messung läuft…',
            transform=self.ax.transAxes,
            color='#88ff88', fontsize=9, va='top', ha='left',
            bbox=dict(facecolor='#13131f', alpha=0.7, edgecolor='none',
                      boxstyle='round,pad=0.3'))

        # Stats text (top right of plot)
        self.txt_stats = self.ax.text(
            0.99, 0.99, '',
            transform=self.ax.transAxes,
            color='#aaaaff', fontsize=8, va='top', ha='right',
            bbox=dict(facecolor='#13131f', alpha=0.7, edgecolor='none',
                      boxstyle='round,pad=0.3'))

        self.fig.canvas.mpl_connect('close_event', self._on_close)

        # Triangle patch collection – rebuilt each frame
        self._tri_patches = []

        # Scan point collections (one per sensor, updated live)
        self.scan_arts = [None] * len(self.sensors)

    # ------------------------------------------------------------------
    def _draw_static(self):
        w = self.room['width']; h = self.room['height']
        cs = self.cs

        # Grid lines
        for x in np.arange(0, w + cs * 0.5, cs):
            self.ax.axvline(x, color='#3a3a4e', lw=0.25, zorder=1)
        for y in np.arange(0, h + cs * 0.5, cs):
            self.ax.axhline(y, color='#3a3a4e', lw=0.25, zorder=1)

        # Margin overlay: draw once as dark rectangles
        m = 1
        for ci in range(self.cols):
            for cj in range(self.rows):
                if cell_is_obstacle_a(ci, cj, self.obs_a, cs): continue
                if cell_is_obstacle_b(ci, cj, self.obs_b, cs): continue
                is_margin = False
                if ci < m or ci >= self.cols - m or cj < m or cj >= self.rows - m:
                    is_margin = True
                else:
                    for ox, oy, ow, oh in self.obs_a:
                        ci0, ci1 = int(ox/cs), int((ox+ow)/cs)
                        cj0, cj1 = int(oy/cs), int((oy+oh)/cs)
                        if (ci0-m) <= ci <= (ci1+m) and (cj0-m) <= cj <= (cj1+m):
                            is_margin = True; break
                if is_margin:
                    self.ax.add_patch(mpatches.Rectangle(
                        (ci*cs, cj*cs), cs, cs,
                        facecolor='#333344', edgecolor='none',
                        alpha=0.45, zorder=2))

        # Room outline
        self.ax.add_patch(mpatches.Rectangle(
            (0, 0), w, h, lw=2,
            edgecolor=WALL_COLOR, facecolor='none', zorder=3))

        # Cat-A obstacles
        for ox, oy, ow, oh in self.obs_a:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh, lw=1.5,
                edgecolor=OBS_A_EDGE, facecolor=OBS_A_FACE, zorder=4))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow/2, oy + oh/2, 'Cat. A',
                             ha='center', va='center',
                             fontsize=7, color='#ccccdd', zorder=5)

        # Cat-B obstacles
        for ox, oy, ow, oh in self.obs_b:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh, lw=1.5,
                edgecolor=OBS_B_EDGE, facecolor=OBS_B_FACE,
                hatch='///', zorder=4))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow/2, oy + oh/2, 'Cat. B',
                             ha='center', va='center',
                             fontsize=7, color='#aaddff', zorder=5)

        # Sensor markers
        for i, s in enumerate(self.sensors):
            sx, sy = s['meters']
            col = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            self.ax.plot(sx, sy, 'o', color=col,
                         ms=12, zorder=7, mec='white', mew=1.2)
            self.ax.text(sx, sy + cs * 1.8, f'S{s["id"]}',
                         ha='center', fontsize=8,
                         fontweight='bold', color=col, zorder=8)

        # Legend – sensor colors + blind/margin/obstacle
        handles = []
        for i, s in enumerate(self.sensors):
            col = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            handles.append(mpatches.Patch(
                facecolor=col, alpha=0.55,
                label=f'S{s["id"]} abgedeckt'))
        handles += [
            mpatches.Patch(color='#550000', alpha=0.55,
                           label='Blindzone (noch nicht gesehen)'),
            mpatches.Patch(facecolor='#333344', edgecolor='none',
                           alpha=0.65, label='Margin (ausgeschlossen)'),
            mpatches.Patch(facecolor=OBS_A_FACE, edgecolor=OBS_A_EDGE,
                           label='Cat. A – signalblockierend'),
            mpatches.Patch(facecolor=OBS_B_FACE, edgecolor=OBS_B_EDGE,
                           hatch='///', label='Cat. B – nicht erreichbar'),
        ]
        self.ax.legend(handles=handles,
                       loc='upper left',
                       bbox_to_anchor=(1.02, 1.0),
                       borderaxespad=0,
                       facecolor='#22223a', edgecolor='#444466',
                       labelcolor='white', fontsize=8)

        self.ax.set_title('LiDAR Validation Viewer  –  reale Coverage-Messung',
                          color='#aaaacc', fontsize=11, pad=6)

    # ------------------------------------------------------------------
    def _start_measurement(self):
        self.running  = True
        self.start_ts = time.time()
        for t in self.threads: t.start()

        self._timer = self.fig.canvas.new_timer(interval=REDRAW_INTERVAL)
        self._timer.add_callback(self._update)
        self._timer.start()

    # ------------------------------------------------------------------
    def _tri_verts(self, d, x0, y0):
        cs = self.cs
        m  = x0 + cs/2, y0 + cs/2
        if d == TOP:    return [(x0,    y0+cs), (x0+cs, y0+cs), m]
        if d == BOTTOM: return [(x0,    y0),    (x0+cs, y0),    m]
        if d == LEFT:   return [(x0,    y0),    (x0,    y0+cs), m]
        if d == RIGHT:  return [(x0+cs, y0),    (x0+cs, y0+cs), m]

    # ------------------------------------------------------------------
    def _update(self):
        tri_snap = self.obs_map.snapshot_triangles()

        # ── Remove old triangle patches ───────────────────────────────
        for p in self._tri_patches:
            p.remove()
        self._tri_patches.clear()

        # ── Draw triangles ────────────────────────────────────────────
        cs = self.cs
        for ci, cj, d in self.required_triangles:
            x0, y0 = ci * cs, cj * cs
            verts   = self._tri_verts(d, x0, y0)
            tri_key = (ci, cj, d)
            if tri_key in tri_snap:
                s_idx = tri_snap[tri_key]
                color = SENSOR_COLORS[s_idx % len(SENSOR_COLORS)]
                alpha = 0.50
            else:
                color = '#550000'   # not yet seen → dark red
                alpha = 0.45
            patch = MPoly(verts, closed=True,
                          facecolor=color, edgecolor='none',
                          alpha=alpha, zorder=3)
            self.ax.add_patch(patch)
            self._tri_patches.append(patch)

        # ── Update scan point clouds ──────────────────────────────────
        for i, t in enumerate(self.threads):
            if self.scan_arts[i] is not None:
                self.scan_arts[i].remove()
                self.scan_arts[i] = None
            pts = t.get_points()
            if pts:
                col = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                self.scan_arts[i] = self.ax.scatter(
                    [p[0] for p in pts], [p[1] for p in pts],
                    s=2, c=col, alpha=0.6, zorder=7, linewidths=0)

        # ── Stats ─────────────────────────────────────────────────────
        elapsed = time.time() - self.start_ts
        stats   = self._compute_stats(tri_snap)
        self.txt_status.set_text(
            f"{'⏹ Gestoppt' if self.stopped else '⏺ Messung läuft'}  "
            f"|  {elapsed:.0f} s")
        self.txt_stats.set_text(
            f"Dreiecke gesamt:   {stats['total']:5d}\n"
            f"Abgedeckt:         {stats['covered']:5d}  ({stats['pct']:5.1f} %)\n"
            f"Blindzonen:        {stats['blind']:5d}  ({100-stats['pct']:5.1f} %)")

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _compute_stats(self, tri_snap):
        total   = len(self.required_triangles)
        covered = sum(1 for tri in self.required_triangles if tri in tri_snap)
        blind   = total - covered
        pct     = 100 * covered / total if total else 0
        return dict(total=total, covered=covered, blind=blind, pct=pct)

    # ------------------------------------------------------------------
    def _on_stop(self, _):
        if self.stopped:
            return
        self.stopped = True
        self.running = False
        for t in self.threads: t.stop()
        if hasattr(self, '_timer'): self._timer.stop()

        tri_snap = self.obs_map.snapshot_triangles()
        stats    = self._compute_stats(tri_snap)
        elapsed  = time.time() - self.start_ts

        print('\n' + '=' * 52)
        print('  VALIDIERUNGSERGEBNIS')
        print('=' * 52)
        print(f"  Messdauer:         {elapsed:.1f} s")
        print(f"  Dreiecke gesamt:   {stats['total']}")
        print(f"  Abgedeckt:         {stats['covered']:5d}  ({stats['pct']:.1f} %)")
        print(f"  Blindzonen:        {stats['blind']:5d}  ({100-stats['pct']:.1f} %)")
        print('=' * 52)

        # Force final redraw
        self._update()

    def _on_export(self, _):
        tri_snap = self.obs_map.snapshot_triangles()
        stats    = self._compute_stats(tri_snap)
        ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname    = f'validation_{ts}.csv'
        dir_names = {TOP: 'TOP', RIGHT: 'RIGHT', BOTTOM: 'BOTTOM', LEFT: 'LEFT'}

        with open(fname, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['# LiDAR Validation Export', ts])
            w.writerow(['# Raum', f"{self.room['width']}x{self.room['height']} m"])
            w.writerow(['# Zellgröße (m)', self.cs])
            w.writerow(['# Dreiecke gesamt', stats['total']])
            w.writerow(['# Abgedeckt (%)', f"{stats['pct']:.1f}"])
            w.writerow(['# Blindzonen (%)', f"{100-stats['pct']:.1f}"])
            w.writerow([])
            w.writerow(['ci', 'cj', 'direction',
                        'x_m', 'y_m',
                        'covered_by_sensor', 'result'])
            for ci, cj, d in sorted(self.required_triangles):
                tri = (ci, cj, d)
                s_idx = tri_snap.get(tri, -1)
                result = f'S{s_idx+1}' if s_idx >= 0 else 'BLIND'
                w.writerow([ci, cj, dir_names[d],
                            round((ci + 0.5) * self.cs, 3),
                            round((cj + 0.5) * self.cs, 3),
                            s_idx if s_idx >= 0 else '',
                            result])

        print(f"CSV exportiert: {fname}")
        self.txt_status.set_text(f"Exportiert: {fname}")
        self.fig.canvas.draw_idle()

    def _on_close(self, _):
        if not self.stopped:
            self.stopped = True
            for t in self.threads: t.stop()
            if hasattr(self, '_timer'): self._timer.stop()
        print('Viewer beendet.')


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == '__main__':
    dialog = SetupDialog()
    if dialog.result is None:
        print('Abgebrochen.')
        sys.exit(0)
    ValidationViewer(dialog.result)