"""
LiDAR Live Viewer
=================
Startet mit einer Tkinter-GUI zum Laden der JSON-Datei und Zuweisen
der COM-Ports. Dann öffnet sich das Matplotlib-Fenster mit:
  - Berechneter Coverage (Dreiecke, farbig je Sensor)
  - Live-Scandaten aller verbundenen RPLidar-Sensoren

Hindernis-Kategorien:
  Category A: Signalblockierende Hindernisse (z.B. Möbel).
              Strahlen werden blockiert. Zellen dahinter sind potenzielle
              Blindzonen → Coverage erforderlich.
              Im JSON unter "obstacles" oder "obstacles_a".
  Category B: Nicht erreichbare Bereiche (z.B. eingebaute Schränke, Säulen).
              Person kann dort nicht sein → aus Coverage-Metrik ausgeschlossen.
              Strahlen passieren transparent. Kein Sicherheitsabstand.
              Im JSON unter "obstacles_b".

Voraussetzungen:
    pip install rplidar-roboticia matplotlib numpy pyserial
"""

import sys, json, threading, time, math
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
OBS_A_FACE    = '#555577'   # Category A
OBS_A_EDGE    = '#8888aa'
OBS_B_FACE    = '#335577'   # Category B (blau)
OBS_B_EDGE    = '#55aacc'
WALL_COLOR    = '#aaaacc'
SENSOR_COLORS = ['#ff4444', '#44ff88', '#4488ff', '#ffaa00', '#cc44ff', '#00ffff']
SENSOR_RANGE  = 6.0
TOP, RIGHT, BOTTOM, LEFT = 0, 1, 2, 3
REDRAW_INTERVAL = 50   # ms


# =============================================================================
# COVERAGE HELPERS
# =============================================================================

def cell_is_obstacle_a(ci, cj, obstacles_a, cs):
    """Nur Category-A blockiert Strahlen."""
    cx = (ci + 0.5) * cs
    cy = (cj + 0.5) * cs
    return any(ox <= cx <= ox + ow and oy <= cy <= oy + oh
               for ox, oy, ow, oh in obstacles_a)


def cell_is_obstacle_b(ci, cj, obstacles_b, cs):
    """Category-B: nicht erreichbar, aber transparent für Strahlen."""
    cx = (ci + 0.5) * cs
    cy = (cj + 0.5) * cs
    return any(ox <= cx <= ox + ow and oy <= cy <= oy + oh
               for ox, oy, ow, oh in obstacles_b)


def has_los(si, sj, cx, cy, cols, rows, obstacles_a, cs):
    """Line-of-Sight: nur Category-A blockiert den Strahl."""
    steps = int(max(abs(cx - si), abs(cy - sj)) * 4) + 2
    for t in np.linspace(0, 1, steps):
        ri = int(si + t * (cx - si))
        rj = int(sj + t * (cy - sj))
        if 0 <= ri < cols and 0 <= rj < rows:
            if cell_is_obstacle_a(ri, rj, obstacles_a, cs):
                return False
    return True


def compute_coverage(si, sj, cols, rows, obstacles_a, obstacles_b, cs):
    """
    Berechnet abgedeckte Dreiecke für einen Sensor.
    - Category-A-Zellen: solid, blockieren Strahlen, werden übersprungen
    - Category-B-Zellen: transparent, nicht erreichbar → nicht in Coverage
    """
    covered = set()
    rc = SENSOR_RANGE / cs
    for ci in range(cols):
        for cj in range(rows):
            if cell_is_obstacle_a(ci, cj, obstacles_a, cs):
                continue
            if cell_is_obstacle_b(ci, cj, obstacles_b, cs):
                continue   # unreachable → not part of coverage metric
            if (ci - si) ** 2 + (cj - sj) ** 2 > rc ** 2:
                continue
            cx, cy = ci + 0.5, cj + 0.5
            if has_los(si, sj, cx, cy, cols, rows, obstacles_a, cs):
                if sj > cy: covered.add((ci, cj, TOP))
                if si > cx: covered.add((ci, cj, RIGHT))
                if sj < cy: covered.add((ci, cj, BOTTOM))
                if si < cx: covered.add((ci, cj, LEFT))
    return covered


def build_coverage_map(sensors, room):
    cols = int(round(room['width']  / room['cell_size']))
    rows = int(round(room['height'] / room['cell_size']))
    cs   = room['cell_size']
    obs_a = room.get('obstacles_a', room.get('obstacles', []))
    obs_b = room.get('obstacles_b', [])
    cmap = {}
    for idx, s in enumerate(sensors):
        si, sj = s['grid']
        for tri in compute_coverage(si, sj, cols, rows, obs_a, obs_b, cs):
            if tri not in cmap:
                cmap[tri] = idx
    return cmap, cols, rows


# =============================================================================
# SENSOR THREAD
# =============================================================================

class SensorThread(threading.Thread):
    def __init__(self, sensor_info, room):
        super().__init__(daemon=True)
        self.info    = sensor_info
        self.room    = room
        self.port    = sensor_info.get('port', '').strip()
        self.sx      = sensor_info['meters'][0]
        self.sy      = sensor_info['meters'][1]
        self.points  = []
        self.lock    = threading.Lock()
        self.running = True
        self.status  = 'init'
        self._lidar  = None
        self.empty   = (self.port == '<empty>')
        if self.empty:
            self.port = ''

    def run(self):
        if self.empty:
            self.status = 'no sensor'
            return
        if self.port:
            self._run_real()
        else:
            self.status = 'simulated'
            self._run_simulated()

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

                lidar.stop()
                lidar.stop_motor()
                time.sleep(0.3)
                for attr in ('_serial', 'serial'):
                    s = getattr(lidar, attr, None)
                    if s:
                        try: s.reset_input_buffer()
                        except Exception: pass
                        break

                lidar.start_motor()
                time.sleep(1.0)
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
                        pts.append((self.sx + r * math.sin(a),
                                    self.sy + r * math.cos(a)))
                    with self.lock:
                        self.points = pts

            except Exception as e:
                retry += 1
                self.status = f'retry {retry}/{MAX_RETRIES}'
                print(f"S{self.info['id']} error (retry {retry}): {e}")
                time.sleep(1.0)

            finally:
                self._lidar = None
                if lidar is not None:
                    try:
                        lidar.stop()
                        lidar.stop_motor()
                        time.sleep(0.1)
                        lidar.disconnect()
                    except Exception:
                        pass

        if retry >= MAX_RETRIES:
            self.status = 'failed->sim'
            self._run_simulated()

    def _run_simulated(self):
        """
        Simulation: Strahlen werden nur von Category-A-Hindernissen und
        Raumwänden reflektiert. Category-B wird transparent behandelt.
        """
        w    = self.room['width']
        h    = self.room['height']
        obs_a = self.room.get('obstacles_a', self.room.get('obstacles', []))
        # Cat-B intentionally excluded – rays pass through
        angle = 0.0
        while self.running:
            pts = []
            for i in range(360):
                a = math.radians(angle + i)
                for d in np.arange(0.05, SENSOR_RANGE, 0.02):
                    rx = self.sx + d * math.sin(a)
                    ry = self.sy + d * math.cos(a)
                    if not (0 <= rx <= w and 0 <= ry <= h):
                        pts.append((rx, ry))
                        break
                    if any(ox <= rx <= ox + ow and oy <= ry <= oy + oh
                           for ox, oy, ow, oh in obs_a):
                        pts.append((rx, ry))
                        break
            with self.lock:
                self.points = pts
            angle = (angle + 3) % 360
            time.sleep(0.05)

    def get_points(self):
        with self.lock: return list(self.points)

    def pause(self, seconds=3):
        """Stop motor for seconds, then restart. Non-blocking."""
        def _do():
            lidar = self._lidar
            if lidar is None:
                return
            try:
                self.status = f'paused ({seconds}s)...'
                with self.lock:
                    self.points = []
                lidar.stop_motor()
                time.sleep(seconds)
                lidar.start_motor()
                time.sleep(1.0)
                self.status = 'connected'
            except Exception as e:
                self.status = f'pause error: {e}'
        threading.Thread(target=_do, daemon=True).start()

    def stop(self):
        self.running = False


# =============================================================================
# SETUP DIALOG (Tkinter)
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
    """
    Tkinter-Fenster:
      1. JSON-Datei laden (Browse oder Pfad eingeben)
      2. Jedem Sensor einen COM-Port zuweisen (Dropdown)
      3. Start → liefert data-Dict, Cancel → None
    """

    def __init__(self):
        self.result = None
        self._data  = None
        self._port_vars = []

        self.root = tk.Tk()
        self.root.title('LiDAR Viewer – Setup')
        self.root.resizable(False, False)
        self.root.configure(bg='#1e1e2e')
        self._apply_style()
        self._build_ui()
        self.root.mainloop()

    def _apply_style(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('.',
                     background='#1e1e2e', foreground='#ccccdd',
                     font=('Helvetica', 10))
        s.configure('TLabel',      background='#1e1e2e', foreground='#ccccdd')
        s.configure('TFrame',      background='#1e1e2e')
        s.configure('TLabelframe', background='#1e1e2e', foreground='#8888bb')
        s.configure('TLabelframe.Label',
                     background='#1e1e2e', foreground='#8888bb',
                     font=('Helvetica', 9, 'bold'))
        s.configure('TButton',
                     background='#2a4a8a', foreground='white',
                     borderwidth=0, padding=6)
        s.map('TButton', background=[('active', '#3a5a9a')])
        s.configure('TCombobox',
                     fieldbackground='#22223a', foreground='white',
                     background='#22223a', selectbackground='#22223a')
        s.configure('Start.TButton',
                     background='#1a6a3a', foreground='white',
                     font=('Helvetica', 11, 'bold'), padding=8)
        s.map('Start.TButton', background=[('active', '#2a8a4a')])

    def _build_ui(self):
        r = self.root
        pad = dict(padx=14, pady=6)

        tk.Label(r, text='LiDAR Live Viewer',
                 font=('Helvetica', 15, 'bold'),
                 bg='#1e1e2e', fg='white').pack(pady=(16, 2))
        tk.Label(r, text='Setup – Datei laden & Ports zuweisen',
                 bg='#1e1e2e', fg='#666688',
                 font=('Helvetica', 9)).pack(pady=(0, 6))

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        # File row
        ff = ttk.Frame(r); ff.pack(fill='x', **pad)
        ttk.Label(ff, text='JSON-Datei:').pack(side='left')
        self._file_var = tk.StringVar(value='sensor_positions.json')
        ttk.Entry(ff, textvariable=self._file_var, width=34).pack(
            side='left', padx=(8, 4))
        ttk.Button(ff, text='Browse…', command=self._browse).pack(side='left')

        ttk.Button(r, text='JSON laden', command=self._load).pack(pady=(2, 6))

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        # Sensor port table
        self._sf = ttk.LabelFrame(r, text='Sensor-Ports')
        self._sf.pack(fill='x', padx=14, pady=8)
        ttk.Label(self._sf, text='Lade zuerst eine JSON-Datei.',
                  foreground='#666688').pack(pady=10)

        ttk.Separator(r, orient='horizontal').pack(fill='x')

        # Info + buttons
        self._info_var = tk.StringVar(value='')
        ttk.Label(r, textvariable=self._info_var,
                  foreground='#88aaff').pack(**pad)

        bf = ttk.Frame(r); bf.pack(pady=(0, 16))
        ttk.Button(bf, text='Abbrechen', command=r.destroy).pack(
            side='left', padx=8)
        self._start_btn = ttk.Button(
            bf, text='▶  Viewer starten',
            style='Start.TButton',
            command=self._start,
            state='disabled')
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

        room = self._data['room']
        sensors = self._data['sensors']
        obs_a = room.get('obstacles_a', room.get('obstacles', []))
        obs_b = room.get('obstacles_b', [])
        self._info_var.set(
            f"Raum {room['width']} × {room['height']} m  |  "
            f"{len(sensors)} Sensor(en)  |  "
            f"Zellgröße {room['cell_size'] * 100:.0f} cm  |  "
            f"Hind. A: {len(obs_a)}  B: {len(obs_b)}")

        for w in self._sf.winfo_children():
            w.destroy()

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
            cb = ttk.Combobox(self._sf, textvariable=var,
                              values=avail, width=22, state='normal')
            cb.grid(row=i, column=2, padx=(4, 4), pady=4, sticky='w')
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
                info = lidar.get_info()
                lidar.stop()
                lidar.stop_motor()
                lidar.disconnect()
                sn = str(info.get('serialnumber', 'N/A'))
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

    def set_threads(self, threads):
        """Called by LiveViewer to hand over thread references for pausing."""
        self._threads = {t.info['id']: t for t in threads}

    def _pause_sensor(self, sensor_id):
        threads = getattr(self, '_threads', {})
        t = threads.get(sensor_id)
        if t:
            t.pause(3)

    def _start(self):
        if self._data is None: return
        for i, var in enumerate(self._port_vars):
            val = var.get().strip()
            self._data['sensors'][i]['port'] = \
                '' if val == '(simulated)' else val
        self.result = self._data
        self.root.destroy()


# =============================================================================
# LIVE VIEWER (Matplotlib)
# =============================================================================

class LiveViewer:

    def __init__(self, data, setup=None):
        self.room    = data['room']
        self.sensors = data['sensors']
        self.setup   = setup

        # Normalize obstacle keys: support both old ("obstacles") and new format
        if 'obstacles_a' not in self.room and 'obstacles' in self.room:
            self.room['obstacles_a'] = self.room['obstacles']
        if 'obstacles_b' not in self.room:
            self.room['obstacles_b'] = []

        print('Coverage wird berechnet…', end=' ', flush=True)
        self.cmap, self.cols, self.rows = build_coverage_map(
            self.sensors, self.room)
        print(f'{len(self.cmap)} Dreiecke.')

        self.threads = [SensorThread(s, self.room) for s in self.sensors]
        for t in self.threads: t.start()
        if hasattr(self.setup, 'set_threads'):
            self.setup.set_threads(self.threads)

        self._build_figure()
        self._draw_static()
        self._start_animation()
        plt.show()

    def _build_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor(BG)
        self.ax.set_facecolor(ROOM_FACE)
        self.fig.canvas.manager.set_window_title('LiDAR Live Viewer')
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

        ax_btn = self.fig.add_axes([0.01, 0.01, 0.07, 0.04])
        self.btn_stop = MplButton(ax_btn, 'Stop',
                                  color='#5a2222', hovercolor='#8a3333')
        self.btn_stop.label.set_color('white')
        self.btn_stop.on_clicked(self._on_stop)

        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.scan_cols  = [None] * len(self.sensors)
        self.stat_texts = [None] * len(self.sensors)

    def _draw_static(self):
        cs    = self.room['cell_size']
        w     = self.room['width']
        h     = self.room['height']
        obs_a = self.room.get('obstacles_a', [])
        obs_b = self.room.get('obstacles_b', [])

        # Room outline
        self.ax.add_patch(mpatches.Rectangle(
            (0, 0), w, h, lw=2,
            edgecolor=WALL_COLOR, facecolor='none', zorder=1))

        # Category-A obstacles (solid, signal-blocking)
        for ox, oy, ow, oh in obs_a:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh, lw=1.5,
                edgecolor=OBS_A_EDGE, facecolor=OBS_A_FACE, zorder=2))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow / 2, oy + oh / 2, 'Cat. A',
                             ha='center', va='center',
                             fontsize=7, color='#ccccdd', zorder=3)

        # Category-B obstacles (unreachable, transparent – hatched blue)
        for ox, oy, ow, oh in obs_b:
            self.ax.add_patch(mpatches.Rectangle(
                (ox, oy), ow, oh, lw=1.5,
                edgecolor=OBS_B_EDGE, facecolor=OBS_B_FACE,
                hatch='///', zorder=2))
            if ow > 0.25 and oh > 0.15:
                self.ax.text(ox + ow / 2, oy + oh / 2, 'Cat. B',
                             ha='center', va='center',
                             fontsize=7, color='#aaddff', zorder=3)

        # Sensor markers + range rings
        for i, s in enumerate(self.sensors):
            sx, sy = s['meters']
            col    = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            self.ax.add_patch(plt.Circle(
                (sx, sy), SENSOR_RANGE,
                color=col, fill=False, alpha=0.12,
                ls='--', lw=1, zorder=3))
            self.ax.plot(sx, sy, 'o', color=col,
                         ms=12, zorder=5, mec='white', mew=1.2)
            self.ax.text(sx, sy + cs * 1.8, f'S{s["id"]}',
                         ha='center', fontsize=8,
                         fontweight='bold', color=col, zorder=6)
            self.stat_texts[i] = self.ax.text(
                sx, sy - cs * 2.8, '',
                ha='center', fontsize=6, color=col, zorder=6, alpha=0.8)

        # Legend
        handles = [
            mpatches.Patch(color=SENSOR_COLORS[i % len(SENSOR_COLORS)],
                           alpha=0.6, label=f'S{s["id"]}')
            for i, s in enumerate(self.sensors)]
        handles += [
            mpatches.Patch(facecolor=OBS_A_FACE, edgecolor=OBS_A_EDGE,
                           label='Cat. A – signalblockierend'),
            mpatches.Patch(facecolor=OBS_B_FACE, edgecolor=OBS_B_EDGE,
                           hatch='///', label='Cat. B – nicht erreichbar'),
        ]
        self.ax.legend(handles=handles, loc='upper right',
                       facecolor='#22223a', edgecolor='#444466',
                       labelcolor='white', fontsize=8)

        self.ax.set_title('LiDAR Live Viewer',
                          color='#aaaacc', fontsize=11, pad=6)

    def _start_animation(self):
        self._timer = self.fig.canvas.new_timer(interval=REDRAW_INTERVAL)
        self._timer.add_callback(self._update)
        self._timer.start()

    def _update(self):
        for i, t in enumerate(self.threads):
            pts = t.get_points()
            if self.scan_cols[i] is not None:
                self.scan_cols[i].remove()
                self.scan_cols[i] = None
            if pts:
                col = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                self.scan_cols[i] = self.ax.scatter(
                    [p[0] for p in pts], [p[1] for p in pts],
                    s=2, c=col, alpha=0.75, zorder=4, linewidths=0)
            if self.stat_texts[i]:
                self.stat_texts[i].set_text(t.status)
        self.fig.canvas.draw_idle()

    def _on_stop(self, _): self._shutdown()
    def _on_close(self, _): self._shutdown()

    def _shutdown(self):
        if hasattr(self, '_timer'): self._timer.stop()
        for t in self.threads: t.stop()
        print('Viewer beendet.')


# =============================================================================
# ENTRY
# =============================================================================

if __name__ == '__main__':
    dialog = SetupDialog()
    if dialog.result is None:
        print('Abgebrochen.')
        sys.exit(0)
    LiveViewer(dialog.result, setup=dialog)