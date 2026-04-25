import numpy as np
from collections import deque
from typing import List
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models import PositionEstimate


# ==============================================================================
# SIMPLE PLOT UPDATER (manually update simple plot every N frames)
# ==============================================================================

class SimplePlotUpdater:
    """
    Real-time 3D position plot. Pre-creates Line3D / scatter artists per
    position and updates their data each tick instead of clearing+redrawing.
    Throttled by wall-clock so the plot can't starve the camera loop.
    """
    def __init__(self, position_names: List[str], buffer_seconds=10):
        self.position_names = position_names
        self.buffer_seconds = buffer_seconds
        self.history = {name: deque(maxlen=5000) for name in position_names}
        self.start_time = None
        self.last_plot_update = 0.0
        self.min_update_interval = 1.0 / 20  # 20 Hz cap for plot

        self.colors = plt.cm.tab10(np.linspace(0, 1, max(len(position_names), 1)))
        self.color_map = dict(zip(position_names, self.colors))

        plt.ion()
        self.fig = plt.figure(figsize=(7, 6))
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('Live 3D Position')
        self.ax_3d.invert_yaxis()
        self.ax_3d.scatter([0], [0], [0], c='black', s=80, marker='^', label='Camera')

        self.lines = {}
        self.markers = {}
        for name in position_names:
            color = self.color_map[name]
            line, = self.ax_3d.plot([], [], [], color=color, linewidth=1.5, alpha=0.85, label=name)
            # Marker as a Line3D with no line — 3D scatter's _offsets3d hack
            # leaves the internal facecolor array unsized, so the dot never renders.
            marker, = self.ax_3d.plot([], [], [], color=color, marker='o',
                                       markersize=8, linestyle='None')
            self.lines[name] = line
            self.markers[name] = marker

        self.ax_3d.legend(fontsize='small', loc='upper left')
        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-1, 1])
        self.ax_3d.set_zlim([0, 2])

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

    def add_point(self, name: str, x: float, y: float, z: float, timestamp: float):
        if self.start_time is None:
            self.start_time = timestamp
        self.history[name].append({
            't': timestamp - self.start_time,
            'x': x, 'y': y, 'z': z
        })

    def update(self, now: float):
        if now - self.last_plot_update < self.min_update_interval:
            return
        self.last_plot_update = now

        if self.start_time is None:
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
            return

        elapsed = now - self.start_time
        cutoff = elapsed - self.buffer_seconds

        max_dist = 0.5
        empty = np.empty(0)
        for name in self.position_names:
            hist = list(self.history[name])
            recent = [h for h in hist if h['t'] >= cutoff]
            if not recent:
                self.lines[name].set_data_3d(empty, empty, empty)
                self.markers[name].set_data_3d(empty, empty, empty)
                continue

            xs = np.array([h['x'] for h in recent])
            ys = np.array([h['y'] for h in recent])
            zs = np.array([h['z'] for h in recent])

            self.lines[name].set_data_3d(xs, ys, zs)
            self.markers[name].set_data_3d(xs[-1:], ys[-1:], zs[-1:])

            d = float(np.sqrt(xs**2 + ys**2 + zs**2).max())
            if d > max_dist:
                max_dist = d

        cur_xhi = self.ax_3d.get_xlim()[1]
        target = max_dist * 1.3
        if target > cur_xhi or target < cur_xhi * 0.5:
            self.ax_3d.set_xlim([-target, target])
            self.ax_3d.set_ylim([-target, target])
            self.ax_3d.set_zlim([0, max_dist * 1.5])

        # --- CHANGED: force a full redraw instead of idle draw ---
        try:
            self.fig.canvas.draw()                # immediate redraw
            self.fig.canvas.flush_events()
            plt.pause(0.001)                      # let the GUI breathe
        except Exception:
            pass

    def close(self):
        plt.ioff()
        plt.close('all')


# ==============================================================================
# PLAYBACK - unchanged
# ==============================================================================

def show_playback(records: List[PositionEstimate], start_time: float):
    if not records:
        return
    
    by_name = {}
    for r in records:
        by_name.setdefault(r.position_name, []).append(r)
    
    num = len(by_name)
    fig = plt.figure(figsize=(6 * num, 8))
    
    for i, (name, recs) in enumerate(by_name.items()):
        timestamps = np.array([r.timestamp - start_time for r in recs])
        translations = np.array([r.translation.flatten() for r in recs])
        
        ax = fig.add_subplot(2, num, i + 1, projection='3d')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title(f'{name} - 3D')
        ax.invert_yaxis()
        ax.scatter([0], [0], [0], c='black', s=50, marker='^')
        
        colors = plt.cm.plasma(timestamps / max(timestamps) if max(timestamps) > 0 else 1)
        for j in range(len(translations) - 1):
            ax.plot3D(translations[j:j+2, 0], translations[j:j+2, 1], translations[j:j+2, 2],
                     color=colors[j], linewidth=1, alpha=0.7)
        
        ax.scatter(*translations[0], c='green', s=40, marker='o', label='Start')
        ax.scatter(*translations[-1], c='red', s=40, marker='s', label='End')
        ax.legend()
        
        ax2 = fig.add_subplot(2, num, num + i + 1)
        for comp, label in enumerate(['X', 'Y', 'Z']):
            ax2.plot(timestamps, translations[:, comp], label=label, linewidth=1)
        ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position (m)')
        ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Summary
    print(f"\n=== Playback ===")
    print(f"Frames: {len(records)}")
    for name, recs in by_name.items():
        trans = np.array([r.translation.flatten() for r in recs])
        print(f"  {name}: dist={np.mean(np.linalg.norm(trans,axis=1)):.3f}m")
    
    plt.show()