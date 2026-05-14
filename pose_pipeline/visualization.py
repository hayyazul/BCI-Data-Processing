import numpy as np
from collections import deque
from typing import List
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models import PositionEstimate


class SimplePlotUpdater:
    def __init__(self, position_names: List[str], buffer_seconds=10, hfov_deg=70):
        self.position_names = position_names
        self.buffer_seconds = buffer_seconds
        self.hfov_deg = hfov_deg
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

        # Camera marker at origin
        self.ax_3d.scatter([0], [0], [0], c='black', s=80, marker='^', label='Camera')

        # Camera FOV cone (apex at origin, axis along +Z = camera forward)
        self._draw_view_cone(hfov_deg=hfov_deg, length=0.4)

        # "Up" rod: in OpenCV camera frame +Y is down, so world-up is -Y.
        up_len = 0.25
        self.ax_3d.plot([0, 0], [0, -up_len], [0, 0],
                        color='green', linewidth=2.5, alpha=0.9, label='Up')

        self.lines = {}
        self.markers = {}
        for name in position_names:
            color = self.color_map[name]
            line, = self.ax_3d.plot([], [], [], color=color, linewidth=1.5, alpha=0.85, label=name)
            marker = self.ax_3d.scatter([], [], [], color=[color], s=40, marker='o')
            self.lines[name] = line
            self.markers[name] = marker

        self.ax_3d.legend(fontsize='small', loc='upper left')
        self.ax_3d.set_xlim([-1, 1])
        self.ax_3d.set_ylim([-1, 1])
        self.ax_3d.set_zlim([0, 2])

        # View from slightly behind and above the camera, looking forward (+Z).
        # In OpenCV-camera coords (X right, Y down, Z forward) the matplotlib
        # viewer position is on a sphere: viewer at (-Y above, -Z behind) maps
        # to azim ≈ -90° (so cos(az) ≈ 0, viewer in the Y/Z plane) and a
        # negative elevation so the viewer sits at -Z (behind, since +Z is
        # camera-forward). invert_yaxis flips the Y display so the up-rod
        # ends up pointing visually upward.
        self.ax_3d.view_init(elev=-20, azim=-90)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

    def _draw_view_cone(self, hfov_deg: float, length: float):
        half_angle = np.radians(hfov_deg) / 2
        radius = length * np.tan(half_angle)

        # Edge lines from apex (camera) to base circle
        n_edges = 12
        for i in range(n_edges):
            a = 2 * np.pi * i / n_edges
            self.ax_3d.plot([0, radius * np.cos(a)],
                            [0, radius * np.sin(a)],
                            [0, length],
                            color='red', alpha=0.25, linewidth=0.7)

        # Base circle of the cone
        t = np.linspace(0, 2 * np.pi, 60)
        self.ax_3d.plot(radius * np.cos(t), radius * np.sin(t),
                        np.full_like(t, length),
                        color='red', alpha=0.7, linewidth=1.2,
                        label=f'Camera FOV ({hfov_deg:.0f}°)')

        # Centerline so the facing direction reads even when the cone is small on-screen
        self.ax_3d.plot([0, 0], [0, 0], [0, length],
                        color='red', alpha=0.6, linewidth=1.5)

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
                self.markers[name]._offsets3d = (empty, empty, empty)
                self.markers[name].stale = True
                continue

            xs = np.array([h['x'] for h in recent])
            ys = np.array([h['y'] for h in recent])
            zs = np.array([h['z'] for h in recent])

            self.lines[name].set_data_3d(xs, ys, zs)
            self.markers[name]._offsets3d = (xs[-1:], ys[-1:], zs[-1:])
            self.markers[name].stale = True

            d = float(np.sqrt(xs**2 + ys**2 + zs**2).max())
            if d > max_dist:
                max_dist = d

        cur_xhi = self.ax_3d.get_xlim()[1]
        target = max_dist * 1.3
        if target > cur_xhi or target < cur_xhi * 0.5:
            self.ax_3d.set_xlim([-target, target])
            self.ax_3d.set_ylim([-target, target])
            self.ax_3d.set_zlim([0, max_dist * 1.5])

        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def close(self):
        plt.ioff()
        plt.close('all')


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
    
    print(f"\n=== Playback ===")
    print(f"Frames: {len(records)}")
    for name, recs in by_name.items():
        trans = np.array([r.translation.flatten() for r in recs])
        print(f"  {name}: dist={np.mean(np.linalg.norm(trans,axis=1)):.3f}m")
    
    plt.show()