"""Interactive 3D arm visualization.

ArmVisualizer overlays one or more arms (shoulder -> elbow -> wrist) in a
single Plotly scene with a play/pause animation and a time slider. Each arm
can be supplied either as joint angles (driven through a forward-kinematics
function) or as raw elbow/wrist position arrays.
"""

import numpy as np
import matplotlib.colors as mcolors
import plotly.graph_objects as go


class ArmVisualizer:
    def __init__(self, fk_func=None, L1=None, L2=None, scene_bounds=(-0.8, 0.8)):
        self.fk_func = fk_func
        self.L1 = L1
        self.L2 = L2
        self.scene_bounds = scene_bounds
        self._traces = []
        self._n_frames = None

    def _to_hex(self, color):
        try:
            return mcolors.to_hex(color)
        except ValueError:
            return color

    def _to_rgb(self, color):
        hex_color = self._to_hex(color).lstrip('#')
        return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    def _blend_color(self, base_color, tint_color, factor=0.2):
        """factor=0.0 -> pure base_color; factor=1.0 -> pure tint_color."""
        br, bg, bb = self._to_rgb(base_color)
        tr, tg, tb = self._to_rgb(tint_color)
        r = int(br + (tr - br) * factor)
        g = int(bg + (tg - bg) * factor)
        b = int(bb + (tb - bb) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'

    def add_arm(self, name, color='orange', dash='solid', width=4,
                q1=None, q2=None, q3=None, q4=None, L1=None, L2=None,
                elbow=None, wrist=None):
        if q1 is not None:
            if self.fk_func is None:
                raise ValueError("You must supply fk_func in constructor to use FK arms.")
            L1 = L1 if L1 is not None else self.L1
            L2 = L2 if L2 is not None else self.L2
            if L1 is None or L2 is None:
                raise ValueError("L1 and L2 must be given either here or in constructor.")
            q1 = np.atleast_1d(q1); q2 = np.atleast_1d(q2)
            q3 = np.atleast_1d(q3); q4 = np.atleast_1d(q4)
            frames = []
            for i in range(len(q1)):
                elb, wri = self.fk_func(q1[i], q2[i], q3[i], q4[i], L1, L2)
                frames.append((np.asarray(elb), np.asarray(wri)))
        elif elbow is not None and wrist is not None:
            elbow = np.atleast_2d(elbow) if np.ndim(elbow) == 1 else np.asarray(elbow)
            wrist = np.atleast_2d(wrist) if np.ndim(wrist) == 1 else np.asarray(wrist)
            frames = [(elbow[i], wrist[i]) for i in range(len(elbow))]
        else:
            raise ValueError("Provide joint angles (q1..q4) or elbow+wrist positions.")

        n = len(frames)
        if self._n_frames is None:
            self._n_frames = n
        elif n == 1:
            frames = frames * self._n_frames
        elif n != self._n_frames:
            raise ValueError(f"Frame count mismatch: {n} vs {self._n_frames}")

        blend = 0.2
        joint_colors = [
            self._blend_color('#000000', color, factor=blend),   # shoulder
            self._blend_color('#3366cc', color, factor=blend),   # elbow
            self._blend_color('#cc3333', color, factor=blend),   # wrist
        ]

        self._traces.append(dict(name=name, color=color, dash=dash, width=width,
                                 joint_colors=joint_colors, frames=frames))
        return self

    def show(self, times=None):
        if not self._traces:
            print("No arms added.")
            return
        n = self._n_frames
        if times is None:
            times = list(range(n))
        elif len(times) != n:
            raise ValueError("times length must match frame count")

        fig = go.Figure()
        for tr in self._traces:
            elb, wri = tr['frames'][0]
            fig.add_trace(go.Scatter3d(
                x=[0, elb[0], wri[0]], y=[0, elb[1], wri[1]], z=[0, elb[2], wri[2]],
                mode='lines+markers',
                marker=dict(size=6, color=tr['joint_colors']),
                line=dict(color=tr['color'], width=tr['width'], dash=tr['dash']),
                name=tr['name']))

        plot_frames = []
        for i in range(n):
            traces = []
            for tr in self._traces:
                elb, wri = tr['frames'][i]
                traces.append(go.Scatter3d(
                    x=[0, elb[0], wri[0]], y=[0, elb[1], wri[1]], z=[0, elb[2], wri[2]],
                    mode='lines+markers',
                    marker=dict(size=6, color=tr['joint_colors']),
                    line=dict(color=tr['color'], width=tr['width'], dash=tr['dash']),
                    showlegend=False))
            plot_frames.append(go.Frame(data=traces, name=f'frame{i}'))
        fig.frames = plot_frames

        b = self.scene_bounds
        fig.update_layout(
            scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
                       aspectmode='cube', xaxis_range=[b[0], b[1]],
                       yaxis_range=[b[0], b[1]], zaxis_range=[b[0], b[1]]),
            updatemenus=[dict(type='buttons',
                              buttons=[dict(label='▶ Play', method='animate',
                                            args=[None, {'frame': {'duration': 50, 'redraw': True},
                                                         'fromcurrent': True}]),
                                       dict(label='⏸ Pause', method='animate',
                                            args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                                           'mode': 'immediate', 'transition': {'duration': 0}}])],
                              direction='left', pad={'r': 10, 't': 87}, showactive=False,
                              x=0.1, xanchor='right', y=0, yanchor='top')],
            sliders=[dict(active=0, yanchor='top', xanchor='left',
                          currentvalue=dict(font=dict(size=16), prefix='Time: ', visible=True, xanchor='right'),
                          transition=dict(duration=30, easing='cubic-in-out'),
                          pad=dict(b=10, t=50), len=0.9, x=0.1, y=0,
                          steps=[dict(args=[[f'frame{k}'], {'frame': {'duration': 0, 'redraw': True},
                                                            'mode': 'immediate', 'transition': {'duration': 0}}],
                                      label=f'{times[k]:.2f}s', method='animate') for k in range(n)])]
        )
        fig.show(renderer='iframe')
