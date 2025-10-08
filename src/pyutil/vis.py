
from __future__ import annotations
import colorsys
import math
import random
from typing import List, Iterable, Tuple, Optional

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, Normalize
import matplotlib.colors as mcolors

import os
import pickle
import plotly.graph_objects as go
import plotly.express as px

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#region Color maps
def get_cmap_with_black_for_0(cmap='jet', reversed_Q=False):
    cmap_hdl = plt.cm.get_cmap(cmap, 256)
    cmap_val = cmap_hdl(np.linspace(0, 1, 256))
    if reversed_Q:
        cmap_val = cmap_val[::-1]
    cmap_val[0] = [0, 0, 0, 1]  # RGBA (Black with full opacity)
    new_cmap = mcolors.ListedColormap(cmap_val)
    return new_cmap

def get_random_bright_color_in_hex(seed=None, s_range=(0.70, 0.95), v_range=(0.90, 1.00)):
    rng = random.Random(seed) if seed is not None else random
    h = rng.random()                              # 0..1
    s = rng.uniform(*s_range)                     # keep saturation high
    v = rng.uniform(*v_range)                     # keep value/brightness high
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'

def _rgb_to_hex(r: float, g: float, b: float) -> str:
    """r,g,b in [0,1] → '#RRGGBB'"""
    return "#{:02X}{:02X}{:02X}".format(
        int(round(max(0, min(1, r)) * 255)),
        int(round(max(0, min(1, g)) * 255)),
        int(round(max(0, min(1, b)) * 255)),
    )

def _hsv_to_hex(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return _rgb_to_hex(r, g, b)

def _circular_hue_distance(a: float, b: float) -> float:
    """Smallest distance around the hue circle in [0,1]."""
    d = abs((a % 1.0) - (b % 1.0))
    return min(d, 1.0 - d)

def _perceptual_distance_hsv(a: Tuple[float, float, float],
                             b: Tuple[float, float, float]) -> float:
    """
    Fast, HSV-based distance tuned for label distinction.
    We weight hue a lot, saturation/value a bit.
    Returns ~[0..1.7] (not a true metric, but good enough to separate labels).
    """
    dh = _circular_hue_distance(a[0], b[0]) * 2.0      # hue matters most
    ds = abs(a[1] - b[1]) * 0.6
    dv = abs(a[2] - b[2]) * 0.5
    return dh + ds + dv

class DistinctColorGenerator:
    """
    Generate bright, high-contrast hex colors suitable for differentiating labels.

    Strategy:
      • Distribute hues using the golden-angle step for maximal spread.
      • Lock in high saturation & value (defaults: s=0.90, v=0.95) → vivid colors.
      • Optionally enforce a minimum HSV distance from previously returned colors
        to avoid near-duplicates when repeat=False.

    Parameters
    ----------
    saturation : float (0..1)
        Default 0.90 for vivid colors.
    value : float (0..1)
        Default 0.95 for brightness.
    seed : Optional[int]
        For reproducible sequences (affects initial hue jitter).
    min_distance : float
        Minimum HSV distance from prior colors when repeat=False.
        0.65–0.9 is a good range. Default 0.8.
    start_hue : Optional[float] in [0,1]
        Where to start on the color wheel. Random if not provided.
    allow_repeat : bool
        If True, colors can repeat/cycle. If False, we avoid near-duplicates.

    Notes
    -----
    • This is fast and has no dependencies. For ultra-rigorous perceptual spacing
      (e.g., print colorimetry), consider generating in CIELAB and using ΔE.
    """

    GOLDEN_ANGLE = (math.sqrt(5) - 1) / 2  # ~0.61803398875 (fraction of full turn)

    def __init__(self,
                 saturation: float = 0.90,
                 value: float = 0.95,
                 seed: Optional[int] = None,
                 min_distance: float = 0.80,
                 start_hue: Optional[float] = None,
                 allow_repeat: bool = False):
        self.s = float(saturation)
        self.v = float(value)
        self.min_distance = float(min_distance)
        self.allow_repeat = bool(allow_repeat)
        self._i = 0
        self._used_hsv: List[Tuple[float, float, float]] = []

        rng = random.Random(seed)
        self._h0 = rng.random() if start_hue is None else (start_hue % 1.0)
        # Small random jitter helps avoid aligning with primary/secondary axes.
        self._jitter = (rng.random() - 0.5) * 0.05  # ±0.025 hue

    def _candidate_hue(self, idx: int) -> float:
        # Evenly hop around the wheel using the golden angle
        h = (self._h0 + idx * self.GOLDEN_ANGLE + self._jitter) % 1.0
        return h

    def _is_far_enough(self, hsv: Tuple[float, float, float]) -> bool:
        return all(_perceptual_distance_hsv(hsv, seen) >= self.min_distance
                   for seen in self._used_hsv)

    def next_hex(self) -> str:
        """
        Get the next color as '#RRGGBB'.
        Respects allow_repeat and min_distance.
        """
        attempt = 0
        while True:
            h = self._candidate_hue(self._i)
            self._i += 1
            hsv = (h, self.s, self.v)

            if self.allow_repeat or not self._used_hsv:
                self._used_hsv.append(hsv)
                return _hsv_to_hex(*hsv)

            if self._is_far_enough(hsv):
                self._used_hsv.append(hsv)
                return _hsv_to_hex(*hsv)

            # If we're being too strict, slightly vary saturation/value to escape clashes.
            attempt += 1
            if attempt > 32:
                # Relax constraints to avoid infinite loops when many colors are requested.
                self.min_distance *= 0.95
                self.s = min(1.0, self.s + 0.01)
                self.v = min(1.0, self.v + 0.01)

    def take(self, n: int) -> List[str]:
        """Get n colors at once."""
        return [self.next_hex() for _ in range(n)]

def distinct_hex_colors(n: int,
                        allow_repeat: bool = False,
                        saturation: float = 0.90,
                        value: float = 0.95,
                        seed: Optional[int] = None,
                        min_distance: float = 0.80) -> List[str]:
    """
    Convenience function: generate n bright, distinct hex colors.

    Examples
    --------
    >>> distinct_hex_colors(5)
    ['#F23D3D', '#42F26E', '#3E45F2', '#F2E43E', '#9E3EF2']
    """
    gen = DistinctColorGenerator(
        saturation=saturation,
        value=value,
        seed=seed,
        min_distance=min_distance,
        allow_repeat=allow_repeat,
    )
    return gen.take(n)
#endregion

def compute_three_view_mip(data):
    # data is a 3-dimension array in the order of zyx
    mip = {}
    mip['yx'] = np.amax(data, axis=0)
    mip['zx'] = np.amax(data, axis=1)
    mip['zy'] = np.amax(data, axis=2)
    return mip

def imfuse_2d(im1, im2, method='stack'):
    # Assume uint16 at the moment. Convert to 8 bit 
    im1 = (im1 / 256).astype('uint8')
    im2 = (im2 / 256).astype('uint8')
    if method == 'stack':
        return np.dstack((im1, im2, im1))
    elif method == 'matlab':
        return np.dstack((im1, im2, np.maximum(im1, im2)))
    elif method == 'VS':
        # Red vessel, gray skull
        imc = np.tile(im2, (3, 1, 1)) * 0.5
        imc[0] = np.maximum(imc[2], im1)
        return np.transpose(imc.astype('uint8'), (1, 2, 0))

def implay(data, xlabel=None, ylabel=None, mag=1.5, fix_scale_Q=False, show_scale_Q=False, cmap=None, reverse_y_dir_Q=True):
    # Note: to open a seperate window for visualization can use the following code: 
    # import webbrowser
    # import tempfile
    # temp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    # fig.write_html(temp.name)
    # webbrowser.open('file://' + os.path.realpath(temp.name))
    data_shape = data.shape
    frames = [data[i] for i in range(data_shape[0])]
    if fix_scale_Q: 
        zmin = data.flatten().min()
        zmax = data.flatten().max()
    else:
        zmin, zmax = None, None
    if reverse_y_dir_Q:
        y_axis_dir = 'reversed'
    else:
        y_axis_dir = True
    
    if cmap is None:
        if data.dtype == 'bool':
            cmap = 'Grey'
        else:
            cmap = 'Jet'

    fig = go.Figure()

    # Add images as traces, only the first image is visible
    for i, frame in enumerate(frames):
        fig.add_trace(
            go.Heatmap(
                z=frame,
                colorscale=cmap,
                showscale=show_scale_Q,
                visible=True if i == 0 else False, 
                zmin=zmin, zmax=zmax
            )
        )

    # Create and add slider
    steps = []
    for i in range(len(frames)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(frames)},
                {"title": f"Frame {i + 1}"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders, 
        xaxis=dict(scaleanchor="y", scaleratio=1, title=xlabel),  # Ensuring 1:1 aspect ratio
        yaxis=dict(autorange=y_axis_dir, title=ylabel),
        width=int(data_shape[2] * mag),  # Adjust width and height if needed
        height=int(data_shape[1] * mag),
        title='Image Stack Viewer'
    )

    fig.show()

def scatter3(x=None, y=None, z=None, figsize=(600, 600), reverse_z_dir_Q=False, \
             marker_size=1, aspectmode='cube', sub=None, title=None):
    if reverse_z_dir_Q:
        z_dir = "reversed"
    else:
        z_dir = True

    if sub is not None: 
        x = sub[2]
        y = sub[1]
        z = sub[0]

    fig = px.scatter_3d(x=x, y=y, z=z)
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(
        width=figsize[1], height=figsize[0],
        scene=dict(zaxis=dict(autorange=z_dir),
                   aspectmode=aspectmode), 
        title=title
                   )
    fig.show()
    return fig
    

def imshow_w_dots(data, pos, mag=1, cmap='Gray'):
    assert isinstance(pos, (tuple, list)), "pos should be a tuple of (x : np.array, y : np.array)"
    fig = go.Figure(go.Heatmap(
        z=data,
        x=np.arange(data.shape[1]),  # Assuming square data
        y=np.arange(data.shape[0]),
        colorscale=cmap
    ))

    # Add the scatter plot
    fig.add_trace(go.Scatter(
        x=pos[2],
        y=pos[1],
        mode='markers',
        marker=dict(
            size=5,
            color='Green',  # Contrast color
            symbol='circle'
        )
    ))

    # Update layout if needed
    fig.update_layout(
        title='Heatmap with Scatter Plot Overlay',
        xaxis=dict(scaleanchor="y", scaleratio=1, title='X'),  # Ensuring 1:1 aspect ratio
        yaxis=dict(autorange="reversed", title='Y'),
        width=int(data.shape[1] * mag),  # Adjust width and height if needed
        height=int(data.shape[0] * mag),
    )

    # Show the figure
    fig.show()

def vis_3_view_mip(mips, figsize=(8, 8), cmap='gray', colorbarQ=False,
                   cbar_label=None, c_scale='normal', show_axes_Q=False, 
                   fig_title=None, visQ=True, v_min=None, v_max=None):
    """
    Show the 3-view MIP (max intensity projection) in a single figure with
    the YX view in the center, the ZY view on the left, and the ZX view below.

    Parameters
    ----------
    mip_yx : 2D numpy array of shape (Y, X)
        The max intensity projection over Z, so it shows the YX plane.
    mip_zy : 2D numpy array of shape (Z, Y)
        The max intensity projection over X, so it shows the ZY plane.
    mip_zx : 2D numpy array of shape (Z, X)
        The max intensity projection over Y, so it shows the ZX plane.
    cmap : str, optional
        Colormap for displaying the images.
    """
    mip_yx = mips['yx']
    mip_zx = mips['zx']
    mip_zy = mips['zy']

    # Dimensions for each MIP
    y_size, x_size = mip_yx.shape     
    z_size, y2_size = mip_zy.shape    
    z2_size, x2_size = mip_zx.shape   

    # Sanity check (not strictly required, but nice to confirm)
    assert y2_size == y_size, "mip_zy second dim must match mip_yx first dim"
    assert x2_size == x_size, "mip_zx second dim must match mip_yx second dim"
    assert z_size == z2_size, "mip_zy first dim must match mip_zx first dim"

    # Use a 2x2 GridSpec, leaving bottom-left cell empty
    # We want widths to be [width of ZY, width of YX] = [z_size, x_size]
    # and heights to be [height of YX, height of ZX] = [y_size, z_size].
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[x_size, z_size],
        height_ratios=[y_size, z_size]
    )

    # Top-left: ZY view
    ax_yz = fig.add_subplot(gs[0, 1])
    # Top-right: YX view
    ax_yx = fig.add_subplot(gs[0, 0])
    # Bottom-right: ZX view
    ax_zx = fig.add_subplot(gs[1, 0])
    # Bottom-left: unused (blank)
    ax_blank = fig.add_subplot(gs[1, 1])
    if c_scale == 'normal': 
        c_norm = Normalize()
    elif c_scale == 'log':
        c_norm = LogNorm()
        min_val = np.inf
        for tmp_im in mips.values():
            min_val = np.minimum(min_val, np.min(tmp_im[tmp_im > 0]))
        mip_zy[mip_zy == 0] = 0.5 * min_val
        mip_yx[mip_yx == 0] = 0.5 * min_val
        mip_zx[mip_zx == 0] = 0.5 * min_val

    if cmap.endswith('0'):
        cmap = get_cmap_with_black_for_0(cmap[:-1])
    elif cmap.endswith('0r'): 
        cmap = get_cmap_with_black_for_0(cmap[:-2], reversed_Q=True)

    ax_yz.set_xlabel('Z')
    ax_yz.set_ylabel('Y')
    
    ax_zx.set_xlabel('X')
    ax_zx.set_ylabel('Z')

    ax_yx.set_xlabel('X')
    ax_yx.set_ylabel('Y')

    im_yz = ax_yz.imshow(mip_zy.T, cmap=cmap, aspect='auto', norm=c_norm, interpolation='nearest')
    im_yx = ax_yx.imshow(mip_yx, cmap=cmap, aspect='auto', norm=c_norm, interpolation='nearest')
    im_zx = ax_zx.imshow(mip_zx, cmap=cmap, aspect='auto', norm=c_norm, interpolation='nearest')
    
    if not show_axes_Q: 
        for ax in [ax_yz, ax_yx, ax_zx]:
            ax.axis('off')

    if fig_title is not None: 
        fig.suptitle(fig_title)    
    plt.tight_layout()    

    if colorbarQ: 
        pos = ax_blank.get_position()  # get current position [left, bottom, width, height]
        ax_blank.set_position([pos.x0, pos.y0, 0.2 * pos.width, pos.height])  # make it narrower
        cbar = fig.colorbar(im_yx, cax=ax_blank, label=cbar_label)
        # if cbar_label is not None: 
        #     cbar.ax.set_ylabel = cbar_label
    else: 
        ax_blank.set_visible(False)

    if (v_max is not None) or (v_min is not None):
         curr_vmin, curr_vmax = im_yz.get_clim()
         v_min = curr_vmin if v_min is None else v_min
         v_max = curr_vmax if v_max is None else v_max
         im_yx.set_clim(vmin=v_min, vmax=v_max)
         im_zx.set_clim(vmin=v_min, vmax=v_max)
         im_yz.set_clim(vmin=v_min, vmax=v_max)

    if visQ: 
        plt.show()
    return fig, ax_yx, ax_yz, ax_zx, ax_blank

def vis_mips(mips, pts_zyx=None, figsize=(8, 8), cmap='gray', vis_line_Q=False, vis_pt_sz=5, 
             colorbarQ=False, cbar_label=None, c_scale='normal', show_axes_Q=False, 
             fig_title=None, visQ=True, v_min=None, v_max=None):
    """
    Show the 3-view MIP (max intensity projection) in a single figure with
    the YX view in the center, the ZY view on the left, and the ZX view below.

    Parameters
    ----------
    mip_yx : 2D numpy array of shape (Y, X)
        The max intensity projection over Z, so it shows the YX plane.
    mip_zy : 2D numpy array of shape (Z, Y)
        The max intensity projection over X, so it shows the ZY plane.
    mip_zx : 2D numpy array of shape (Z, X)
        The max intensity projection over Y, so it shows the ZX plane.
    pts_zyx: (3, N) np.array, point positions
    cmap : str, optional
        Colormap for displaying the images.
    """
    fig, ax_yx, ax_yz, ax_zx, ax_blank = vis_3_view_mip(mips, figsize=figsize, cmap=cmap, colorbarQ=colorbarQ, 
                                                        cbar_label=cbar_label, c_scale=c_scale, show_axes_Q=show_axes_Q,  
                                                        fig_title=fig_title, visQ=False, v_min=v_min, v_max=v_max)
    vis_pt_alpha = 1
    if pts_zyx is not None:
        if vis_line_Q:
            ax_yz.plot(pts_zyx[0], pts_zyx[1], color='g', alpha=0.75)
            ax_yx.plot(pts_zyx[2], pts_zyx[1], color='g', alpha=0.75)
            ax_zx.plot(pts_zyx[2], pts_zyx[0], color='g', alpha=0.75)
        ax_yz.scatter(pts_zyx[0], pts_zyx[1], vis_pt_sz, color='r', alpha=vis_pt_alpha)
        ax_yx.scatter(pts_zyx[2], pts_zyx[1], vis_pt_sz, color='r', alpha=vis_pt_alpha)
        ax_zx.scatter(pts_zyx[2], pts_zyx[0], vis_pt_sz, color='r', alpha=vis_pt_alpha)

    if visQ: 
        plt.show()
    return fig

def vis_pts_w_local_mips(pts_zyx, im_vol, bbox_expand=10, vis_line_Q=True, show_axes_Q=False, fig_title=None):
    assert pts_zyx.shape[0] == 3, ValueError(f"pts_zyx must be a (3, N) np.array")
    bbox_mm = np.maximum(0, np.min(pts_zyx, axis=1) - bbox_expand).astype(np.int32)
    bbox_xx = np.minimum(im_vol.shape, np.max(pts_zyx, axis=1) + bbox_expand).astype(np.int32)
    local_zyx = pts_zyx - bbox_mm[:, None]
    local_im = im_vol[bbox_mm[0] : bbox_xx[0], bbox_mm[1] : bbox_xx[1], bbox_mm[2] : bbox_xx[2]]
    local_mips = compute_three_view_mip(local_im)

    fg = vis_mips(local_mips, pts_zyx=local_zyx, vis_line_Q=vis_line_Q, show_axes_Q=show_axes_Q, fig_title=fig_title)
    return fg

def vis_mips_w_ptl(mips, ptl_zyx=None, figsize=(8, 8), vis_type='line',cmap='gray', show_legend_Q=True, 
             colorbarQ=False, cbar_label=None, c_scale='normal', show_axes_Q=False, 
             fig_title=None, visQ=True, v_min=None, v_max=None):
    """
    Show the 3-view MIP (max intensity projection) in a single figure with
    the YX view in the center, the ZY view on the left, and the ZX view below.

    Parameters
    ----------
    mip_yx : 2D numpy array of shape (Y, X)
        The max intensity projection over Z, so it shows the YX plane.
    mip_zy : 2D numpy array of shape (Z, Y)
        The max intensity projection over X, so it shows the ZY plane.
    mip_zx : 2D numpy array of shape (Z, X)
        The max intensity projection over Y, so it shows the ZX plane.
    ptl_zyx: list of (3, N) np.array, point positions
    cmap : str, optional
        Colormap for displaying the images.
    """
    fig, ax_yx, ax_yz, ax_zx, ax_blank = vis_3_view_mip(mips, figsize=figsize, cmap=cmap, colorbarQ=colorbarQ, 
                                                        cbar_label=cbar_label, c_scale=c_scale, show_axes_Q=show_axes_Q,  
                                                        fig_title=fig_title, visQ=False, v_min=v_min, v_max=v_max)
    if not isinstance(vis_type, list): 
        vis_type = [vis_type]
    
    if len(vis_type) > 1: 
        vis_alpha = 0.5
    else: 
        vis_alpha = 1

    dot_sz = 2.5
    if ptl_zyx is not None:
        num_lines = len(ptl_zyx)
        if isinstance(ptl_zyx, list):
            # colors = pl.cm.jet(np.linspace(0,1,len(ptl_zyx)))
            for i, zyx in enumerate(ptl_zyx): 
                if 'line' in vis_type: 
                    ax_yz.plot(zyx[0], zyx[1], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                    ax_yx.plot(zyx[2], zyx[1], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                    ax_zx.plot(zyx[2], zyx[0], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                if 'dot' in vis_type: 
                    ax_yz.scatter(zyx[0], zyx[1], dot_sz, label=f"{i}", alpha=vis_alpha)
                    ax_yx.scatter(zyx[2], zyx[1], dot_sz, label=f"{i}", alpha=vis_alpha)
                    ax_zx.scatter(zyx[2], zyx[0], dot_sz, label=f"{i}", alpha=vis_alpha)
        elif isinstance(ptl_zyx, dict):
            for i, zyx in ptl_zyx.items(): 
                if 'line' in vis_type: 
                    ax_yz.plot(zyx[0], zyx[1], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                    ax_yx.plot(zyx[2], zyx[1], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                    ax_zx.plot(zyx[2], zyx[0], label=f"{i}", marker='.', markersize=dot_sz, alpha=vis_alpha)
                if 'dot' in vis_type: 
                    ax_yz.scatter(zyx[0], zyx[1], dot_sz, label=f"{i}", alpha=vis_alpha)
                    ax_yx.scatter(zyx[2], zyx[1], dot_sz, label=f"{i}", alpha=vis_alpha)
                    ax_zx.scatter(zyx[2], zyx[0], dot_sz, label=f"{i}", alpha=vis_alpha)
                    
        if show_legend_Q: 
            ax_yx.legend(ncol=3, handlelength=0.5, columnspacing=0.25)

    if visQ: 
        plt.show()
    return fig

def vis_mips_w_ptl_n_weight(mips, ptl_zyx=None, plt_weight=None, figsize=(8, 8), cmap='gray', show_legend_Q=True, 
             colorbarQ=False, cbar_label=None, c_scale='normal', show_axes_Q=False, 
             fig_title=None, visQ=True, v_min=None, v_max=None):
    """
    Show the 3-view MIP (max intensity projection) in a single figure with
    the YX view in the center, the ZY view on the left, and the ZX view below.

    Parameters
    ----------
    mip_yx : 2D numpy array of shape (Y, X)
        The max intensity projection over Z, so it shows the YX plane.
    mip_zy : 2D numpy array of shape (Z, Y)
        The max intensity projection over X, so it shows the ZY plane.
    mip_zx : 2D numpy array of shape (Z, X)
        The max intensity projection over Y, so it shows the ZX plane.
    ptl_zyx: list of (3, N) np.array, point positions
    cmap : str, optional
        Colormap for displaying the images.
    """
    fig, ax_yx, ax_yz, ax_zx, ax_blank = vis_3_view_mip(mips, figsize=figsize, cmap=cmap, colorbarQ=colorbarQ, 
                                                        cbar_label=cbar_label, c_scale=c_scale, show_axes_Q=show_axes_Q,  
                                                        fig_title=fig_title, visQ=False, v_min=v_min, v_max=v_max)
    if ptl_zyx is not None:
        if isinstance(ptl_zyx, list):
            # colors = pl.cm.jet(np.linspace(0,1,len(ptl_zyx)))
            for i, zyx in enumerate(ptl_zyx): 
                # if vis_line_Q:
                ax_yz.plot(zyx[0], zyx[1], label=f"{i}", linewidth=plt_weight[i])
                ax_yx.plot(zyx[2], zyx[1], label=f"{i}", linewidth=plt_weight[i])
                ax_zx.plot(zyx[2], zyx[0], label=f"{i}", linewidth=plt_weight[i])
                # ax_yz.scatter(zyx[0], zyx[1], 1, label=f"{i}")
                # ax_yx.scatter(zyx[2], zyx[1], 1, label=f"{i}")
                # ax_zx.scatter(zyx[2], zyx[0], 1, label=f"{i}")
        elif isinstance(ptl_zyx, dict):
            for i, zyx in ptl_zyx.items(): 
                # if vis_line_Q:
                ax_yz.plot(zyx[0], zyx[1], label=f"{i}", linewidth=plt_weight[i])
                ax_yx.plot(zyx[2], zyx[1], label=f"{i}", linewidth=plt_weight[i])
                ax_zx.plot(zyx[2], zyx[0], label=f"{i}", linewidth=plt_weight[i])
        if show_legend_Q: 
            ax_yx.legend(ncol=3, handlelength=0.5, columnspacing=0.25)

    if visQ: 
        plt.show()
    return fig

def vis_ax_vis_tree(ax_tree, tree_dict):

    def find_all_nodes(tree_dict):
        """
        Return a set of all nodes mentioned either as keys (parents)
        or in the child lists (children).
        """
        all_nodes = set(tree_dict.keys())
        for children in tree_dict.values():
            all_nodes.update(children)
        return all_nodes

    def find_roots(tree_dict):
        """
        Find all nodes that are not listed as children of any other node.
        These will be the roots of (possibly) multiple connected components.
        """
        all_nodes = find_all_nodes(tree_dict)
        children_set = set()
        for parent, children in tree_dict.items():
            children_set.update(children)
        # Roots = all_nodes that are never children
        roots = all_nodes - children_set
        return list(roots)

    def layout_tree(tree_dict):
        """
        Assign (x, y) coordinates to each node so that:
        - leaves are placed at distinct y-levels (from top to bottom),
        - each parent node is placed at the average y of its children,
        - x increases with depth from the root.
        
        Returns:
        positions: dict of node -> (x_coord, y_coord)
        all_nodes_in_layout: the list of nodes for reference (not sorted by y)
        
        Note: This is the same layout logic shown previously, but
        you are free to substitute your own layout method (BFS, etc.).
        """
        positions = {}
        next_leaf_y = [0]  # We'll increment this for each leaf encountered
        
        def dfs(node, x):
            """
            Depth-first search that assigns positions recursively.
            """
            if node not in tree_dict or len(tree_dict[node]) == 0:
                # It's a leaf
                y = next_leaf_y[0]
                positions[node] = (x, y)
                next_leaf_y[0] += 1
                return
            
            # Otherwise, recurse on children
            child_y_positions = []
            for child in tree_dict[node]:
                dfs(child, x + 1)
                child_y_positions.append(positions[child][1])
            
            # Place the current node at the average y of its children
            y = float(np.mean(child_y_positions))
            positions[node] = (x, y)
        
        roots = find_roots(tree_dict)
        roots.sort()  # for deterministic processing order
        for root in roots:
            dfs(root, x=0)
        
        all_nodes_in_layout = list(positions.keys())
        return positions, all_nodes_in_layout

    positions, all_nodes_in_layout = layout_tree(tree_dict)        
    # Draw edges from parent to child
    for parent, children in tree_dict.items():
        if parent in positions:
            x_parent, y_parent = positions[parent]
            for child in children:
                if child in positions:
                    x_child, y_child = positions[child]
                    ax_tree.plot([x_parent, x_child], [y_parent, y_child], 'k-')
    
    # Draw each node and label it
    for node, (x, y) in positions.items():
        ax_tree.plot(x, y, 'ko', ms=5)
        ax_tree.text(x, y, str(node), va='center', ha='right', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='none', pad=0.5))
    
    # Flip y-axis (so smaller y-values appear at the top visually)
    ax_tree.invert_yaxis()
    ax_tree.set_xticks([])
    ax_tree.set_yticks([])
    ax_tree.set_title("Tree Structure")
    return ax_tree

def point_2d_cloud_stat_to_ellipse(mu, eig_vec, eig_s, n_std=1):
    """
    Convert point cloud statistics (mean, eigenvectors, eigenvalues) into an ellipse representation.
    Parameters:
        mu: Mean vector (2D)
        eig_vec: Eigenvector matrix (2x2)
        eig_s: Eigenvalue vector (2D)
    """
    t = np.linspace(0.0, 2 * np.pi, 100)
    circ_xy = np.vstack((np.cos(t), np.sin(t)))
    ellip_xy = eig_vec @ np.diag(n_std * eig_s) @ circ_xy + mu[:, None]
    return ellip_xy

def vis_2d_ellipse(mu, eig_vec, eig_s, n_std=1, f=None, ax=None, 
                   label=None, color=None, alpha=1, s=None, 
                   xlabel=None, ylabel=None):
    """
    Visualize a 2D ellipse based on point cloud statistics.
    Parameters:
        mu: Mean vector (2D)
        eig_vec: Eigenvector matrix (2x2)
        eig_s: Eigenvalue vector (2D)
        n_std: Number of standard deviations to draw (default is 1)
        ax: Matplotlib axis to draw on (if None, a new figure is created)
    """
    ellip_xy = point_2d_cloud_stat_to_ellipse(mu, eig_vec, eig_s, n_std=n_std)
    if ax is None:
        f, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(mu[0], mu[1], s=s, label=label, color=color, alpha=alpha)
    ax.plot(ellip_xy[0], ellip_xy[1], alpha=alpha, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return f, ax