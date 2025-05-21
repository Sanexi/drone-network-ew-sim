from __future__ import annotations
import math, numpy as np, matplotlib.pyplot as plt
from typing import List, Sequence, Tuple
from matplotlib import colormaps


# ────────────────────────────────────────────────────────────────────
# 0.  Helper functions
# ────────────────────────────────────────────────────────────────────
def fspl_db(freq_hz: float, dist_m: np.ndarray) -> np.ndarray:
    """
    Free-space loss in dB.

        FSPL[dB] = 32.44 + 20·log10(d[km]) + 20·log10(f[MHz])

    Everything coming in is **Hz** → convert inside.
    """
    freq_mhz = freq_hz / 1e6
    dist_km  = np.maximum(dist_m, 1e-3) / 1000.0          # avoid log(0)
    return 32.44 + 20*np.log10(dist_km) + 20*np.log10(freq_mhz)


def build_subbands(f_min_hz: float = 433e6,
                   f_max_hz: float = 1200e6,
                   n_bands:  int   = 4):
    """
    Split [f_min_hz, f_max_hz] into `n_bands` equal slices.

    Returns
    -------
    centres_hz : list[float]   – mid-frequency of each band (Hz)
    band_bw_hz : float         – width of every band (Hz)
    """
    edges   = np.linspace(f_min_hz, f_max_hz, n_bands + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres.tolist(), (edges[1] - edges[0])


# ────────────────────────────────────────────────────────────────────
# 1.  World2D – analysis grid
# ────────────────────────────────────────────────────────────────────
class World2D:
    def __init__(self,
                 x_rng: Tuple[float, float],
                 y_rng: Tuple[float, float],
                 nx: int = 200, ny: int = 200):

        self.xs = np.linspace(*x_rng, nx)
        self.ys = np.linspace(*y_rng, ny)
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.nx, self.ny = nx, ny

# ────────────────────────────────────────────────────────────────────
# 2.  Static transmitter
# ────────────────────────────────────────────────────────────────────
class Tx2D:
    def __init__(self,
                 world: World2D,
                 xy_idx: Tuple[int, int],          # grid indices
                 freq_hz: float,                   # **Hz**
                 p_tx_w:  float,                   # total RF power (W)
                 bw_hz:   float,                   # occupied BW (Hz)
                 fspl_thresh_dbm: float = -110):

        self.w = world
        ix, iy = xy_idx
        self.x, self.y = world.xs[ix], world.ys[iy]

        self.freq_hz = freq_hz          # centre freq  (Hz)
        self.bw_hz   = bw_hz            # signal BW    (Hz)
        self.tx_power_dbm = 10*math.log10(p_tx_w*1e3)

        # pre-compute received-power map
        r = np.hypot(world.X - self.x, world.Y - self.y)
        self.power = self.tx_power_dbm - fspl_db(freq_hz, r)      # dBm
        self.cover = self.power >= fspl_thresh_dbm


# # ────────────────────────────────────────────────────────────────────
# # 3.  Network2D : collection of static transmitters + utilities
# # ────────────────────────────────────────────────────────────────────
# class Network2D:
#     def __init__(self, txs: Sequence[Tx2D]):
#         self.txs = list(txs)
#         tab10    = colormaps['tab10']
#         self.tx_colors = [tab10(i % 10) for i, _ in enumerate(txs)]

#     # ...............................................................
#     # Received power *within a receiver band* at a single point
#     # ...............................................................
#     def power_in_band(self, x: float, y: float,
#                       f_rx: float, bw_hz: float) -> float:
#         """
#         Sum received power (dBm) from **all** Tx whose spectra overlap the
#         Rx band [f_rx ± bw_hz/2] at map point (x,y).

#         Each Tx contribution is scaled by (spectral_overlap / tx.bw_hz),
#         assuming a uniform power spectral density.
#         """
#         p_total_lin = 0.0  # linear mW accumulator

#         for tx in self.txs:
#             # --- 1. overlap between Rx and this Tx -----------------
#             fmin_rx, fmax_rx = f_rx - bw_hz/2,  f_rx + bw_hz/2
#             fmin_tx = tx.freq_mhz*1e6 - tx.bw_hz/2
#             fmax_tx = tx.freq_mhz*1e6 + tx.bw_hz/2

#             overlap_hz = max(0.0, min(fmax_tx, fmax_rx) - max(fmin_tx, fmin_rx))
#             if overlap_hz == 0.0:
#                 continue                               # skip if no spectral overlap

#             frac = overlap_hz / tx.bw_hz               # fraction of Tx power seen

#             # --- 2. bilinear interpolation of Tx power map --------
#             ix = np.interp(x, tx.w.xs, np.arange(tx.w.nx))
#             iy = np.interp(y, tx.w.ys, np.arange(tx.w.ny))
#             ix0, iy0 = int(ix), int(iy)                # cell origin

#             if not (0 <= ix0 < tx.w.nx-1 and 0 <= iy0 < tx.w.ny-1):
#                 continue                               # outside grid

#             wx, wy = ix - ix0, iy - iy0                # bilinear weights
#             p_dBm = (
#                 (1-wx)*(1-wy)*tx.power[ix0    , iy0    ] +
#                  wx   *(1-wy)*tx.power[ix0+1  , iy0    ] +
#                 (1-wx)* wy   *tx.power[ix0    , iy0 + 1] +
#                  wx   * wy   *tx.power[ix0+1  , iy0 + 1]
#             )

#             # --- 3. accumulate in linear domain -------------------
#             p_total_lin += frac * 10**(p_dBm / 10)     # convert dBm → mW

#         return -200.0 if p_total_lin == 0 else 10*math.log10(p_total_lin)

#     # ...............................................................
#     # Power-field plot – 4 percentile contours per Tx
#     # ...............................................................
#     def plot_power(self, ax: plt.Axes, *, linewidth: float = 1.8):
#         """
#         Draw exactly four contour lines (20-,40-,60-,80-th percentiles)
#         for each transmitter so that rings are always visible.
#         """
#         for tx, col in zip(self.txs, self.tx_colors):
#             levels = np.quantile(tx.power, [0.2, 0.4, 0.6, 0.8])  # ascending
#             ax.contour(tx.w.X, tx.w.Y, tx.power,
#                        levels=levels, colors=[col], linewidths=linewidth)
#             ax.scatter(tx.x, tx.y, marker="X", s=80,
#                        color=col, edgecolors="k", zorder=10)

#         ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")

# ────────────────────────────────────────────────────────────────────
# 3.  Network2D – many static Tx
# ────────────────────────────────────────────────────────────────────
class Network2D:
    def __init__(self, txs: Sequence[Tx2D]):
        self.txs = list(txs)
        tab10   = colormaps['tab10']
        self.tx_colors = [tab10(i % 10) for i, _ in enumerate(txs)]

    # ...............................................................
    def power_in_band(self, x: float, y: float,
                      f_rx_hz: float, bw_hz: float) -> float:
        """
        Aggregate received power (dBm) within Rx band [f_rx_hz ± bw_hz/2].
        """
        p_lin = 0.0  # mW
        for tx in self.txs:
            # --- spectral overlap ---------------------------------
            f_min_rx, f_max_rx = f_rx_hz - bw_hz/2, f_rx_hz + bw_hz/2
            f_min_tx = tx.freq_hz - tx.bw_hz/2
            f_max_tx = tx.freq_hz + tx.bw_hz/2
            overlap = max(0.0, min(f_max_tx, f_max_rx) - max(f_min_tx, f_min_rx))
            if overlap == 0:                                     # no overlap
                continue
            frac = overlap / tx.bw_hz                            # PSD fraction

            # --- bilinear interpolation of tx.power --------------
            ix = np.interp(x, tx.w.xs, np.arange(tx.w.nx))
            iy = np.interp(y, tx.w.ys, np.arange(tx.w.ny))
            ix0, iy0 = int(ix), int(iy)
            if not (0 <= ix0 < tx.w.nx-1 and 0 <= iy0 < tx.w.ny-1):
                continue

            wx, wy = ix - ix0, iy - iy0
            p_dbm = ((1-wx)*(1-wy)*tx.power[ix0  , iy0  ] +
                      wx   *(1-wy)*tx.power[ix0+1, iy0  ] +
                     (1-wx)* wy   *tx.power[ix0  , iy0+1] +
                      wx   * wy   *tx.power[ix0+1, iy0+1])

            p_lin += frac * 10**(p_dbm/10)

        return -200.0 if p_lin == 0 else 10*math.log10(p_lin)

    # ...............................................................
    def plot_power(self, ax: plt.Axes, *, linewidth: float = 1.8):
        for tx, col in zip(self.txs, self.tx_colors):
            levels = np.quantile(tx.power, [0.2, 0.4, 0.6, 0.8])
            ax.contour(tx.w.X, tx.w.Y, tx.power,
                       levels=levels, colors=[col], linewidths=linewidth)
            ax.scatter(tx.x, tx.y, marker="X", s=80,
                       color=col, edgecolors="k", zorder=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")



# ────────────────────────────────────────────────────────────────────
# 4.  Drones, Links, Swarm (2-D geometry)
# ────────────────────────────────────────────────────────────────────
class Drone2D:
    def __init__(self, drone_id: int, rel_xy: np.ndarray):
        self.id = drone_id
        self.rel_xy = rel_xy.astype(float)            # relative to master
        self.xy: np.ndarray | None = None             # world coords

    def interference(self, f_center: float, bw_hz: float, net: Network2D) -> float:
        """Helper used by Link2D – external noise at this drone."""
        return net.power_in_band(self.xy[0], self.xy[1], f_center, bw_hz)


class Link2D:
    """
    Point-to-point link inside the swarm.  Computes Shannon capacity
    each time `update()` is called.
    """
    def __init__(self,
                 tx: Drone2D, rx: Drone2D,
                 *, freq_hz: float, p_tx_w: float,
                 bw_hz: float, rx_thresh_dbm: float = -120,
                 is_susceptible: bool = True):

        self.tx, self.rx = tx, rx
        self.freq_hz = freq_hz
        self.bw_hz   = bw_hz
        self.p_tx_dbm = 10*math.log10(p_tx_w*1e3)
        self.rx_thresh_dbm = rx_thresh_dbm
        self.is_susceptible = is_susceptible
        self.capacity_bps: float | None = None

    def update(self, net: Network2D) -> None:
        """
        Re-compute Shannon capacity **unless** the link is tagged
        non-susceptible *and* we already cached a value.
        """
        if (not self.is_susceptible) and (self.capacity_bps is not None):
            return                                    # nothing changes

        # signal
        d = np.hypot(*(self.tx.xy - self.rx.xy))
        p_rx_dbm = self.p_tx_dbm - fspl_db(self.freq_hz, d)

        # noise
        N_th = -174 + 10*math.log10(self.bw_hz)          # thermal
        I_ext = self.rx.interference(self.freq_hz, self.bw_hz, net) \
                if self.is_susceptible else -200.0
        N_tot = 10*math.log10(10**(N_th/10) + 10**(I_ext/10))

        snr_lin = 10**((p_rx_dbm - N_tot)/10)
        self.capacity_bps = self.bw_hz * math.log2(1 + snr_lin)


class Swarm2D:
    """
    Holds all drones & links.  
    `update()` = translate swarm → recompute capacities → return snapshot.
    """
    def __init__(self,
                 master: Drone2D,
                 drones: Sequence[Drone2D],
                 links:  Sequence[Link2D],
                 *, scale: float = 1.0):

        self.master  = master
        self.drones  = list(drones)
        self.links   = list(links)
        self.scale   = scale

    # ------------------------------------------------------------------
    def update(self, master_xy: np.ndarray, net: Network2D) -> dict[tuple, dict]:
        # 1 Move whole formation
        self.master.xy = master_xy
        for d in self.drones:
            d.xy = master_xy + d.rel_xy * self.scale

        # 2 Re-evaluate links
        for lk in self.links:
            lk.update(net)

        # 3 Snapshot for the higher-level simulator
        snap = {}
        for lk in self.links:
            key = tuple(sorted((lk.tx.id, lk.rx.id)))
            snap[key] = {
                'is_susceptible': lk.is_susceptible,
                'current_max_capacity_after_ew': lk.capacity_bps,
            }
        return snap


# ────────────────────────────────────────────────────────────────────
# 5.  Demo – one tower + 8-drone swarm
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) world --------------------------------------------------------
    w = World2D((0, 1000), (0, 1000))

    # 2) static transmitter occupying 1st sub-band -------------------
    centres, sub_bw_mhz = build_subbands()
    tx_static = Tx2D(
        w, (60, 120),
        freq_mhz = centres[0],
        p_tx_w   = 10.0,                          # 10 W total RF power
        bw_hz    = sub_bw_mhz * 1e6,              # full ¼-band (~192 kHz)
        fspl_thresh_dbm = -200                    # plot everything
    )
    net = Network2D([tx_static])

    # 3) swarm geometry ----------------------------------------------
    N_DRONES = 8
    angles   = np.linspace(0, 2*np.pi, N_DRONES, endpoint=False)
    master   = Drone2D(0, np.zeros(2))
    drones   = [master] + [
        Drone2D(i+1, np.array([np.cos(a), np.sin(a)]))
        for i, a in enumerate(angles)
    ]

    # 4) links: round-robin sub-band allocation ----------------------
    link_bw_hz = 0.15e6
    links = [
        Link2D(master, d,
               freq_mhz = centres[(i-1) % 4],
               p_tx_dbm = 20,
               bw_hz    = link_bw_hz)
        for i, d in enumerate(drones[1:], 1)
    ]

    swarm = Swarm2D(master, drones, links, scale=20)
    swarm.move(np.array([600, 300]))
    swarm.evaluate(net)

    # 5) plotting -----------------------------------------------------
    fig, (ax_map, ax_local) = plt.subplots(1, 2, figsize=(13, 5))

    # full-map: four-ring field-lines
    net.plot_power(ax_map)
    ax_map.set_title("Static tower – power field")
    for d in drones:
        ax_map.scatter(*d.xy, marker="o",
                       c="w" if d is master else "k", edgecolors="k")

    # zoomed local view with capacity shading
    REF_CAP = 5e6
    palette = colormaps['tab10']
    for lk in links:
        colour = palette(centres.index(lk.freq_mhz) % 10)
        alpha  = np.clip(lk.capacity_bps / REF_CAP, 0, 1)
        ax_local.plot([lk.tx.xy[0], lk.rx.xy[0]],
                      [lk.tx.xy[1], lk.rx.xy[1]],
                      c=(*colour[:3], alpha), lw=3)
        midpoint = (lk.tx.xy + lk.rx.xy) / 2
        ax_local.text(midpoint[0], midpoint[1],
                      f"{lk.capacity_bps/1e6:.2f} Mb/s",
                      ha="center", va="center", fontsize=8)

    for d in drones:
        ax_local.scatter(*d.xy, marker="o",
                         c="w" if d is master else "k", edgecolors="k")
    ax_local.set_xlim(master.xy[0]-60, master.xy[0]+60)
    ax_local.set_ylim(master.xy[1]-60, master.xy[1]+60)
    ax_local.set_title("Swarm local view (capacity shading)")
    ax_local.set_aspect("equal")

    plt.tight_layout(); plt.show()