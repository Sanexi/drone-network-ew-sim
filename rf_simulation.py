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
                 is_susceptible: bool = True, nf_db: float = 10.0, 
                 env_noise_db: float = 5.0):

        self.tx, self.rx = tx, rx
        self.freq_hz = freq_hz
        self.bw_hz   = bw_hz
        self.p_tx_dbm = 10*math.log10(p_tx_w*1e3)
        self.rx_thresh_dbm = rx_thresh_dbm
        self.is_susceptible = is_susceptible
        self.capacity_bps: float | None = None
        self.nf_db        = nf_db
        self.env_noise_db = env_noise_db

    def update(self, net: Network2D) -> None:
        """
         Re-compute Shannon capacity **unless** the link is tagged
         non-susceptible *and* we already cached a value.
         """
        if (not self.is_susceptible) and (self.capacity_bps is not None):
            return

        # ---------- signal -------------------------------------------------
        d = np.hypot(*(self.tx.xy - self.rx.xy))
        p_rx_dbm = self.p_tx_dbm - fspl_db(self.freq_hz, d)

        # ---------- noise --------------------------------------------------
        N_therm = -174 + 10*math.log10(self.bw_hz)             # kTB
        N_sys   = N_therm + self.nf_db + self.env_noise_db     # radio + env.

        I_ext = self.rx.interference(self.freq_hz,
                                     self.bw_hz, net) if self.is_susceptible else -200.0

        N_tot_lin = 10**(N_sys/10) + 10**(I_ext/10)            # mW
        N_tot_dbm = 10*math.log10(N_tot_lin)

        # ---------- Shannon capacity --------------------------------------
        snr_lin = 10**((p_rx_dbm - N_tot_dbm)/10)
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
    
    def plot_snapshot(self,
                  *,
                  pad_frac: float = 0.15,
                  ax: plt.Axes | None = None) -> None:
        """
        Visual sanity-check of the RF-world swarm geometry.

        *   Coloured arrows from **tx → rx**, annotated with the
            current Shannon capacity (Mb s⁻¹) stored in `Link2D.capacity_bps`.
        *   Drone IDs next to every marker.
        *   Axes automatically scaled so the whole swarm plus `pad_frac`
            margin fits the view.
        """

        # --------------------------------------------------------- figure/axes
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 8))

        # --------------------------------------------------- swarm bounding box
        xs = np.array([d.xy[0] for d in self.drones])
        ys = np.array([d.xy[1] for d in self.drones])

        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
        width = xmax - xmin
        height = ymax - ymin
        pad = pad_frac * max(width, height)

        xmin -= pad; xmax += pad
        ymin -= pad; ymax += pad

        # ----------------------------------------------------------- draw links
        palette = plt.colormaps["tab10"]
        for lk in self.links:
            band_idx = int(round(lk.freq_hz / 1e6)) % 10           # simple hash
            col = palette(band_idx)

            ax.annotate(
                "", xy=lk.rx.xy, xytext=lk.tx.xy,
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color=col),
                zorder=2,
            )

            # capacity label (Mb/s) at arrow mid-point
            if lk.capacity_bps is not None:
                mid = (lk.tx.xy + lk.rx.xy) / 2
                ax.text(mid[0], mid[1],
                        f"{lk.capacity_bps/1e6:.2f} Mb/s",
                        fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

        # --------------------------------------------------------- draw drones
        label_offset = 0.015 * max(width, height)
        for d in self.drones:
            ax.scatter(*d.xy,
                    s=80 if d is self.master else 50,
                    c="white" if d is self.master else "black",
                    edgecolors="k", zorder=3)
            ax.text(d.xy[0] + label_offset,
                    d.xy[1] + label_offset,
                    str(d.id),
                    fontsize=9, ha="left", va="bottom")

        # --------------------------------------------------------- cosmetics
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x [m]");   ax.set_ylabel("y [m]")
        ax.set_title("RF-world swarm snapshot")
        ax.set_aspect("equal", adjustable="box")

        # simple legend placed outside plot
        link_h = plt.Line2D([0], [0], lw=2.2, color=palette(0),
                            marker=r'$\rightarrow$', label="Tx→Rx link")
        mast_h = plt.Line2D([0], [0], marker="o", color="white",
                            markeredgecolor="k", label="Master")
        drn_h  = plt.Line2D([0], [0], marker="o", color="black",
                            markeredgecolor="k", label="Drone")
        ax.legend(handles=[link_h, mast_h, drn_h],
                loc="upper left", bbox_to_anchor=(1.02, 1.0))

        plt.show()
    
