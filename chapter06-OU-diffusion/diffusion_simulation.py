import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk

# ── simulation parameters ─────────────────────────────────────────────────────

DT         = 0.03
MAX_TRAIL  = 150
N_TRAILS   = 40
XMIN, XMAX = -12, 12
BINS       = 50

# ── simulation state ──────────────────────────────────────────────────────────

class DiffusionSimulation:
    def __init__(self, N=300, D=1.0, v=0.0):
        self.N  = N
        self.D  = D       # diffusion coefficient
        self.v  = v       # drift velocity
        self.reset()

    def reset(self):
        self.positions = np.zeros(self.N)   # all start at origin
        n_tr = min(self.N, N_TRAILS)
        self.trails = [list([0.0]) for _ in range(n_tr)]
        self.t = 0.0

    def step(self):
        dx = self.v * DT + np.sqrt(2 * self.D * DT) * np.random.randn(self.N)
        self.positions += dx
        self.t += DT
        for i, trail in enumerate(self.trails):
            trail.append(float(self.positions[i]))
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

    # ── analytical solutions ──────────────────────────────────────────────────

    def theory_mean(self):
        return self.v * self.t

    def theory_std(self):
        return np.sqrt(2 * self.D * self.t) if self.t > 0 else 0.0

    def theory_pdf(self, xs):
        if self.t <= 0:
            return np.zeros_like(xs)
        std = self.theory_std()
        mu  = self.theory_mean()
        return (np.exp(-0.5 * ((xs - mu) / std) ** 2)
                / (std * np.sqrt(2 * np.pi)))

    @property
    def empirical_mean(self):
        return float(np.mean(self.positions))

    @property
    def empirical_std(self):
        return float(np.std(self.positions))

    @property
    def msd(self):
        return float(np.mean(self.positions ** 2))


# ── GUI ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root    = root
        self.root.title("Brownian diffusion with drift")
        self.running = True
        self.sim     = DiffusionSimulation()

        self._build_controls()
        self._build_figure()
        self._start_animation()

    # ── controls ──────────────────────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        self.vars = {}
        sliders = [
            ("D  (diffusion)", "D",     0.05, 4.0,  0.05, 1.0),
            ("v  (drift)",     "v",    -3.0,  3.0,  0.1,  0.0),
            ("N  particles",   "npart", 50,   800,  50,   300),
        ]

        for label, key, lo, hi, step, val in sliders:
            ttk.Label(ctrl, text=label, width=16).pack(anchor="w")
            var = tk.DoubleVar(value=val)
            self.vars[key] = var
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=2)
            ttk.Scale(row, from_=lo, to=hi, variable=var,
                      orient=tk.HORIZONTAL, length=190,
                      command=lambda val, k=key: self._on_slider(k)).pack(side=tk.LEFT)
            lbl = ttk.Label(row, width=6, text=f"{val:.2f}")
            lbl.pack(side=tk.LEFT, padx=4)
            setattr(self, f"lbl_{key}", lbl)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack()
        ttk.Button(btn_frame, text="Reset",
                   command=self._reset).pack(side=tk.LEFT, padx=4)
        self.pause_btn = ttk.Button(btn_frame, text="Pause",
                                    command=self._toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=4)

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(ctrl, text="statistics", font=("", 10, "bold")).pack(anchor="w")
        stats = [
            ("time  t",          "t"),
            ("empirical mean",   "emean"),
            ("theory mean  vt",  "tmean"),
            ("empirical std",    "estd"),
            ("theory std  √2Dt", "tstd"),
            ("MSD  ⟨x²⟩",        "msd"),
        ]
        self.stat_labels = {}
        for label, key in stats:
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, width=18,
                      foreground="gray").pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="—", width=8, anchor="e")
            lbl.pack(side=tk.LEFT)
            self.stat_labels[key] = lbl

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        note = ("dx = v·dt + √(2D·dt)·ξ\n"
                "ξ ~ N(0,1)\n\n"
                "Variance grows as 2Dt.\n"
                "Drift shifts the mean by vt.")
        ttk.Label(ctrl, text=note, foreground="gray",
                  justify=tk.LEFT, font=("Courier", 9)).pack(anchor="w")

    def _on_slider(self, key):
        val = self.vars[key].get()
        fmt = ".0f" if key == "npart" else ".2f"
        getattr(self, f"lbl_{key}").config(text=f"{val:{fmt}}")
        self.sim.D = self.vars["D"].get()
        self.sim.v = self.vars["v"].get()
        new_N = int(self.vars["npart"].get())
        if new_N != self.sim.N:
            self.sim.N = new_N
            self._reset()

    def _reset(self):
        self.sim.D = self.vars["D"].get()
        self.sim.v = self.vars["v"].get()
        self.sim.N = int(self.vars["npart"].get())
        self.sim.reset()

    def _toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    # ── figure ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.Figure(figsize=(11, 8), tight_layout=True)
        gs = gridspec.GridSpec(2, 2, figure=self.fig,
                               height_ratios=[1.6, 1])

        self.ax_traj = self.fig.add_subplot(gs[0, 0])
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.ax_msd  = self.fig.add_subplot(gs[1, 0])
        self.ax_mean = self.fig.add_subplot(gs[1, 1])

        self.msd_times  = []
        self.msd_vals   = []
        self.mean_times = []
        self.mean_vals  = []

        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.canvas = canvas

    # ── animation ─────────────────────────────────────────────────────────────

    def _start_animation(self):
        self.anim = FuncAnimation(
            self.fig, self._update,
            interval=30, blit=False, cache_frame_data=False
        )
        self.canvas.draw()

    def _update(self, frame):
        if not self.running:
            return
        for _ in range(2):
            self.sim.step()

        self.msd_times.append(self.sim.t)
        self.msd_vals.append(self.sim.msd)
        self.mean_times.append(self.sim.t)
        self.mean_vals.append(self.sim.empirical_mean)

        self._draw_trajectories()
        self._draw_histogram()
        self._draw_msd()
        self._draw_mean()
        self._update_stats()
        self.canvas.draw_idle()

    # ── panels ────────────────────────────────────────────────────────────────

    def _draw_trajectories(self):
        ax = self.ax_traj
        ax.cla()
        t_window = MAX_TRAIL * DT
        xlim_lo  = max(0, self.sim.t - t_window)
        ax.set_xlim(xlim_lo, xlim_lo + t_window)

        mu = self.sim.theory_mean()
        std = self.sim.theory_std()
        dyn_lo = min(XMIN, mu - 3.5 * std)
        dyn_hi = max(XMAX, mu + 3.5 * std)
        ax.set_ylim(dyn_lo, dyn_hi)
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.set_title("particle trajectories")

        ax.axhline(0, color="lightgray", lw=0.8)

        if std > 0:
            ax.fill_between(
                [xlim_lo, xlim_lo + t_window],
                [mu - std, mu - std], [mu + std, mu + std],
                color="#185FA5", alpha=0.07, label="±1σ theory"
            )

        tvals = np.linspace(xlim_lo, xlim_lo + t_window, MAX_TRAIL)
        for trail in self.sim.trails:
            tr = np.array(trail)
            tv = tvals[-len(tr):]
            ax.plot(tv, tr, color="#185FA5", alpha=0.3, lw=0.8)
            ax.plot(tv[-1], tr[-1], "o", color="#185FA5", ms=2.5, alpha=0.8)

        theo_x = np.array([xlim_lo, xlim_lo + t_window])
        ax.plot(theo_x,
                self.sim.v * theo_x,
                color="#D85A30", lw=1.5, ls="--", label=f"drift vt")
        ax.legend(fontsize=8, loc="upper left")

    def _draw_histogram(self):
        ax = self.ax_hist
        ax.cla()

        mu  = self.sim.theory_mean()
        std = self.sim.theory_std()
        lo  = min(XMIN, mu - 4 * std) if std > 0 else XMIN
        hi  = max(XMAX, mu + 4 * std) if std > 0 else XMAX
        ax.set_xlim(lo, hi)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title(f"position distribution  (t = {self.sim.t:.1f})")

        ax.hist(self.sim.positions, bins=BINS, range=(lo, hi),
                density=True, color="#185FA5", alpha=0.45, label="empirical")

        if self.sim.t > 0.05:
            xs    = np.linspace(lo, hi, 400)
            gauss = self.sim.theory_pdf(xs)
            ax.plot(xs, gauss, color="#D85A30", lw=2, label="theory")
            ax.axvline(mu, color="#1D9E75", lw=1.5, ls="--",
                       label=f"⟨x⟩ = {mu:.2f}")
        ax.legend(fontsize=8)

    def _draw_msd(self):
        ax = self.ax_msd
        ax.cla()
        ax.set_xlabel("time t")
        ax.set_ylabel("MSD  ⟨x²⟩")
        ax.set_title("mean squared displacement")

        if len(self.msd_times) > 1:
            ts = np.array(self.msd_times)
            ax.plot(ts, self.msd_vals, color="#185FA5",
                    lw=1.2, alpha=0.8, label="empirical MSD")
            theory_msd = 2 * self.sim.D * ts + (self.sim.v * ts) ** 2
            ax.plot(ts, theory_msd, color="#D85A30",
                    lw=2, ls="--", label="2Dt + (vt)²")
            ax.legend(fontsize=8)

    def _draw_mean(self):
        ax = self.ax_mean
        ax.cla()
        ax.set_xlabel("time t")
        ax.set_ylabel("mean position")
        ax.set_title("mean position vs time")

        if len(self.mean_times) > 1:
            ts = np.array(self.mean_times)
            ax.plot(ts, self.mean_vals, color="#185FA5",
                    lw=1.2, alpha=0.8, label="empirical mean")
            ax.plot(ts, self.sim.v * ts, color="#D85A30",
                    lw=2, ls="--", label=f"theory: vt  (v={self.sim.v:.1f})")
            ax.axhline(0, color="lightgray", lw=0.8)
            ax.legend(fontsize=8)

    def _update_stats(self):
        s = self.sim
        self.stat_labels["t"].config(text=f"{s.t:.2f}")
        self.stat_labels["emean"].config(text=f"{s.empirical_mean:+.3f}")
        self.stat_labels["tmean"].config(text=f"{s.theory_mean():+.3f}")
        self.stat_labels["estd"].config(text=f"{s.empirical_std:.3f}")
        self.stat_labels["tstd"].config(text=f"{s.theory_std():.3f}")
        self.stat_labels["msd"].config(text=f"{s.msd:.3f}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x620")
    app  = App(root)
    root.mainloop()
