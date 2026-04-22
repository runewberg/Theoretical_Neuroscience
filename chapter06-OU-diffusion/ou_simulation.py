import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk

# ── simulation state ──────────────────────────────────────────────────────────

DT         = 0.03
MAX_TRAIL  = 120
N_TRAILS   = 40       # trajectories to draw
XMIN, XMAX = -6, 6
BINS       = 40

class OUSimulation:
    def __init__(self, N=300, lam=0.5, sig=1.0, mu=0.0):
        self.N   = N
        self.lam = lam
        self.sig = sig
        self.mu  = mu
        self.reset()

    def reset(self):
        self.positions = np.random.randn(self.N) * 0.3
        n_tr = min(self.N, N_TRAILS)
        self.trails = [list([self.positions[i]]) for i in range(n_tr)]
        self.t = 0.0

    def step(self):
        dx = (-self.lam * (self.positions - self.mu) * DT
              + self.sig * np.sqrt(DT) * np.random.randn(self.N))
        self.positions += dx
        self.t += DT
        for i, trail in enumerate(self.trails):
            trail.append(self.positions[i])
            if len(trail) > MAX_TRAIL:
                trail.pop(0)

    @property
    def theory_std(self):
        return self.sig / np.sqrt(2 * self.lam)

    @property
    def empirical_mean(self):
        return float(np.mean(self.positions))

    @property
    def empirical_std(self):
        return float(np.std(self.positions))


# ── GUI ───────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, root):
        self.root    = root
        self.root.title("Ornstein–Uhlenbeck particle simulation")
        self.running = True
        self.sim     = OUSimulation()

        self._build_controls()
        self._build_figure()
        self._start_animation()

    # ── controls ──────────────────────────────────────────────────────────────

    def _build_controls(self):
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        self.vars = {}
        sliders = [
            ("λ  (leak)",    "lam",   0.05, 3.0,  0.05, 0.5),
            ("σ  (noise)",   "sig",   0.1,  3.0,  0.1,  1.0),
            ("μ  (mean)",    "mu",   -3.0,  3.0,  0.1,  0.0),
            ("N  particles", "npart", 50,   800,  50,   300),
        ]

        for label, key, lo, hi, step, val in sliders:
            ttk.Label(ctrl, text=label, width=14).pack(anchor="w")
            var = tk.DoubleVar(value=val)
            self.vars[key] = var
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=2)
            sl  = ttk.Scale(row, from_=lo, to=hi, variable=var,
                            orient=tk.HORIZONTAL, length=180,
                            command=lambda v, k=key: self._on_slider(k))
            sl.pack(side=tk.LEFT)
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
        self.stat_labels = {}
        for stat in ("empirical mean", "empirical std", "theory std  σ/√2λ"):
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=stat, width=18, foreground="gray").pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="—", width=7, anchor="e")
            lbl.pack(side=tk.LEFT)
            self.stat_labels[stat] = lbl

    def _on_slider(self, key):
        val = self.vars[key].get()
        fmt = ".0f" if key == "npart" else ".2f"
        getattr(self, f"lbl_{key}").config(text=f"{val:{fmt}}")
        self.sim.lam = self.vars["lam"].get()
        self.sim.sig = self.vars["sig"].get()
        self.sim.mu  = self.vars["mu"].get()
        new_N = int(self.vars["npart"].get())
        if new_N != self.sim.N:
            self.sim.N = new_N
            self._reset()

    def _reset(self):
        self.sim.lam = self.vars["lam"].get()
        self.sim.sig = self.vars["sig"].get()
        self.sim.mu  = self.vars["mu"].get()
        self.sim.N   = int(self.vars["npart"].get())
        self.sim.reset()

    def _toggle_pause(self):
        self.running = not self.running
        self.pause_btn.config(text="Resume" if not self.running else "Pause")

    # ── figure ────────────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.Figure(figsize=(10, 5), tight_layout=True)
        gs  = gridspec.GridSpec(1, 2, figure=self.fig)
        self.ax_traj = self.fig.add_subplot(gs[0])
        self.ax_hist = self.fig.add_subplot(gs[1])

        self.ax_traj.set_xlim(0, MAX_TRAIL * DT)
        self.ax_traj.set_ylim(XMIN, XMAX)
        self.ax_traj.set_xlabel("time")
        self.ax_traj.set_ylabel("x")
        self.ax_traj.set_title("particle trajectories")

        self.ax_hist.set_xlim(XMIN, XMAX)
        self.ax_hist.set_xlabel("x")
        self.ax_hist.set_ylabel("density")
        self.ax_hist.set_title("position distribution")

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

        self._draw_trajectories()
        self._draw_histogram()
        self._update_stats()
        self.canvas.draw_idle()

    def _draw_trajectories(self):
        ax = self.ax_traj
        ax.cla()
        ax.set_xlim(0, MAX_TRAIL * DT)
        ax.set_ylim(XMIN, XMAX)
        ax.set_xlabel("time")
        ax.set_ylabel("x")
        ax.set_title("particle trajectories")

        mu = self.sim.mu
        ax.axhline(0,  color="lightgray", lw=0.8, ls="-")
        ax.axhline(mu, color="#1D9E75",   lw=1.2, ls="--", label=f"μ = {mu:.1f}")

        n_tr = len(self.sim.trails)
        tvals = np.linspace(0, MAX_TRAIL * DT, MAX_TRAIL)

        for i, trail in enumerate(self.sim.trails):
            tr = np.array(trail)
            tv = tvals[-len(tr):]
            ax.plot(tv, tr, color="#185FA5", alpha=0.3, lw=0.8)
            ax.plot(tv[-1], tr[-1], "o", color="#185FA5", ms=2.5, alpha=0.8)

        ax.legend(fontsize=9, loc="upper right")

    def _draw_histogram(self):
        ax = self.ax_hist
        ax.cla()
        ax.set_xlim(XMIN, XMAX)
        ax.set_xlabel("x")
        ax.set_ylabel("density")
        ax.set_title("position distribution")

        pos = self.sim.positions
        ax.hist(pos, bins=BINS, range=(XMIN, XMAX),
                density=True, color="#185FA5", alpha=0.45, label="empirical")

        xs   = np.linspace(XMIN, XMAX, 300)
        tstd = self.sim.theory_std
        gauss = (np.exp(-0.5 * ((xs - self.sim.mu) / tstd) ** 2)
                 / (tstd * np.sqrt(2 * np.pi)))
        ax.plot(xs, gauss, color="#D85A30", lw=2, label="theory")
        ax.legend(fontsize=9)

    def _update_stats(self):
        self.stat_labels["empirical mean"].config(
            text=f"{self.sim.empirical_mean:+.3f}")
        self.stat_labels["empirical std"].config(
            text=f"{self.sim.empirical_std:.3f}")
        self.stat_labels["theory std  σ/√2λ"].config(
            text=f"{self.sim.theory_std:.3f}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x520")
    app  = App(root)
    root.mainloop()
