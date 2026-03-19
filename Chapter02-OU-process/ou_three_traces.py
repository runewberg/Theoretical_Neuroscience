import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================
dt    = 0.1       # ms
T     = 2000.0    # ms
t     = np.arange(0, T, dt)
N     = len(t)

V_rest = -70.0    # resting potential (mV)
sigma  = 2.0      # noise amplitude
C_m    = 1.0      # membrane capacitance (uF/cm^2)

# Three different time constants (ms)
taus    = [5.0, 20.0, 100.0]
colors  = ['steelblue', 'tomato', 'seagreen']
labels  = [f'τ = {tau} ms' for tau in taus]

# ============================================================
# Simulate OU process for each tau
# C dV/dt = -G_leak(V - V_rest) + sigma * noise
# where G_leak = C / tau
# ============================================================
np.random.seed(42)
traces = []
for tau in taus:
    G_leak = C_m / tau
    V      = np.zeros(N)
    V[0]   = V_rest
    for k in range(1, N):
        dV   = (dt / C_m) * (-G_leak * (V[k-1] - V_rest)
                              + sigma * np.random.randn() / np.sqrt(dt))
        V[k] = V[k-1] + dV
    traces.append(V)

# ============================================================
# Analytical steady-state std for each tau
# std = sigma * sqrt(tau / (2 * C_m))
# ============================================================
stds = [sigma * np.sqrt(tau / (2 * C_m)) for tau in taus]

# ============================================================
# Plotting
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

for i, (V, tau, color, label, std) in enumerate(zip(traces, taus, colors, labels, stds)):
    axes[i].plot(t, V, color=color, linewidth=0.7, alpha=0.9)
    axes[i].axhline(V_rest, color='black', linestyle='--', linewidth=1.0,
                    label=f'Mean = {V_rest} mV')
    axes[i].axhline(V_rest + std, color=color, linestyle=':', linewidth=1.2,
                    label=f'+1 std = {V_rest + std:.1f} mV')
    axes[i].axhline(V_rest - std, color=color, linestyle=':', linewidth=1.2,
                    label=f'−1 std = {V_rest - std:.1f} mV')
    axes[i].set_ylabel('V (mV)')
    axes[i].set_title(label + f'   —   analytical std = {std:.2f} mV,'
                      f'  simulated std = {V.std():.2f} mV')
    axes[i].legend(fontsize=8, loc='upper right')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(V_rest - 4*std - 1, V_rest + 4*std + 1)

axes[-1].set_xlabel('Time (ms)')
fig.suptitle('Ornstein-Uhlenbeck traces with different time constants\n'
             r'$C_m \frac{dV}{dt} = -G_{leak}(V - V_{rest}) + \sigma\, n(t)$',
             fontsize=13)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ou_three_traces.png', dpi=150)
plt.show()
print("Done.")
