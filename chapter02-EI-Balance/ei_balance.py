import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

# ============================================================
# Parameters
# ============================================================
dt      = 0.1       # ms
T       = 5000.0    # ms
t       = np.arange(0, T, dt)
N       = len(t)

# Neuron parameters
C_m     = 1.0       # membrane capacitance (uF/cm^2)
G_leak  = 0.05      # leak conductance (mS/cm^2)
V_rest  = -70.0     # resting potential (mV)
V_thr   = -55.0     # spike threshold (mV)
V_reset = -75.0     # reset potential (mV)
t_ref   = 2.0       # refractory period (ms)
ref_steps = int(t_ref / dt)

# Synaptic parameters
E_exc   = 0.0       # excitatory reversal potential (mV)
E_inh   = -80.0     # inhibitory reversal potential (mV)
tau_exc = 5.0       # excitatory synaptic time constant (ms)
tau_inh = 10.0      # inhibitory synaptic time constant (ms)

# Input parameters — balanced E/I
# Many weak excitatory inputs balanced by fewer strong inhibitory inputs
r_exc   = 2000      # excitatory Poisson input rate (Hz)
r_inh   = 500       # inhibitory Poisson input rate (Hz)
g_exc   = 0.012     # excitatory synaptic weight (mS/cm^2)
g_inh   = 0.048     # inhibitory synaptic weight (mS/cm^2)

# ============================================================
# Simulation — Leaky Integrate-and-Fire with conductance input
# C dV/dt = -G_leak(V-V_rest) - g_e(V-E_exc) - g_i(V-E_inh)
# ============================================================
V          = np.zeros(N);  V[0] = V_rest
g_e        = np.zeros(N)   # excitatory conductance trace
g_i        = np.zeros(N)   # inhibitory conductance trace
spikes     = []
refractory = 0

for k in range(1, N):

    # Poisson spike arrivals (number of events in dt)
    n_exc = np.random.poisson(r_exc * dt * 1e-3)
    n_inh = np.random.poisson(r_inh * dt * 1e-3)

    # Conductances: exponential decay + instantaneous jump at each input spike
    g_e[k] = g_e[k-1] * np.exp(-dt / tau_exc) + g_exc * n_exc
    g_i[k] = g_i[k-1] * np.exp(-dt / tau_inh) + g_inh * n_inh

    # Refractory period — clamp to reset
    if refractory > 0:
        V[k] = V_reset
        refractory -= 1
        continue

    # Euler integration of membrane equation
    dV = (dt / C_m) * (
        -G_leak * (V[k-1] - V_rest)
        - g_e[k] * (V[k-1] - E_exc)
        - g_i[k] * (V[k-1] - E_inh)
    )
    V[k] = V[k-1] + dV

    # Threshold crossing — fire spike
    if V[k] >= V_thr:
        V[k] = 20.0          # spike peak for visualisation
        spikes.append(t[k])
        refractory = ref_steps

spikes = np.array(spikes)
ISIs   = np.diff(spikes)     # interspike intervals (ms)

# ============================================================
# Statistics
# ============================================================
mean_rate = len(spikes) / (T * 1e-3)
CV        = np.std(ISIs) / np.mean(ISIs)   # coefficient of variation
                                            # Poisson process: CV = 1.0
print(f"Mean firing rate : {mean_rate:.1f} Hz")
print(f"Number of spikes : {len(spikes)}")
print(f"Mean ISI         : {np.mean(ISIs):.1f} ms")
print(f"Std ISI          : {np.std(ISIs):.1f} ms")
print(f"CV of ISI        : {CV:.3f}  (Poisson = 1.0)")

# ============================================================
# Plotting
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(12, 12))

# --- 1. Membrane potential trace ---
t_show = 1000          # ms to display
n_show = int(t_show / dt)
axes[0].plot(t[:n_show], V[:n_show], color='steelblue', linewidth=0.8)
axes[0].axhline(V_thr,  color='red',  linestyle='--', linewidth=1,
                label=f'Threshold = {V_thr} mV')
axes[0].axhline(V_rest, color='gray', linestyle='--', linewidth=1,
                label=f'Rest = {V_rest} mV')
axes[0].set_ylabel('Membrane potential (mV)')
axes[0].set_title('Leaky Integrate-and-Fire: Balanced Excitation and Inhibition')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

# --- 2. Synaptic conductances ---
axes[1].plot(t[:n_show], g_e[:n_show], color='tomato',    linewidth=0.8,
             label='$g_e$ (excitatory)')
axes[1].plot(t[:n_show], g_i[:n_show], color='steelblue', linewidth=0.8,
             label='$g_i$ (inhibitory)')
axes[1].set_ylabel('Conductance (mS/cm²)')
axes[1].set_title('Synaptic conductances')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

# --- 3. ISI histogram vs exponential (Poisson) ---
axes[2].hist(ISIs, bins=60, density=True, color='steelblue', alpha=0.7,
             label=f'Simulated ISI  (CV = {CV:.2f})')
x_isi      = np.linspace(0, ISIs.max(), 500)
rate_param = 1.0 / np.mean(ISIs)
axes[2].plot(x_isi, expon.pdf(x_isi, scale=1.0 / rate_param),
             'r--', linewidth=2, label='Exponential (Poisson, CV = 1)')
axes[2].set_xlabel('ISI (ms)')
axes[2].set_ylabel('Probability density')
axes[2].set_title(f'ISI distribution  —  Mean rate = {mean_rate:.1f} Hz,  CV = {CV:.2f}')
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

# --- 4. Return map of ISIs ---
axes[3].scatter(ISIs[:-1], ISIs[1:], alpha=0.3, s=8, color='steelblue')
axes[3].set_xlabel('ISI$_n$ (ms)')
axes[3].set_ylabel('ISI$_{n+1}$ (ms)')
axes[3].set_title('Return map of ISIs  (diffuse cloud = Poisson-like irregular spiking)')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/ei_balance_spiking.png', dpi=150)
plt.show()
print("Done.")
