import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

class neuron:
    def __init__(self, initial_voltage = -65):
        # Capacitance and conductance constants
        self.membrane_capacitance = 1
        self.max_K_conductance = 36
        self.max_Na_conductance = 120
        self.max_Leak_conductance = 0.3

        # Equilibrium potentials
        self.K_equilibrium_potential = -77.0  
        self.Na_equilibrium_potential = 50
        self.Leak_equilibrium_potential = -54.387

        # Variables of the state
        self.v = initial_voltage
        self.n = self.alpha_n(self.v) / (self.alpha_n(self.v) + self.beta_n(self.v))
        self.m = self.alpha_m(self.v) / (self.alpha_m(self.v) + self.beta_m(self.v))
        self.h = self.alpha_h(self.v) / (self.alpha_h(self.v) + self.beta_h(self.v))

    def alpha_n(self, voltage):
        relative_v = voltage + 55
        return 0.1 if relative_v == 0 else (0.01 * relative_v) / (1 - np.exp(-relative_v / 10))

    def alpha_m(self, voltage):
        relative_v = voltage + 40
        return 1.0 if relative_v == 0 else (0.1 * relative_v) / (1 - np.exp(-relative_v / 10))

    def alpha_h(self, voltage):
        return 0.07 * np.exp(-(voltage + 65) / 20)

    def beta_n(self, voltage):
        return 0.125 * np.exp(-(voltage + 65) / 80)

    def beta_m(self, voltage):
        return 4 * np.exp(-(voltage + 65) / 18)

    def beta_h(self, voltage):
        return 1 / (1 + np.exp(-(voltage + 35) / 10))

    def step(self, I_ext, dt):
        I_k = self.max_K_conductance * (self.n**4) * (self.v - self.K_equilibrium_potential)
        I_Na = self.max_Na_conductance * (self.m**3) * self.h * (self.v - self.Na_equilibrium_potential)
        I_Leak = self.max_Leak_conductance * (self.v - self.Leak_equilibrium_potential)

        dV_dt = (I_ext - I_Na - I_Leak - I_k) / self.membrane_capacitance
        dn_dt = self.alpha_n(self.v) * (1 - self.n) - self.beta_n(self.v) * self.n 
        dm_dt = self.alpha_m(self.v) * (1 - self.m) - self.beta_m(self.v) * self.m
        dh_dt = self.alpha_h(self.v) * (1 - self.h) - self.beta_h(self.v) * self.h

        self.v += dV_dt * dt
        self.n += dn_dt * dt
        self.m += dm_dt * dt
        self.h += dh_dt * dt

        return self.v, self.n, self.m, self.h


class synapse:
    def __init__(self, source, target, weight: float, delay_steps: int):
        self.source = source      
        self.target = target      
        self.strength = weight    
        self.delay_steps = delay_steps  # Physical distance-based propagation delay


if __name__ == "__main__":
    # --- Network Parameters ---
    N = 150                    
    sensory_neurons = 10       # Indices 0 to 9 are sensory
    connection_radius = 35.0   
    connection_prob = 0.4      
    synaptic_weight = 25.0     
    conduction_velocity = 15.0 # Units per ms (controls wave speed)

    dt = 0.005
    t_max = 250.0       # Original timeline for 20 waves spaced every 10ms
    time = np.arange(0, t_max, dt)

    # 1. Generate spatial coordinates [-70, 70]
    np.random.seed(42)
    x = np.random.uniform(-70, 70, N)
    y = np.random.uniform(-70, 70, N)

    # Cluster sensory neurons tightly in the center
    x[:sensory_neurons] = np.random.uniform(-15, 15, sensory_neurons)
    y[:sensory_neurons] = np.random.uniform(-15, 15, sensory_neurons)

    coords = np.column_stack((x, y))

    # 2. Instantiate population
    neuron_population = [neuron(initial_voltage=-65.0) for _ in range(N)]

    # 3. Build spatial adjacency matrix & physical delay synapses
    distances = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    adjacency_matrix = np.zeros((N, N), dtype=bool)
    synapse_list = []

    for i in range(N):
        for j in range(i + 1, N):
            dist = distances[i, j]
            if dist < connection_radius:
                if np.random.rand() < connection_prob:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
                    
                    # Calculate real physical propagation delay in time steps based on distance
                    delay_ms = dist / conduction_velocity
                    delay_steps = max(1, int(round(delay_ms / dt)))
                    
                    synapse_list.append(synapse(source=i, target=j, weight=synaptic_weight, delay_steps=delay_steps))
                    synapse_list.append(synapse(source=j, target=i, weight=synaptic_weight, delay_steps=delay_steps))

    # 4. Run simulation loop
    v_all_hist = np.zeros((N, len(time)))
    spike_tracker = np.zeros(N, dtype=bool)
    spike_history = {i: [] for i in range(N)}
    
    # Synaptic activation tracking array (smoothly decays over time)
    synaptic_activation = np.zeros(N)
    decay_factor = np.exp(-dt / 2.0)

    for idx, t in enumerate(time):
        # Decay existing synaptic currents smoothly
        synaptic_activation *= decay_factor

        # Check which neurons are spiking (crossing 0 mV threshold)
        for i in range(N):
            if neuron_population[i].v > 0.0 and not spike_tracker[i]:
                spike_tracker[i] = True
                spike_history[i].append(idx)
            elif neuron_population[i].v < -20.0:
                spike_tracker[i] = False

        # Apply synaptic transmission using true physical distance delays
        for syn in synapse_list:
            src = syn.source
            tgt = syn.target
            arrival_idx = idx - syn.delay_steps
            
            if arrival_idx >= 0 and arrival_idx in spike_history[src]:
                synaptic_activation[tgt] += syn.strength

        # Update each neuron
        for i in range(N):
            I_ext = 0.0
            
            # ALL sensory neurons receive 20 gradually increasing pulses every 10 ms (Original setup)
            if i < sensory_neurons:
                for pulse_idx in range(20):
                    pulse_start = 10.0 + (pulse_idx * 10.0)
                    pulse_end = pulse_start + 4.0
                    if pulse_start <= t <= pulse_end:
                        I_ext = 5.0 + (pulse_idx * 2.5)
                        break

            total_input = I_ext + synaptic_activation[i]

            v, _, _, _ = neuron_population[i].step(total_input, dt)
            v_all_hist[i, idx] = v

    # 5. Interactive Visualization Setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), facecolor='white')
    
    # Adjust layout to make room at the bottom for interactive widgets
    plt.subplots_adjust(bottom=0.22)

    # --- Left Plot: Spatial Topology ---
    for i in range(N):
        for j in range(i + 1, N):
            if adjacency_matrix[i, j]:
                ax1.plot([x[i], x[j]], [y[i], y[j]], color='#cbd5e1', alpha=0.4, linewidth=0.8, zorder=1)
                
    ax1.scatter(x[sensory_neurons:], y[sensory_neurons:], c='#3b82f6', s=45, edgecolors='#1e3a8a', linewidths=0.6, label="Network Neurons", zorder=2)
    ax1.scatter(x[:sensory_neurons], y[:sensory_neurons], c='#f97316', s=90, edgecolors='#7c2d12', linewidths=1.0, label="Sensory Neurons", zorder=3)
    
    ax1.set_title(f"Spatial Network Topology ({N} Neurons)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("X Position ($\mu$m)", fontsize=11)
    ax1.set_ylabel("Y Position ($\mu$m)", fontsize=11)
    ax1.legend(loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.set_xlim(-75, 75)
    ax1.set_ylim(-75, 75)

    # --- Right Plot: Voltage Traces ---
    ax2.plot(time, v_all_hist[0], label="Sensory (Neuron 0)", color='#f97316', linewidth=2.0, zorder=4)
    for n_id in range(15, N, 15):
        ax2.plot(time, v_all_hist[n_id], label=f"Neuron {n_id}", alpha=0.7, linewidth=1.2)

    ax2.set_xlabel("Time (ms)", fontsize=11)
    ax2.set_ylabel("Voltage (mV)", fontsize=11)
    ax2.set_ylim(-85, 55)
    ax2.legend(loc='upper right', frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1', fontsize=9, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.3)

    # Initial view set to Wave 1 focused view (tight 30ms window)
    initial_wave = 1
    p_idx = initial_wave - 1
    w_start = 10.0 + (p_idx * 10.0)
    ax2.set_xlim(w_start - 2.0, w_start + 28.0)
    ax2.set_title(f"Single Wave Propagation (Wave #{initial_wave} Isolated)", fontsize=13, fontweight='bold', pad=10)

    # --- Add Interactive Widgets at the Bottom ---
    ax_slider = plt.axes([0.15, 0.08, 0.50, 0.03], facecolor='#f8fafc')
    slider_wave = Slider(ax_slider, 'Wave Select', 1, 20, valinit=initial_wave, valstep=1)

    ax_radio = plt.axes([0.72, 0.04, 0.18, 0.12], facecolor='#f8fafc')
    radio_mode = RadioButtons(ax_radio, ('Single Wave', 'All Waves'), active=0)

    def update(val):
        wave_num = int(slider_wave.val)
        mode = radio_mode.value_selected
        
        if mode == 'Single Wave':
            p_idx = wave_num - 1
            w_start = 10.0 + (p_idx * 10.0)
            # Tight 30ms window centered strictly on the selected wave
            ax2.set_xlim(w_start - 2.0, w_start + 28.0)
            ax2.set_title(f"Single Wave Propagation (Wave #{wave_num} Isolated)", fontsize=13, fontweight='bold', pad=10)
        else:
            ax2.set_xlim(0.0, t_max)
            ax2.set_title("Full Network Activity (All 20 Waves)", fontsize=13, fontweight='bold', pad=10)
        
        fig.canvas.draw_idle()

    slider_wave.on_changed(update)
    radio_mode.on_clicked(update)

    # Save high-res backup image and show interactive popup window
    plt.savefig("neuron_network_output.png", dpi=300)
    print("Simulation finished! Graph saved as 'neuron_network_output.png' and opening interactive display window...")
    plt.show()