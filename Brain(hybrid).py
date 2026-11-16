import math
import random
from collections import defaultdict


class Synapse:
    def __init__(self, source, target, weight: float):
        self.source = source
        self.target = target
        self.weight = weight


class Neuron:
    def __init__(self, x: float, y: float, neuron_type: str = "relay"):
        self.x = x
        self.y = y
        self.neuron_type = neuron_type

        self.base = -70.0
        self.threshold = -55.0
        self.potential = self.base

        self.last_spike_time = -float("inf")
        self.synapse_inputs = []
        self.connections = []

    def receive_input(self, input_vesicles: int, synapse_strength: float):
        self.potential += input_vesicles * synapse_strength

    def check_and_fire(self, current_time: float) -> bool:
        if self.potential >= self.threshold:
            self.last_spike_time = current_time
            for syn in self.connections:
                syn.target.receive_input(input_vesicles=1, synapse_strength=syn.weight)
            self.potential = self.base
            return True
        return False


def connect_network(
    source_neurons: list,
    target_neurons: list,
    max_distance: float = 3.0,
    connection_prob: float = 0.4,
    weight_range: tuple = (10, 20),
) -> int:
    grid = defaultdict(list)
    cell_size = max_distance

    for target in target_neurons:
        cx = int(math.floor(target.x / cell_size))
        cy = int(math.floor(target.y / cell_size))
        grid[(cx, cy)].append(target)

    synapse_count = 0
    max_dist_sq = max_distance ** 2

    for source in source_neurons:
        sx = int(math.floor(source.x / cell_size))
        sy = int(math.floor(source.y / cell_size))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for target in grid.get((sx + dx, sy + dy), ()):
                    if source is target:
                        continue
                    dist_sq = (source.x - target.x) ** 2 + (source.y - target.y) ** 2
                    if dist_sq <= max_dist_sq and random.random() < connection_prob:
                        weight = round(random.uniform(*weight_range), 2)
                        syn = Synapse(source=source, target=target, weight=weight)
                        source.connections.append(syn)
                        target.synapse_inputs.append(syn)
                        synapse_count += 1

    return synapse_count


# --- 1. Sensory Receptors (1,200 total) ---
left_eye_receptors = [Neuron(random.uniform(0, 20), random.uniform(20, 40), "visual") for _ in range(200)]
right_eye_receptors = [Neuron(random.uniform(30, 50), random.uniform(20, 40), "visual") for _ in range(200)]
left_ear_receptors = [Neuron(random.uniform(0, 20), random.uniform(50, 70), "auditory") for _ in range(200)]
right_ear_receptors = [Neuron(random.uniform(30, 50), random.uniform(50, 70), "auditory") for _ in range(200)]
left_nose_receptors = [Neuron(random.uniform(-20, 0), random.uniform(50, 70), "olfactory") for _ in range(200)]
right_nose_receptors = [Neuron(random.uniform(-50, -30), random.uniform(50, 70), "olfactory") for _ in range(200)]

all_sensory_receptors = (
    left_eye_receptors + right_eye_receptors +
    left_ear_receptors + right_ear_receptors +
    left_nose_receptors + right_nose_receptors
)

# --- 2. Relay & Central Output Integrator ---
relay_neurons = [Neuron(random.uniform(-50, 50), random.uniform(0, 70), "relay") for _ in range(10000)]
summing_target_neuron = Neuron(x=0.0, y=35.0, neuron_type="output")

# --- 3. Wire the Network (Calibrated to prevent early saturation) ---
for receptors in [
    left_eye_receptors, right_eye_receptors,
    left_ear_receptors, right_ear_receptors,
    left_nose_receptors, right_nose_receptors,
]:
    # Larger receptive radius (6.0) + lower weights (1.5 - 3.5 mV) gives smooth, non-saturating recruitment
    connect_network(receptors, relay_neurons, max_distance=6.0, connection_prob=0.6, weight_range=(1.5, 3.5))

# Light lateral coupling for spatial smoothing without runaway cascades
connect_network(relay_neurons, relay_neurons, max_distance=2.0, connection_prob=0.2, weight_range=(1.0, 2.5))

# Connect all relay regions to the output integrator
output_synapses = connect_network(
    relay_neurons, [summing_target_neuron], max_distance=75.0, connection_prob=0.4, weight_range=(0.05, 0.15)
)


def run_forward_pass(active_receptors: list, stimulus_strength: float, base_time: float):
    summing_target_neuron.potential = summing_target_neuron.base
    for r in all_sensory_receptors:
        r.potential = r.base
    for relay in relay_neurons:
        relay.potential = relay.base

    for receptor in active_receptors:
        receptor.potential += stimulus_strength * 20.0
        receptor.check_and_fire(base_time)

    # Synchronous evaluation so lateral inputs add cleanly without multi-wave explosion
    firing_relays = [r for r in relay_neurons if r.potential >= r.threshold]
    for relay in firing_relays:
        relay.check_and_fire(base_time + 1.0)


# --- 4. Training Loop ---
learning_rate = 0.08

for epoch in range(1, 2001):
    time_step = float(epoch * 10)
    active_count = random.randint(20, len(all_sensory_receptors))
    target_total_sum = float(active_count)

    # Mix random patterns and contiguous organ slices so both generalize accurately
    if epoch % 2 == 0:
        active_receptors = random.sample(all_sensory_receptors, active_count)
    else:
        active_receptors = all_sensory_receptors[:active_count]

    run_forward_pass(active_receptors, stimulus_strength=1.0, base_time=time_step)

    active_output_synapses = [
        syn for syn in summing_target_neuron.synapse_inputs
        if syn.source.last_spike_time == (time_step + 1.0)
    ]

    actual_output = summing_target_neuron.potential - summing_target_neuron.base
    error = target_total_sum - actual_output

    if active_output_synapses:
        weight_delta = (learning_rate * error) / len(active_output_synapses)
        for syn in active_output_synapses:
            syn.weight = max(0.0001, syn.weight + weight_delta)

    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Target: {target_total_sum:7.2f} | Output: {actual_output:7.2f} | Error: {error:+7.2f}")

# --- 5. Evaluation Loop ---
print("\n--- Testing Trained Network (Contiguous Slices) ---")
for active_count in [150, 300, 600, 900]:
    run_forward_pass(all_sensory_receptors[:active_count], stimulus_strength=1.0, base_time=50000.0)
    readout = summing_target_neuron.potential - summing_target_neuron.base
    print(f"Active Inputs: {active_count:4d} | Expected: {active_count:7.2f} mV | Output Readout: {readout:7.2f} mV")

print("\n--- Testing Trained Network (Random Across All Organs) ---")
for active_count in [150, 300, 600, 900]:
    run_forward_pass(random.sample(all_sensory_receptors, active_count), stimulus_strength=1.0, base_time=60000.0)
    readout = summing_target_neuron.potential - summing_target_neuron.base
    print(f"Active Inputs: {active_count:4d} | Expected: {active_count:7.2f} mV | Output Readout: {readout:7.2f} mV")