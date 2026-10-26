import math
import random

color_threshold = 0.5
class synapse:
    def __init__(self, target, weight: float):
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

        self.last_spike_time = -float('inf')
        self.synapse_inputs = []
        self.connections = []

    def recieve_input(self, input_vesicles, synapse_strength):
        self.potential += input_vesicles * synapse_strength
    def update_stdp(self, current_time: float, is_pre_synaptic_firing: bool, synapse: synapse):
        A_plus, tau_plus = 1.5, 20
        A_minus, tau_minus = 1.2,20

        if is_pre_synaptic_firing:
            time_since_last_activation = synapse.target.last_spike_time-current_time
            if time_since_last_activation < 0 and math.isfinite(synapse.target.last_spike_time):
                dW = -A_minus * math.exp(time_since_last_activation/tau_minus)
                synapse.weight = max(0.0, min(30, synapse.weight+dW))
        else:
            time_since_last_activation = current_time - self.last_spike_time
            if time_since_last_activation > 0 and math.isfinite(self.last_spike_time):
                dW = A_plus * math.exp(-time_since_last_activation/ tau_plus)
                synapse.weight = max(0.0, min(30.0, synapse.weight + dW))
    
    def check_and_fire(self, current_time:float) -> bool:
        if self.potential >= self.threshold:
            self.last_spike_time = current_time

            for syn in self.connections:
                syn.target.recieve_input(input_vesicles=1, synapse_strength = syn.weight)


            self.potential = self.base
            return True
        return False
    
def connect_network(source_neurons: list, target_neurons: list, max_distance: float = 3.0, connection_prob: float = 0.4, weight_range: tuple = (10, 20)) -> int:
    synapse_count = 0
    for source in source_neurons:
        for target in target_neurons:
            dist = math.sqrt((source.x - target.x)**2 + (source.y - target.y)**2)
            
            if dist <= max_distance:
                if random.random() < connection_prob:
                    # Assign a random synaptic weight between 5.0 and 20.0 mV
                    weight = round(random.uniform(*weight_range), 2)
                    syn = synapse(target=target, weight=weight)

                    source.connections.append(syn)
                    target.synapse_inputs.append(syn)
                    synapse_count += 1
                    
    return synapse_count


#eye neuron
left_eye_receptors = [Neuron(x=random.uniform(0, 20), y=random.uniform(20, 40), neuron_type="visual") for i in range(200)]
right_eye_receptors = [Neuron(x=random.uniform(30, 50), y=random.uniform(20, 40), neuron_type="visual") for i in range(200)]

#ear neuron
left_ear_receptors = [Neuron(x=random.uniform(0, 20), y=random.uniform(50, 70), neuron_type="auditory") for i in range(200)]
right_ear_receptors = [Neuron(x=random.uniform(30, 50), y=random.uniform(50, 70), neuron_type="auditory") for i in range(200)]

#nose
left_nose_receptors = [Neuron(x=random.uniform(-20, 0), y=random.uniform(50, 70), neuron_type="auditory") for i in range(200)]
right_nose_receptors = [Neuron(x=random.uniform(-30, -50), y=random.uniform(50, 70), neuron_type="auditory") for i in range(200)]
#relay neuron
transport_neurones_sensory_to_thalamus = [Neuron(x=random.uniform(-50, 50), y=random.uniform(0, 70), neuron_type="relay") for i in range(10000)]

#synapses
total_left_eye_synapses = connect_network(left_eye_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
total_right_eye_synapses = connect_network(right_eye_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
total_right_ear_synapses = connect_network(right_ear_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
total_left_ear_synapses = connect_network(left_ear_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
total_right_nose_synapses = connect_network(right_nose_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
total_left_nose_synapses = connect_network(left_nose_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7,weight_range = (10,20))
transport_to_transport_synapses = connect_network(transport_neurones_sensory_to_thalamus, transport_neurones_sensory_to_thalamus, max_distance=2.0, connection_prob=0.5,weight_range = (10,20))

print(f"Successfully created {total_right_eye_synapses} synapses from Right Eye to Relay network.")
print(f"Successfully created {total_left_ear_synapses} synapses from Left Ear to Relay network.")
print(f"Successfully created {total_left_eye_synapses} synapses from Left Eye to Relay network.")
print(f"Successfully created {total_right_ear_synapses} synapses from Right Ear to Relay network.")
print(f"Successfully created {total_right_nose_synapses} synapses from Right Nose to Relay network.")
print(f"Successfully created {total_left_nose_synapses} synapses from Left Nose to Relay network.")

summing_target_neuron = Neuron(x=45.0, y=65.0, neuron_type="output")
transport_to_thalamus_to_output_synapses = connect_network(transport_neurones_sensory_to_thalamus, [summing_target_neuron], max_distance = 20, connection_prob = 0.97,weight_range = (0.1,0.5))

print("initial synaptic weights ")
for syn in summing_target_neuron.synapse_inputs[:5]:
    print(f"weight {syn.weight:.2f} mV")

all_sensory_receptors = (
    left_eye_receptors + right_eye_receptors +
    left_ear_receptors + right_ear_receptors +
    left_nose_receptors + right_nose_receptors
)
external_stimulus = 1
learning_rate = 0.01

num_output_synapses = max(1, len(summing_target_neuron.synapse_inputs))

for epoch in range(1,401):
    time_step = float(epoch*10)

    summing_target_neuron.potential = summing_target_neuron.base
    for r in all_sensory_receptors:
        r.potential = r.base
    for relay in transport_neurones_sensory_to_thalamus:
        relay.potential = relay.base

    active_count = random.randint(50, len(all_sensory_receptors))
    target_total_sum = float(active_count)
    active_receptors = random.sample(all_sensory_receptors, active_count)

    for receptor in active_receptors:
        receptor.potential += external_stimulus*20
        receptor.check_and_fire(time_step)
    active_relays =0
    for relay in transport_neurones_sensory_to_thalamus:
       if relay.check_and_fire(time_step + 1):
           active_relays +=1

    actual_output = summing_target_neuron.potential - summing_target_neuron.base
    error = target_total_sum - actual_output
    weight_delta = (learning_rate * error) / num_output_synapses

    if active_relays > 0:
        weight_delta = (learning_rate * error) / num_output_synapses
        for syn in summing_target_neuron.synapse_inputs:
            syn.weight = max(0.001, syn.weight + weight_delta)
    if epoch%40 == 0:
        print(f"Epoch {epoch:2d}, target sum = {target_total_sum:.2f}, output = {actual_output:.2f}, error = {error:.2f}")

external_stimulus = int(input("enter some number(integer)"))

for active_count in [150, 300, 600, 900]:
    summing_target_neuron.potential = summing_target_neuron.base
    for r in all_sensory_receptors:
        r.potential = r.base
    for relay in transport_neurones_sensory_to_thalamus:
        relay.potential = relay.base

    # Fire ONLY the first 'active_count' receptors with input = 1.0
    for receptor in all_sensory_receptors[:active_count]:
        receptor.potential += 20*external_stimulus
        receptor.check_and_fire(1000)

    # Propagate to output
    for relay in transport_neurones_sensory_to_thalamus:
        relay.check_and_fire(1001)

    readout = summing_target_neuron.potential - summing_target_neuron.base
    print(f"Active Inputs: {active_count:4d} | Expected: {active_count:4.2f} mV | Output Readout: {readout:7.2f} mV")

