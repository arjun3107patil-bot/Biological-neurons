import math
import random

color_threshold = 0.5


class Neuron:
    def __init__(self, x: float, y: float, neuron_type: str = "relay"):
        self.x = x
        self.y = y
        self.neuron_type = neuron_type
        
        self.base_potential = -70.0
        self.threshold = -55.0
        self.potential = self.base_potential
    
        self.connections = []

    def receive_input(self, input_vesicles, synapse_strength):
        self.potential += input_vesicles * synapse_strength

    def check_and_fire(self) -> bool:
        if self.potential >= self.threshold:
            for target, strength in self.connections:
                target.receive_input(input_vesicles=1, synapse_strength=strength)
            

            self.potential = self.base_potential
            return True
        return False


def connect_network(source_neurons: list, target_neurons: list, max_distance: float = 3.0, connection_prob: float = 0.4) -> int:
    synapse_count = 0
    for source in source_neurons:
        for target in target_neurons:
            dist = math.sqrt((source.x - target.x)**2 + (source.y - target.y)**2)
            
            if dist <= max_distance:
                if random.random() < connection_prob:
                    # Assign a random synaptic weight between 5.0 and 20.0 mV
                    weight = round(random.uniform(5.0, 20.0), 2)
                    source.connections.append((target, weight))
                    synapse_count += 1
                    
    return synapse_count


#eye neuron
left_eye_receptors = [Neuron(x=random.uniform(0, 20), y=random.uniform(20, 40), neuron_type="visual") for i in range(100)]
right_eye_receptors = [Neuron(x=random.uniform(30, 50), y=random.uniform(20, 40), neuron_type="visual") for i in range(100)]

#ear neuron
left_ear_receptors = [Neuron(x=random.uniform(0, 20), y=random.uniform(50, 70), neuron_type="auditory") for i in range(100)]
right_ear_receptors = [Neuron(x=random.uniform(30, 50), y=random.uniform(50, 70), neuron_type="auditory") for i in range(100)]

#nose
left_nose_receptors = [Neuron(x=random.uniform(-20, 0), y=random.uniform(50, 70), neuron_type="auditory") for i in range(100)]
right_nose_receptors = [Neuron(x=random.uniform(-30, -50), y=random.uniform(50, 70), neuron_type="auditory") for i in range(100)]
#relay neuron
transport_neurones_sensory_to_thalamus = [Neuron(x=random.uniform(0, 50), y=random.uniform(0, 70), neuron_type="relay") for i in range(5000)]

#synapses
total_left_eye_synapses = connect_network(left_eye_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
total_right_eye_synapses = connect_network(right_eye_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
total_right_ear_synapses = connect_network(right_ear_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
total_left_ear_synapses = connect_network(left_ear_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
total_right_nose_synapses = connect_network(right_nose_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
total_left_nose_synapses = connect_network(left_nose_receptors, transport_neurones_sensory_to_thalamus, max_distance=3.0, connection_prob=0.7)
transport_to_transport_synapses = connect_network(transport_neurones_sensory_to_thalamus, transport_neurones_sensory_to_thalamus, max_distance=2.0, connection_prob=0.5)

print(f"Successfully created {total_right_eye_synapses} synapses from Right Eye to Relay network.")
print(f"Successfully created {total_left_ear_synapses} synapses from Left Ear to Relay network.")
print(f"Successfully created {total_left_eye_synapses} synapses from Left Eye to Relay network.")
print(f"Successfully created {total_right_ear_synapses} synapses from Right Ear to Relay network.")
print(f"Successfully created {total_right_nose_synapses} synapses from Right Nose to Relay network.")
print(f"Successfully created {total_left_nose_synapses} synapses from Left Nose to Relay network.")

#thalamus
#LGN 
#LGN_neurones = [Neuron(x=random.uniform(60, 80), y=random.uniform(50, 70), neuron_type="relay") for i in range(100)]
#MGN
#MGN_neurones = [Neuron(x=random.uniform(80, 60), y=random.uniform(50, 70), neuron_type="relay") for i in range(100)]



