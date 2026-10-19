import math,random, numpy as np,matplotlib.pyplot as plt

class neuron:
    def __init__(self,initial_voltage = -65):
        #capacitance and conductance constants
        self.membrane_capacitance = 1
        self.max_K_conductance = 36
        self.max_Na_conductance = 120
        self.max_Leak_conductance = 0.3

        #equilibrium potentials
        self.K_equilibrium_potential = -77.0  
        self.Na_equilibrium_potential = 50
        self.Leak_equilibrium_potential = -54.387

        #Variables of the state
        self.v = initial_voltage
        self.n = self.alpha_n(self.v)/(self.alpha_n(self.v)+self.beta_n(self.v))
        self.m = self.alpha_m(self.v)/(self.alpha_m(self.v)+self.beta_m(self.v))
        self.h = self.alpha_h(self.v)/(self.alpha_h(self.v)+self.beta_h(self.v))


    def alpha_n(self,voltage):
        relative_v = voltage+55
        if relative_v == 0:
            output = 0.1
        else:
           output = (0.01*relative_v)/(1-np.exp(-relative_v/10))
        return output
    def alpha_m(self,voltage):
            relative_v = voltage+40
            if relative_v == 0:
                output = 1
            else:
               output = (0.1*relative_v)/(1-np.exp(-relative_v/10))
            return output
    def alpha_h(self,voltage):
            return 0.07 * np.exp(-(voltage+65)/20)
    def beta_n(self,voltage):
            return 0.125 * np.exp(-(voltage+65)/80)
    def beta_m(self,voltage):
            return 4 * np.exp(-(voltage+65)/18)
    def beta_h(self,voltage):
            return 1/(1+np.exp(-(voltage+35)/10))
    def step(self,I_ext, dt):
        I_k = self.max_K_conductance*(self.n**4)*(self.v - self.K_equilibrium_potential)
        I_Na = self.max_Na_conductance*(self.m**3)*self.h*(self.v - self.Na_equilibrium_potential)
        I_Leak = self.max_Leak_conductance * (self.v - self.Leak_equilibrium_potential)

        dV_dt = (I_ext - I_Na - I_Leak - I_k)/self.membrane_capacitance
        dn_dt = self.alpha_n(self.v)*(1-self.n) - self.beta_n(self.v) * self.n 
        dm_dt = self.alpha_m(self.v)*(1-self.m) - self.beta_m(self.v) * self.m
        dh_dt = self.alpha_h(self.v)*(1-self.h) - self.beta_h(self.v) * self.h

        self.v += dV_dt*dt
        self.n += dn_dt*dt
        self.m += dm_dt*dt
        self.h += dh_dt*dt

        return self.v,self.n,self.m,self.h

if __name__ == "__main__":
    dt = 0.005
    t_max = 70
    time = np.arange(0,t_max, dt)

    neuron = neuron(initial_voltage=-65.0)

    v_hist, n_hist, m_hist, h_hist = [],[],[],[]

    for t in time :
            I_ext = 10.0 if 10.0 <= t <= 50.0 else 0.0 # add current from 10ms to 30 ms


            v,n,m,h = neuron.step(I_ext , dt)

            v_hist.append(v)
            n_hist.append(n)
            m_hist.append(m)
            h_hist.append(h)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(time, v_hist, "b", label="Voltage (mV)")
    ax1.set_ylabel("Membrane Potential (mV)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time, n_hist, label="n (K+ activation)")
    ax2.plot(time, m_hist, label="m (Na+ activation)")
    ax2.plot(time, h_hist, label="h (Na+ inactivation)")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Gating Variables")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
     