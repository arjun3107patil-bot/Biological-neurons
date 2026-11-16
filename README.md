# Biological-neurons
# Hodgkin-Huxley Neuron Model in Python

An implementation of the classic **Hodgkin-Huxley (1952) model** simulating action potential generation in a biological neuron using Python and NumPy.

---
## Scientific theory
The Hodgkin-Huxely model is a mathematical model for action-potential neurons:
**The neuron** A neuron has a lipid bilayer covering it, which seperates the ions inside and outside the neuron
**The ions** Positive Na ions exist in high concentration outside the neuron whereas Positive K ions exist in high concentration inside the neuron(both do exist inside and outside only the concentration varies)
**mechanism of discharge** There are ion channels that exist on the bilayer that are voltage dependant that allow the flow of ions. Both the Na channels and K channels function differently.There are 2 forces acting on the ions(electric force and force via diffusion) which depend on current charge stored therefore it is easier to represent this as a differential equation than absolute algebra.Whichever prevails determines the direction of flow of ions determining the firing of neurons.

## Overview

The Hodgkin-Huxley model describes how action potentials in neurons are initiated and propagated through voltage-gated ion channels. This project models:
* **Sodium ($\text{Na}^+$)** channel activation ($m$) and inactivation ($h$) dynamics.
* **Potassium ($\text{K}^+$)** channel activation ($n$) dynamics.
* **Leak** current and membrane capacitance ($C_m$).
* Numerical integration of membrane potential using the **Forward Euler Method**.

---

## Features

* **Biophysically Accurate Kinetics:** Calculates voltage-dependent rate constants ($\alpha$, $\beta$) for gating variables.
* **Singularity Handling:** Safely handles mathematical limits ($V \to -55\text{ mV}$ and $V \to -40\text{ mV}$) to prevent division-by-zero errors.
* **Visual Simulation:** Plots membrane potential alongside gating variable probabilities ($n, m, h$) over time using Matplotlib.

---

## Parameters

| Parameter | Description | Standard Value |
| :--- | :--- | :--- |
| $C_m$ | Membrane Capacitance | $1.0\ \mu\text{F/cm}^2$ |
| $g_{\text{Na}}$ | Max Sodium Conductance | $120.0\ \text{mS/cm}^2$ |
| $g_{\text{K}}$ | Max Potassium Conductance | $36.0\ \text{mS/cm}^2$ |
| $g_L$ | Max Leak Conductance | $0.3\ \text{mS/cm}^2$ |
| $E_{\text{Na}}$ | Sodium Reversal Potential | $+50.0\ \text{mV}$ |
| $E_{\text{K}}$ | Potassium Reversal Potential | $-77.0\ \text{mV}$ |
| $E_L$ | Leak Reversal Potential | $-54.387\ \text{mV}$ |

---

## Prerequisites

Ensure you have Python 3.7+ installed along with the required libraries:

```bash
pip install numpy matplotlib
