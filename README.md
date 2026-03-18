# Traffic Control using GNN and TCA

Implementation of a **Graph Neural Network (GNN)** framework for learning and classifying traffic states generated from a **Traffic Cellular Automata (TCA)** simulation.  
The model approximates traffic dynamics and performs **three-phase traffic classification** based on simulated traffic data.

This repository is the implementation of the paper:

**Z. Mousavi Ziabari et al. "Labeled Cellular Automata to Three-Phase Traffic Classification", Transportation Research Procedia, 2026.**

---

## Overview

Traffic is simulated using a **Traffic Cellular Automata (TCA)** model where road segments are discretized into cells and vehicle movement follows predefined update rules.

The generated traffic states are labeled according to **three-phase traffic theory**:

- **Free Flow**
- **Synchronized Flow**
- **Wide Moving Jam**

These states are then represented as graphs and used to train a **Graph Neural Network (GNN)** that learns spatial relationships between traffic cells and predicts traffic phases.

---

## Repository Structure
Traffic_Control_TCA_GNN/

- GNN.py – # Main script containing GNN implementation
- TCA.py –Traffic Cellular Automaton implementation
- TrafficGraph.py – # Traffic simulation data generation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ZahraMousaviZiabari/Traffic_Control_TCA_GNN.git
cd Traffic_Control_TCA_GNN
```
Create an environment and install dependencies:

Python, Numpy, Pytorch, PyTorch Geometric, Scikit-learn, Matplotlib

---

## Run

Train and test the GNN model:

python GNN.py

## Citation

If you use this code, please cite:
```bash
@article{ziabari2026tca,
  title={Labeled Cellular Automata to Three-Phase Traffic Classification: An application of graph neural networks for traffic control},
  author={Zahra Mousavi Ziabari, Jonas Mårtensson, Matthieu Barreau},
  journal={Transportation Research Procedia},
  year={2026}
  url={https://www.sciencedirect.com/science/article/pii/S2352146526000761}
}
```
