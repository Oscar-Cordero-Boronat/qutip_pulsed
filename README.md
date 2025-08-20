# Pulsed Squeezing Simulator

This repository contains a Python simulation framework for **pulsed squeezed light**. It provides theoretical models and example notebooks to explore squeezing in 1D cavities.

---

## Repository Structure

- `theoretical_pulsed.py` — Python file containing 3 main classes  
- `tutorial-notebook.ipynb` — Tutorial notebook demonstrating usage  
- `main-notebook.ipynb` — Main notebook that explains the theory and gives some results interpretation 
- `pyproject.toml` — UV app project configuration for dependencies  
- `uv.lock` — UV lock file to ensure reproducible installs  
- `README.md` — This file  

---

## Classes

All main functionality is implemented in `theoretical_pulsed.py`:

- **TheoreticalPulsedSqueezing** — [Simulate pulsed squeezing in an optical cavity with various input pulse shapes.]  
- **TheoreticalPulsed1D** — [1D parameter sweep for pulsed squeezing simulations.]  
- **PulsedSqueezingVisualizer** — [Visualizer for pulsed squeezing simulations with various pulse shapes.]  

---

## Notebooks

- **`tutorial-notebook.ipynb`** — Step-by-step tutorial showing how to use the classes for simulations.  
- **`main-notebook.ipynb`** — Notebook that explains the theory and gives some results interpretation 

---

## Installation

The project uses **UV** to manage Python dependencies. `pyproject.toml` and `uv.lock` define the required packages.
