# Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization

Smart India Hackathon problem statement **SIH26138**. A software platform for predicting vessel
fuel consumption and optimizing green fleet deployment -- vessel mix, capacity, cruising speed,
alternative fuels (LNG, methanol, hydrogen, ammonia), and shore power -- using a quantum-inspired
evolutionary algorithm, benchmarked against classical optimization methods.

No dataset was actually provided with this problem statement (the linked "dataset" redirects back
to the PS document itself), so the platform includes a physics-informed synthetic voyage-data
generator as a first-class module, built to the same schema real AIS/noon-report data would use.

## What's here

| # | Deliverable | Where |
|---|---|---|
| 1 | Fuel Consumption Prediction Model | `src/quantumfleet/fuels/`, `src/quantumfleet/prediction/`, `src/quantumfleet/data_generation/` |
| 2 | Mathematical Optimization Formulation | `src/quantumfleet/optimization/problem.py`, `src/quantumfleet/optimization/constraints.py` |
| 3 | Quantum-Inspired Optimization Algorithm | `src/quantumfleet/optimization/encoding.py`, `qea.py`, `pareto.py`, `repair.py` |
| 4 | Software Platform / Decision Support System | `app/dashboard.py`, `src/quantumfleet/reporting/`, `scripts/` |
| 5 | Demonstration | `scripts/run_case_study.py`, `docs/`, `src/quantumfleet/benchmarking/` |

- **Prediction**: an Admiralty-formula physics baseline plus Random Forest / Gradient Boosting
  models, hyperparameter-tuned by Quantum-behaved Particle Swarm Optimization (QPSO).
- **Optimization**: a qubit-encoded evolutionary algorithm (rotation-gate updates, quantum-NOT
  mutation) driving an external Pareto archive, with a composite-leader recombination step and a
  catastrophe operator for population diversity -- see `docs/algorithm_details.md`.
- **Benchmarking**: the same encoding, constraints, and archive power a classical genetic
  algorithm, random search, and a greedy heuristic, isolating what the quantum-inspired mechanism
  specifically contributes.
- **Decision support**: a Streamlit dashboard for scenario building, running the optimizer,
  exploring the Pareto front, picking a plan by priority, and exporting a report.

## Quick start

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .

python scripts/train_prediction_model.py     # fuel prediction models
python scripts/run_optimization.py           # quantum-inspired optimizer, default scenario
python scripts/run_benchmark.py              # vs. classical baselines
python scripts/run_case_study.py             # full 50-route demonstration -> docs/case_study_results.md
streamlit run app/dashboard.py               # interactive dashboard
pytest                                        # test suite
```

See [`docs/implementation_guide.md`](docs/implementation_guide.md) for setup detail, project
layout, and how to swap in real voyage data. See
[`docs/algorithm_details.md`](docs/algorithm_details.md) for the full mathematical formulation and
algorithm design, including an empirical finding on premature convergence in naive QEA
implementations and how this platform addresses it. See
[`docs/case_study_results.md`](docs/case_study_results.md) for the 50-route demonstration and
benchmark results.

## Stack

Python, numpy/pandas/scikit-learn, matplotlib/plotly, Streamlit, pytest. No quantum SDK --
"quantum-inspired" here means classical metaheuristics using quantum-computing metaphors, matching
the problem statement's own "Quantum Optimization or equivalent" wording.
