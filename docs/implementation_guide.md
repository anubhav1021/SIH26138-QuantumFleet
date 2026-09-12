# Implementation guide

## Setup

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
```

Requires Python 3.10+. Verified working on Python 3.14 (all dependencies installed as prebuilt
Windows wheels); if a future environment lacks wheels for a dependency, install Python 3.12
alongside it and recreate the venv with `py -3.12 -m venv .venv`.

## Project layout

```
configs/                 vessel catalog + scenario definitions (YAML)
  user_scenarios/        user-created scenarios (Scenario Builder), one file per scenario
  user_vessels.yaml       user-added vessel classes, merged onto the built-in catalog
data/                    synthetic datasets (generated, gitignored) + schema docs
src/quantumfleet/        the library -- every script and the dashboard import from here
  fuels/                 fuel property constants and conversions
  prediction/            physics model, ML model, hybrid physics+ML model, features, QPSO tuner, evaluation
  data_generation/       synthetic voyage-record + scenario generators
  optimization/          problem formulation, qubit encoding, the QEA engine, constraints, repair, Pareto archive
  scenarios/             scenario/vessel persistence (ScenarioRepository, VesselCatalogRepository) + validation
  benchmarking/          classical baselines + the multi-instance benchmark runner
  reporting/             charts, explainability/comparison helpers, ports lookup, Markdown report builder
app/dashboard.py         Streamlit decision-support UI (orchestrates the tabs below)
app/scenario_builder_ui.py  Scenario Builder + Fleet Editor tab rendering (UI only; logic is in quantumfleet.scenarios)
app/analysis_ui.py       What-If Analysis tab rendering (UI only; logic is in quantumfleet.reporting.explain)
scripts/                 thin CLI entry points, one per pipeline stage
tests/                   pytest suite, one file per module
docs/                    this guide, algorithm details, and the case-study report
```

Every script in `scripts/` and every tab in the dashboard calls straight into
`src/quantumfleet/...` -- no logic lives only in a script or only in the UI, so the dashboard is
a genuine "software platform" over the same library, not a demo shell around it. The two `app/*_ui.py`
modules exist to keep `dashboard.py` from becoming a monolith, not to hold business logic themselves --
they call `quantumfleet.scenarios`/`quantumfleet.reporting.explain` for everything except widget rendering.

## Dashboard tabs

- **Scenario Builder** -- create, edit, duplicate, delete, save, and load scenarios entirely
  through the UI (no YAML editing). Live structural validation plus a cargo-feasibility pre-check
  (a generous upper-bound estimate, not a guarantee -- see `quantumfleet.scenarios.validation`).
- **Fleet Editor** -- add/edit/duplicate/delete custom vessel classes, layered onto the read-only
  built-in catalog.
- **Optimization Results** -- Pareto front, convergence, named priority presets (Lowest Fuel/Cost/
  Emissions/Balanced/Custom), the Recommended Fleet Plan (with cargo/schedule/emission compliance,
  fuel mix, carbon cost split out from total cost), a "Why this plan" explanation, an infeasibility
  explanation when applicable, and a Baseline vs. Optimized comparison against the greedy heuristic.
- **What-If Analysis** -- vary carbon price / alt-fuel pathway / cargo demand / emission caps and
  re-run the optimizer on both the current and modified scenario for a real, computed comparison.
- **Scenario & Routes** -- a route map (when routes have recognized port names or explicit
  coordinates) plus the full route/vessel catalog tables.
- **Prediction Model** -- physics/linear/RF/GB/QPSO-tuned-RF/hybrid comparison table, actual-vs-
  predicted and residual plots, feature importance, and the residual-learnability diagnostic
  (see "Hybrid prediction model" below).
- **Benchmarking** -- QEA vs. classical GA vs. random search vs. greedy on a scenario you size
  interactively.
- **Methodology** -- a plain-language summary with links to the docs below.

## Running the pipeline

```bash
# 1. Train and evaluate the fuel prediction models (physics/linear/RF/GB/QPSO-tuned RF)
python scripts/train_prediction_model.py

# 2. Run the quantum-inspired optimizer on the default 8-route scenario
python scripts/run_optimization.py

# 3. Benchmark the optimizer against classical baselines
python scripts/run_benchmark.py

# 4. (Re-)generate the 50-route demonstration scenario, if you want a different seed
python scripts/generate_case_study_scenario.py

# 5. Run the full demonstration: 50-route optimization + benchmarking -> docs/case_study_results.md
python scripts/run_case_study.py

# 6. Launch the interactive dashboard
streamlit run app/dashboard.py
```

Run `pytest` from the project root to run the full test suite (one file per module, plus
`tests/test_end_to_end_small.py` for a full-pipeline smoke test).

## Configuration

- `configs/vessel_types.yaml` -- 15 vessel classes (5 types x 3 size tiers): DWT, admiralty
  coefficient, day rate, speed range, compatible fuels, auxiliary load. Edit this to change the
  available fleet.
- `configs/default_scenario.yaml` / `configs/demo_case_study.yaml` -- route definitions (distance,
  cargo demand, schedule, allowed fuels, shore-power availability, emission cap) plus scenario-wide
  carbon price and green/grey alt-fuel pathway toggle.

**Important:** emission caps are *calibrated*, not arbitrary numbers -- see
`quantumfleet.data_generation.scenario_generator.calibrate_emission_cap` and
`docs/algorithm_details.md`. If you hand-edit or add routes, recalibrate their caps the same way
(run the physics model against a minimum single-vessel-class HFO deployment and set the cap at
~1.3x that reference), or the optimizer may find no feasible solution at all -- this exact mistake
is what the first version of `default_scenario.yaml` made, caught by testing.

## Hybrid prediction model

`src/quantumfleet/prediction/hybrid_model.py` implements a physics-informed hybrid predictor: an
ML model trained on the *residual* (actual - physics), added back to the physics estimate, rather
than replacing physics outright. **Honest finding from actually testing this** (not assumed): on
this platform's synthetic dataset, the residual is essentially unpredictable from the available
features (cross-validated R² of predicting it is negative), because the dataset's only error
sources -- a per-vessel random efficiency offset and i.i.d. measurement noise -- aren't correlated
with any observed feature. So the hybrid model performs on par with physics alone here, not
better. It's still built and tested properly because real voyage data typically *does* have
feature-correlated physics-model bias (hull fouling, engine degradation, etc.), where this
architecture is expected to help. `residual_learnability_r2()` is the reusable diagnostic that
tells you which regime you're in for a given dataset -- shown live in the Prediction Model tab.
For this reason, the QEA optimizer's fitness function continues to use the physics formula
directly, never the hybrid model -- there's no accuracy benefit on this data to justify slowing
down the search loop's thousands of evaluations. See `docs/algorithm_details.md` §3.5 for the
related (and similarly evidence-based) finding about the QEA vs. greedy cost comparison.

## Swapping in real data

The synthetic voyage-record generator exists because no dataset was actually provided with this
problem statement (the linked "dataset" redirects to the same PS document). To use real
AIS/noon-report data instead of the synthetic generator:

1. Produce a dataframe with exactly the columns in
   `quantumfleet.data_generation.schema.VoyageRecordColumns` (see
   `data/schema/voyage_record_schema.md` for the full column reference) and set
   `data_source="real"`.
2. Nothing downstream changes: `prediction/features.py`, `prediction/ml_model.py`, and
   `data_generation/validate.py` all reference the schema's column-name constants, never a
   hardcoded string.

## Known scope decisions

See the "Scope" section of the original implementation plan (not tracked in this repo) for the
full list; the load-bearing ones:
- No quantum SDK (Qiskit etc.) is used -- "quantum-inspired" means classical metaheuristics using
  quantum-computing metaphors (qubit superposition, rotation-gate updates), matching the PS's own
  "Quantum Optimization or equivalent" wording.
- No dedicated REST API layer -- the CLI scripts plus the dashboard's shared library calls already
  satisfy "user interface or API" without the added scope of a server layer.
- Hull fouling is modelled as a constant-0 term (the formula supports it; no scenario currently
  varies it) -- the cheapest thing to cut if a future extension needs the budget elsewhere.
