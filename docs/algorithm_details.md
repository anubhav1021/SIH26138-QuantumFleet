# Algorithm details

This document restates the mathematical formulation and the quantum-inspired algorithms
implemented in this platform, in prose. The authoritative version is always the code: every
section below names the module that actually implements it.

## 1. Fuel consumption prediction (Deliverable 1)

### 1.1 Physics baseline

[`prediction/physics_model.py`](../src/quantumfleet/prediction/physics_model.py) implements an
Admiralty-coefficient power model:

```
P_kw = displacement_tonnes^(2/3) * speed_knots^3 / admiralty_coefficient
P_effective_kw = P_kw * resistance_multiplier(wave_height_m, headwind_knots, fouling_days)
```

`displacement_tonnes = lightship_tonnes + dwt_tonnes * load_factor`, with
`lightship_tonnes ≈ 0.35 * dwt_tonnes` (a standard naval-architecture rule of thumb).
`resistance_multiplier = 1 + 0.04*Hs² + 0.01*max(0, headwind) + 0.0003*fouling_days` is a smooth
empirical approximation of added resistance in waves, not a seakeeping computation.

Fuel mass is derived from power via each fuel's specific fuel oil consumption (SFOC), which is
itself *derived* -- not independently guessed -- from the fuel's lower heating value (LHV) and an
assumed brake thermal efficiency (BTE):

```
SFOC_g_per_kWh = 3600 / (BTE * LHV_MJ_per_kg)
fuel_kg = P_effective_kw * duration_hours * SFOC / 1000
```

All fuel constants (LHV, density, BTE, well-to-wake CO2e factors, indicative prices) live in one
place: [`fuels/properties.py`](../src/quantumfleet/fuels/properties.py). They are representative
values assembled from standard maritime-engineering and IMO/EU GHG literature orders of
magnitude -- internally consistent, not citation-pinned, and should be replaced with vetted
primary sources before any real decision-making use.

Two consequences fall directly out of this table and are worth surfacing in scenario analysis:
methanol and ammonia need roughly 2-2.2x HFO's fuel *mass* for the same propulsion energy (their
LHV is about half), while liquid hydrogen needs only ~35% of HFO's mass but, because it is so much
less dense, around 4.6x the *tank volume* for the same voyage energy -- a real alternative-fuel
storage tradeoff, not just a number in a table.

Shore power is deliberately **not** a seventh fuel: it is grid electricity that displaces a
vessel's at-berth auxiliary/hotel load, not a propulsion choice, so it is modelled as a separate
per-assignment flag rather than a peer of LNG/methanol/etc.

### 1.2 Data-driven model and the quantum-inspired tuner

[`data_generation/generator.py`](../src/quantumfleet/data_generation/generator.py) produces
synthetic noon-report-style voyage records (no real dataset was provided with this problem
statement -- see the README) using the physics baseline as ground truth, scaled by a per-vessel
efficiency offset and log-normal measurement noise.
[`prediction/ml_model.py`](../src/quantumfleet/prediction/ml_model.py) wraps Random Forest /
Gradient Boosting / Linear Regression behind one interface, evaluated in
[`prediction/evaluate.py`](../src/quantumfleet/prediction/evaluate.py) against the physics
baseline.

[`prediction/qpso_tuner.py`](../src/quantumfleet/prediction/qpso_tuner.py) implements
**Quantum-behaved Particle Swarm Optimization** (Sun, Feng & Xu, 2004) to tune the Random
Forest's hyperparameters. Unlike classical PSO, a QPSO particle carries no velocity: its next
position is sampled from a distribution centred between its personal best and the swarm's global
best, contracting around the swarm's mean-best position as a shrinking beta
(contraction-expansion) coefficient advances. This is what makes the *prediction* deliverable
quantum-inspired in its own right, independent of the fleet-optimization engine below.

## 2. Mathematical optimization formulation (Deliverable 2)

Implemented in [`optimization/problem.py`](../src/quantumfleet/optimization/problem.py).

**Decision variables.** Each route can host up to `N_SLOTS_PER_ROUTE` (3) independent
assignments -- a route can be served by a mix of vessel/fuel combinations, not just one. Each
assignment is `(vessel_class, fuel_type, cruising_speed, use_shore_power, count)`.

**Objectives** (minimize all three):
1. total fuel mass consumed (tonnes) over the planning period
2. total lifecycle (well-to-wake) CO2e emissions (tonnes)
3. total cost (USD): fuel/energy cost + vessel charter cost + any carbon levy

**Constraints:**
- cargo demand: each route's deliverable capacity must cover its cargo demand over the period
- schedule reliability: each deployed assignment's one-way transit time must not exceed the
  route's maximum transit days
- emission cap: each route's total CO2e must not exceed its emission cap
- fuel/vessel compatibility and shore-power availability (enforced by repair, not search)

## 3. Quantum-inspired optimization algorithm (Deliverable 3)

### 3.1 Qubit encoding

[`optimization/encoding.py`](../src/quantumfleet/optimization/encoding.py). Per assignment slot,
five gene groups: `vessel_class` and `fuel_type` and `speed_bin` are categorical, sharing one
roulette-wheel decode mechanism (`p_j = alpha_j^2 / sum(alpha_i^2)`); `count` uses the classic
Han-Kim binary-positional QEA encoding (3 qubits, values 0-7); `use_shore_power` is a single
Bernoulli-observed qubit. One shared decoder for all categorical groups is what makes a mixed
categorical/integer decision space tractable without inventing a different mechanism per group.

### 3.2 Rotation-gate update

[`optimization/qea.py`](../src/quantumfleet/optimization/qea.py). Each generation, every
individual rotates its qubits toward a concrete leader plan pulled from the Pareto archive: for a
categorical group, the qubit matching the leader's chosen option rotates toward certainty and the
others rotate away; for `count`, each bit rotates toward the leader's corresponding bit. Rotation
is norm-preserving by construction (`alpha^2+beta^2=1` always holds). Quantum mutation (a
"quantum NOT gate": swap alpha and beta) is applied with a small per-qubit probability for
diversity.

### 3.3 Multi-objective handling: the Pareto archive

[`optimization/pareto.py`](../src/quantumfleet/optimization/pareto.py). An external, bounded
archive of non-dominated plans, trimmed by crowding distance when it overflows -- simpler than
full NSGA-II (no combined-population ranking/tournament every generation) while still producing a
genuine, checkable Pareto front. **Constraint-dominance** (Deb's rule) folds constraint handling
directly into the comparator: a feasible plan always beats an infeasible one; among infeasible
plans, lower total normalized violation wins; among feasible plans, standard Pareto dominance on
(fuel, CO2e, cost).

### 3.4 Two mechanisms against premature convergence

Testing surfaced a real failure mode: while the archive holds no feasible plan yet,
constraint-dominance is a strict total order on violation, so the archive collapses to a single
entry, and the entire population -- all rotating toward that one leader -- converges onto a
single point well before a good region of the search space is explored.

1. **Composite-leader recombination.** Each generation, the optimizer tracks the best-known
   sub-solution independently *per route* (possibly discovered by different individuals) and
   recombines them into one composite candidate. Routes have independent qubits, so this is an
   implicit crossover across routes -- the natural independent building blocks here -- and it is
   what lets progress on one route compound with progress on another instead of requiring every
   route to be simultaneously lucky within a single individual. Without this mechanism, empirical
   testing showed feasibility became unreliable beyond 2-3 simultaneous routes; with it, the full
   50-route case study reliably reaches many feasible, non-dominated plans.
2. **Catastrophe operator.** When the best (min-violation, then max-hypervolume) score hasn't
   improved for `stagnation_patience` generations, a fraction of the population is reinitialized
   to fresh full superposition, reintroducing diversity. This is a standard, documented QEA
   technique (present in Han & Kim's original formulation) for exactly this failure mode.

## 4. Benchmarking design (Deliverable 5)

[`benchmarking/baselines_optimization.py`](../src/quantumfleet/benchmarking/baselines_optimization.py)
implements random search, a greedy single-vessel-type heuristic, and a classical genetic algorithm
(tournament selection, per-slot uniform crossover, random-reset mutation) -- all on the *same*
decoded representation and reusing `repair`/`constraints`/`pareto` unchanged. This isolates
exactly what the quantum-inspired rotation-gate mechanism contributes versus a conventional
evolutionary search sharing the same fitness, constraints, and archive.
[`benchmarking/runner.py`](../src/quantumfleet/benchmarking/runner.py) runs all of them across
multiple route-count instances under an identical evaluation budget, recording runtime, final
hypervolume (Monte Carlo estimate, see `pareto.monte_carlo_hypervolume`), and
generations-to-95%-hypervolume for a convergence-speed comparison.

An empirical finding from this benchmark (see `docs/case_study_results.md`): at a modest
evaluation budget, the classical GA -- lacking cross-route recombination -- found materially
fewer or zero feasible solutions where the QEA's composite-leader mechanism reliably found many,
directly demonstrating the value of the quantum-inspired mechanism rather than just asserting it.
