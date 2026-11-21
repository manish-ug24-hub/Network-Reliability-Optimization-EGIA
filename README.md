[README.md](https://github.com/user-attachments/files/35205562/README.md)
# EGIA-NROP: Enhanced Gazelle-Inspired Algorithm for Network Reliability Optimisation

Python implementation of the **Enhanced Gazelle-Inspired Algorithm (EGIA)** for the **Network Reliability Optimisation Problem (NROP)**, based on the paper *"Network Reliability Optimisation Using the Enhanced Gazelle-Inspired Algorithm"* by Sidharth Gupta, Manish and Unnat Pangtey (Department of Information Technology, Netaji Subhas University of Technology, Delhi).

EGIA adapts the Gazelle Optimization Algorithm (GOA) to discrete network design. It uses a predator-evasion phase (exploration), a Lévy-flight grazing phase (exploitation), binary projection, a repair operator and a reliability-aware penalty fitness.

## Problem

A network is a graph `G = (V, E)` where each link works with probability `p`. The decision vector `x` is a binary vector over all candidate edges of the complete graph on `n` nodes. Two modes are supported:

| Mode | Objective | Constraint |
|------|-----------|------------|
| 1 | Minimise total link cost | Reliability `R(x) >= R0` |
| 2 | Maximise reliability | Total cost `<= c0` |

Constraints are handled with penalty terms in the fitness function (lower fitness is better).

## Features

- `NROPInstance`: random link costs, reliability estimator, Monte-Carlo reliability validator, and a repair operator (guarantees connectivity and minimum node degree 2 using the cheapest edges)
- `egia(...)`: Algorithm 1 of the paper, in two variants
  - `variant="paper"`: literal pseudocode with binary state
  - `variant="improved"`: continuous latent state with greedy replacement
- `ga_baseline(...)`: a simple Genetic Algorithm used as a comparison baseline
- `egia_continuous(...)`: the same dynamics for continuous (CEC-style) benchmark functions
- Command-line demo that compares all variants in both modes and can save convergence plots

## Requirements

- Python 3.9+
- `numpy`
- `scipy`
- `matplotlib` (optional, only needed for `--plot`)

```bash
pip install numpy scipy matplotlib
```

## Usage

```bash
# default demo: n=12, p=0.8, 5 runs, 10,000 function evaluations
python Egia_nrop.py

# custom instance and convergence plots
python Egia_nrop.py --n 20 --p 0.95 --runs 10 --fe 10000 --plot
```

### Command-line options

| Option | Default | Description |
|--------|---------|-------------|
| `--n` | 12 | Number of nodes |
| `--p` | 0.8 | Link reliability |
| `--runs` | 5 | Independent runs per algorithm |
| `--fe` | 10000 | Function-evaluation budget |
| `--R0` | 0.95 | Reliability threshold (Mode 1) |
| `--c0_frac` | 0.4 | Mode 2 budget as a fraction of the complete-graph cost |
| `--seed` | 0 | Seed for the problem instance (link costs) |
| `--plot` | off | Save `convergence_mode1.png` and `convergence_mode2.png` |

The script also runs a continuous EGIA sanity check on the Rastrigin function (D=10).

### Use as a library

```python
from Egia_nrop import NROPInstance, make_fitness, egia

inst = NROPInstance(n=12, p=0.8, seed=0)
fitness = make_fitness(inst, mode=1, R0=0.95)
best_x, best_f, history = egia(fitness, inst.repair, inst.D,
                                max_fe=10_000, variant="improved", seed=1)

print("cost:", inst.total_cost(best_x))
print("reliability (estimate):", inst.reliability_estimate(best_x))
print("reliability (Monte-Carlo):", inst.reliability_mc(best_x))
```

## Implementation details

- Link costs are integers drawn uniformly from 1 to 10 (configurable in `NROPInstance`)
- Default `R0 = 0.95` (Mode 1) and `c0 = 0.4 x` cost of the complete graph (Mode 2)
- Reliability is estimated with a fast node-cut (degree) product, `prod_i (1 - q^{d_i})` with `q = 1 - p`, and can be validated with the built-in Monte-Carlo all-terminal estimator. To use a different estimator, override `NROPInstance.reliability_estimate`.
- The Lévy step is clipped for numerical stability
- The GA baseline is a generic implementation (tournament selection, uniform crossover, bit-flip mutation, elitism) and is not the GPC method from the literature

## Example output

`python Egia_nrop.py` (n=12, p=0.8, 5 runs, 10,000 evaluations), Mode 1 mean cost:

| Algorithm | Mean cost |
|-----------|-----------|
| EGIA (paper-literal) | ~121 |
| EGIA (improved) | ~101 |
| GA baseline | ~52 |

Exact numbers vary with the seed, instance and parameters.

## Project structure

```
Egia_nrop.py   # algorithm, problem model, baseline, demo CLI
README.md
LICENSE
```

## Citation

```
Sidharth Gupta, Manish, Unnat Pangtey,
"Network Reliability Optimisation Using the Enhanced Gazelle-Inspired Algorithm",
Netaji Subhas University of Technology, Delhi.
```

Based on the Gazelle Optimization Algorithm: Q. Diao, C. Xie, Y. Yin, H. Lee, H. Yang, *A multi-strategy improved gazelle optimization algorithm for solving numerical optimization and engineering applications*, Expert Systems with Applications, 2023.

## License

Released under the MIT License. See [LICENSE](LICENSE).

MIT License

Copyright (c) 2026 Sidharth Gupta, Manish, Unnat Pangtey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
