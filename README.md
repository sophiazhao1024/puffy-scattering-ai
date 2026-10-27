# Puffy Scattering AI

[中文说明](README.zh-CN.md)

Python numerical calculations and Gaussian-process surrogates for finite-size Yukawa scattering.

**The idea is to calculate a small set of parameter points with a physics solver, then train a model to predict scattering results quickly within the same parameter domain.** Model uncertainty is used to choose where additional physics calculations are performed.

## What the code calculates

The solver computes non-relativistic scattering between two finite-size particles with uniform spherical density profiles and an attractive Yukawa interaction. The motivating application is finite-size scattering in puffy dark matter models.

It calculates the interaction potential, radial solutions, partial-wave phase shifts, and momentum-transfer scattering quantities. Main outputs are:

- `phases`: partial-wave phase shifts.
- `S`: the dimensionless sum `S = sigma_T * k^2 / (4*pi)`.
- `sigma_mphi2`: the dimensionless momentum-transfer cross section `sigma_T * m_phi^2`.
- `tail_fraction`: the fraction contributed by the final three partial-wave terms, used to check truncation.

Inputs are `b = 2*mu*alpha/m_phi`, `kappa = mu*v/m_phi`, `ychi = m_phi*R_chi`, and `yn = m_phi*R_n`. Here `mu` is the reduced mass, `alpha` the total effective coupling, `m_phi` the mediator mass, `v` the relative speed, and the two radii describe the particles. Natural units are used. Both densities integrate to one; no additional nuclear coherence enhancement is applied implicitly.

The current ML benchmark fixes `kappa=0.35`, `ychi=1`, and `yn=0.5`, varying only `0.2 <= b <= 7`. **It is a one-dimensional parameter-slice example, not a general model for arbitrary masses, velocities, and radii.**

## Changes from the original Mathematica workflow

The original notebook's intended scattering workflow evaluates a potential, solves radial equations, extracts phase shifts, sums partial-wave contributions, and saves or plots the result for each parameter point. Scanning new points repeats the numerical calculation.

This project retains that physical workflow but implements a new independent Python solver and adds a machine-learning prediction layer. **It is not a line-by-line port, and numerical equivalence to the original notebook or all paper results has not been established. Training labels come from the new Python solver, not notebook exports.**

| Change | Reason |
|---|---|
| Separate physics, surrogate, experiment, and CLI modules | Reduce dependence on notebook execution order and historical variable state; support testing and repeatable runs. |
| Compute finite-size potentials using sphere-overlap geometry and one-dimensional quadrature | Explicitly handle unequal radii and use consistent density normalization. |
| Transform the radial equation and match to spherical Bessel functions | Make the numerical conventions explicit and avoid direct division by a wavefunction at its nodes. |
| Add partial-wave tail checks and stricter-configuration comparisons | Check the numerical reliability of reference labels before interpreting surrogate errors. |
| Add a Gaussian-process surrogate | Predict new points quickly after training, reducing solver calls during repeated queries. |
| Add uncertainty-based active sampling | Allocate additional solver calls to locations where the model is most uncertain. |
| Add independent holdout evaluation, PCHIP interpolation, and a Born baseline | Measure usefulness against alternatives rather than only displaying a fitted curve. |

The first four changes concern numerical implementation and software structure; **they are not AI**. The ML components are GP prediction and active sampling. Neural networks and residual GPs are not implemented.

## Before and after

Original scattering workflow:

```text
Each parameter point -> potential -> radial solver -> phase shifts
                     -> partial-wave sum -> saved results and plots
Next parameter point -> repeat the numerical calculation
```

New training workflow:

```text
24 initial parameter points -> Python solver -> training labels
                                                    |
                                                 fit GP
                                                    |
                      select the most uncertain unlabeled candidate
                                                    |
                              run the physics solver there and refit
                                                    |
                                 repeat 6 times to obtain the final GP
```

After training, on the same fixed slice and within the training range:

```text
New b value -> GP -> predicted S and model uncertainty
```

The GP learns `log10(b) -> log10(S)`, giving positive predictions after transformation back to the original scale. Active sampling selects the largest posterior standard deviation among 501 candidate locations, excluding already labeled points. Reference labels for 20 independent test points are generated after training and selection are complete; test labels do not guide selection.

## Measured results

The following values come from the included [demo report](results/demo/metrics.json): 24 initial training points, 6 actively selected additions, 20 independent test points, and seed 2026. Errors are measured against the **new Python reference solver**, not experimental truth or original Mathematica output.

| Method | log10 RMSE | Maximum relative error |
|---|---:|---:|
| GP with 24 initial points | 0.000797654 | 0.787% |
| GP with 30 final points | 0.000112669 | 0.116% |
| PCHIP with the same 30 final points | 0.000403888 | 0.242% |
| Born approximation | 0.316122 | 278.443% |

In this experiment, the GP's maximum test error decreased from approximately 0.787% to 0.116% after adding labels, and was smaller than PCHIP's on the same final samples. The Born approximation is a physical approximation with substantial deviations at some points on this slice; that alone does not imply an implementation error.

**Prediction is fast, but preparing the model costs time:**

| Timing measurement | Recorded time |
|---|---:|
| Reference solver for 20 test points | 5.360 s |
| Trained GP prediction for the same 20 points | 0.000292 s |
| Prediction-only speed ratio | Approximately 18368x |
| Generating all 30 training and 20 test labels | 12.852 s |
| All GP fits combined | 0.166 s |

Label-generation time includes test-reference calculations; those rows must not be added together. The reported timing also excludes subsequent stricter-configuration checks and other full-workflow overhead. Sub-millisecond prediction timings depend on hardware, batching, and measurement conditions. The ratio is neither a stable performance guarantee nor a whole-project speedup.

The demonstrated benefit is **fast prediction with small observed test errors on this slice**. It is useful when a trained model can be reused for many queries; directly solving a few points may be preferable. End-to-end net savings have not been established, and an equal-budget random-sampling comparison has not established that active selection is superior.

Not every metric improves: in the [smaller validation run](results/release-smoke/metrics.json), increasing training size from 8 to 10 points improved GP log10 RMSE, while median relative error rose slightly from approximately 0.537% to 0.575%.

## When to trust a prediction

Observed test errors are not domain-wide error bounds. Narrow resonances or minima can be missed by both training and test samples. GP intervals estimate model uncertainty; they are not rigorous physical-error bounds.

The model rejects queries outside its training `b` range, but **does not automatically fall back to the physics solver**. It also does not check whether the other fixed parameters have changed: changing `kappa` or either radius requires new labels and retraining. A complete error-based acceptance mechanism is not implemented. Important physical conclusions should be checked with a converged reference calculation.

Saved reports preserve optimizer warnings under `optimizer_warnings`. This repository is a runnable research prototype for preliminary parameter exploration. It does not validate every paper figure, all resonances across parameter space, or detector event rates.

## Installation and usage

Requires Python 3.11 or later; Python 3.12 is recommended. Run from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

Calculate one point using the physics solver:

```bash
python -m puffy solve --b 2 --kappa 0.35 --ychi 1 --yn 0.5
```

Generate labels, fit the GP, add active samples, and evaluate:

```bash
python -m puffy benchmark --out results/my-run --ntrain 24 --ntest 20 --active 6 --seed 2026
```

This command recalculates training labels; it does not load a bundled pretrained model. Outputs include `training.csv`, `holdout.csv`, `metrics.json`, and `solver_refinement.csv`. Use a new output directory to preserve the bundled records.

Run the introductory examples, including training from an existing CSV:

```bash
python examples/quickstart.py
python examples/train_from_csv.py
```

Optional plotting:

```bash
python -m pip install -e ".[plot]"
python -m puffy.plot_results results/my-run
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/puffy/physics.py` | Finite-sphere potential, radial equations, phase shifts, cross sections, and Born baseline. |
| `src/puffy/surrogate.py` | GP fitting, prediction, domain checks, and PCHIP baseline. |
| `src/puffy/experiment.py` | Label generation, active sampling, holdout tests, and reports. |
| `src/puffy/__main__.py` | `solve` and `benchmark` commands. |
| `tests/` | Physical limits, numerical comparisons, input checks, and surrogate tests. |
| `examples/` | Single-point calculations and training from CSV. |
| `notebooks/01_quickstart.ipynb` | Introductory Python notebook. |
| `results/demo/` | Records for 24+6 training points and 20 test points. |
| `results/release-smoke/` | Records for 8+2 training points and 5 test points. |
| `docs/mathematica_to_python.md` | Relationship to the original notebook and remaining work. |

See [technical notes](docs/technical_notes.md) for detailed physical conventions and numerical methods, and [validation.txt](docs/validation.txt) for the current release's test record.

## License

New code is available under the MIT [LICENSE](LICENSE). The original paper and Mathematica notebook are outside that license's scope; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
