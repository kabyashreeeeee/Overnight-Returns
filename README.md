# Overnight Return Prediction — Kabyashree Dey

Four-output forecasting system for the Quant Researcher Intern, Equity Desk assignment.

Four outputs per stock per session, from three parallel branches plus one
distribution support model. Nothing downstream depends on a previous branch's
*point forecast* except where that dependence is the point (magnitude reliability
must know the magnitude model's own errors).

## Run

```bash
python3 -m pip install -r requirements.txt
python3 main.py --config config.yaml --mode reproduce-all
```

`reproduce-all` runs: panel build (research mode, test excluded before target/feature construction) → magnitude
tournament → direction + quantile models → confidence branches → freeze → guarded
final scoring → CSV emission → report → verified ZIP. Roughly 25–45 minutes on 8 cores
with a warm minute cache; rebuilding the minute cache from raw files takes longer.

## How I spent my time

- 25% data integrity, exchange-calendar alignment, target construction and leakage controls.
- 30% economic hypotheses, feature design and descriptive diagnostics.
- 25% chronological OOF experiments, model comparison and ablation analysis.
- 10% uncertainty, calibration, transaction-cost and robustness analysis.
- 10% reproducibility checks, report writing and clean-room packaging.

Place the supplied Parquets at `../data/daily/` and `../data/minute/` relative to
`code/` (or edit `config.yaml`). If the configured minute cache is absent, the
pipeline deterministically rebuilds it from the raw 09:15–15:29 bars.

Individual stages:

```bash
python3 build_panel.py --mode research     # canonical panel, integrity report, OOF folds
python3 run_magnitude.py                   # Branch A tournament
python3 run_direction.py                   # Branch B + quantile support model
python3 run_confidence.py                  # Branches C and D
python3 run_freeze.py                      # freeze every selection
python3 run_final.py                       # single guarded test evaluation
python3 -m pytest tests/ -q
```

`run_final.py` refuses to start unless every source and research artifact matches
the checksums in `outputs/selection_manifest.json`. Packaging independently verifies
the selection manifest, final manifest, rounded CSV hashes and current source tree.

## Architecture

| Output | Branch | Model |
|---|---|---|
| `pred_magnitude_pct` | A | non-negative OOF blend of Gamma / Tweedie / Poisson / log-L2+smearing / HAR-X Ridge, all conditional-**mean** estimators |
| `pred_direction` | B | `sign(mu_market + lambda * mu_residual)`; Ridge market leg, LightGBM L2 residual leg |
| `conf_direction` | C | logistic on the aligned mean margin **and** the quantile model's `P(r>0)` plus context |
| `conf_magnitude` | D | two-stage: isotonic scale term `E[err\|m]` + Ridge on the scale-free residual error |

The quantile model (11 levels, monotone rearrangement) is a **support** model: it
supplies `P(r>0)`, interval width and tail mass to Branch C. It never produces the
submitted magnitude.

## Design decisions worth defending

**Magnitude targets the conditional mean, not the score.** The brief asks for
`E[|r| | F(T)]`; the headline `magnitude_score` is an L1 statistic minimised by the
conditional *median*. These are different functionals (Gneiting 2011). We target the
mean and take the score cost. A semantic gate refuses to label anything `E[|r|]`
unless its validation mean is within 5% of realised and its calibration slope sits in
[0.7, 1.4]. The score-optimised quantile model is retained as a documented shadow.

**`|r|` has a point mass at zero** — about 7.4% of training rows, where the T+1
auction prints at the previous close. Tweedie (`1<p<2`, compound Poisson-Gamma)
models that mass natively; Gamma needs a positive floor.

**Direction shrinkage is a signal-strength parameter, not a demeaning trick.**
`lambda` scales the residual expected return before the sign is taken, and is selected
on the **inner OOF folds only** — validation is never used for it.

**Direction confidence is not a rescaled score.** It combines two independently
fitted objects: the mean model's aligned margin and the quantile model's conditional
CDF at zero. Those disagree on ~13% of rows because `sign(E[r]) != sign(median)`
under skew.

**Magnitude reliability is split deliberately.** Raw `|m - |a||` is dominated by
aleatoric variance, so any model of it collapses onto `-pred_magnitude`. Stage 1
absorbs the scale term; stage 2 predicts what is left. We report the **within-
magnitude-decile** score alongside the headline as the non-degeneracy check.

**The supplied official close is much closer to an approximate final-30-minute
volume-weighted price constructed from the minute bars than to the 15:29 print**
(mean absolute difference 2.9 bp versus 20.5 bp). The exact gross-return identity is
`1 + r_official = (1 + g_basis)(1 + r_1529_to_open)`; equivalently,
`r_official = g_basis + r_1529_to_open + g_basis*r_1529_to_open`. The additive form
is only an approximation for small returns. The basis term is observable at T and is
included as `bench_gap_pct`. A fixed-specification,
validation-only ablation removes all explicit benchmark-basis variables and is
reported as a post-final diagnostic of the base direction model before the Part 3
confidence-flip rule, not as model selection. The report also scores
the frozen direction on the 15:29-to-open and 15:29-to-close sensitivity targets.

## Leakage controls

- Target uses the **master exchange calendar**, never a symbol's own next available row.
- Every target-derived feature is shifted before any rolling window.
- Imputers, scalers, smearing factors and calibrators are fit on the fold's training
  rows only.
- Same-date cross-sectional features are permitted (known by the close of T).
- Two five-session embargoes, on the master calendar.
- `--mode research` excludes test rows before target and feature construction;
  `run_final.py` is checksum-gated.

## Test governance

1. Training OOF research and candidate development.
2. Validation pass 1 exposed two specification omissions: no direct L2 conditional-mean
   candidate, and confidence based on `P(r>0)` rather than correctness of the emitted sign.
3. Those candidate families were added once, the registry was closed, and validation pass 2
   fixed every final choice.
4. The specification was frozen, evaluated on test, then identically re-executed for
   checksum, rounding-interface, packaging and clean-room verification.

No feature, candidate, hyperparameter, ensemble weight, calibration rule or emitted prediction
changed after the initial test evaluation. The stable specification digest in
`selection_manifest.json` binds the selected families, model configuration, feature list, blend
weights, lambda and model code. `reproduction_audit.json` records the independent raw-data rebuild
and byte-identical cache/CSV hashes. It does **not** claim first-emission identity because the
original first-test artifact was not separately archived.

**Protocol exception required by the brief:** across the overall project, the test block was
touched more than once: an earlier package and this frozen specification were both evaluated, and
the latter was then re-executed unchanged for engineering verification. The fresh model itself was
selected and frozen from training and validation without using test rows, evaluated on test only
afterward, and never changed after that result. However, the later choice between packages was
test-informed. Its test metrics are therefore descriptive rather than pristine holdout selection
evidence. The benchmark
ablation, executable 15:29 target and Part 4 test-baseline analysis are labelled post-final
diagnostics and did not affect the submitted specification.

The two-stage magnitude-reliability model beat inverse magnitude on validation but
did not do so on test. It remains frozen for protocol fidelity and is explicitly
demoted in the report rather than redesigned after observing test.

## Reproducibility

Seed 42 everywhere, LightGBM deterministic with fixed thread count, config-driven
relative paths and pinned dependencies. The minute cache is a rebuildable optimisation,
not an undeclared input. To audit in a clean directory, extract the ZIP, place the raw
data at the configured relative paths, install `code/requirements.txt`, run
`python3 code/main.py --config code/config.yaml --mode reproduce-all`, and compare the
three hashes recorded in `code/final_manifest.json`.
