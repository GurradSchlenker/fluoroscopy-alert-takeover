# Causal Evaluation of Fluoroscopy-Based Robot Navigation Alerts on Surgeon Takeover in Peripheral Vascular Intervention

This is the analysis release for the manuscript of that title. It implements the emulated
target trial the paper reports: person-windows of a robot-assisted peripheral vascular
procedure are the units of analysis, delivery of a fluoroscopy-derived navigation alert
inside a window is the exposure, and the outcomes are surgeon takeover and procedure
completion. The effects are estimated by cross-fitted augmented inverse-probability
weighting on a certified-overlap region, the completion effect is decomposed through the
surgeon's hand-back under the interventional analogue, the identification assumptions are
probed by a pre-specified falsification pair, and the reporting tables are produced from
the same run. The alert itself is produced by a learned spatial predicate over the
fluoroscopic stream, with a linear-time state-space alternative.

## Layout

```
configs/            one directory per protocol stage, plus the experiment files
  protocol/         the emulated trial: time zero, follow-up, windows, outcomes
  cohort/           the two data layers and their linkage
  alert/            the navigation alert policy and its training settings
  gate/             the certified-overlap gate of Algorithm 1
  estimator/        Eq. (2): family, folds, trimming, the ablation grid
  mediation/        Eq. (1): the interventional analogue and the window sweep
  falsification/    Algorithm 2: negative controls, the E-value, the bias-factor grid
  operating_point/  Algorithm 5: the threshold sweep and the abstention band
  strata/           Table 3 strata and the honest-split causal forest
  inference/        cluster-robust, bootstrap and the two multiplicity families
  experiment/       one file per reported experiment, ablation and sensitivity analysis
scripts/            thin launchers that only set PYTHONPATH and call a command
src/fluoroscopy_alert_takeover/
  protocol/         analytic schema, window layout, trial strategies, identification probes
  cohort/           readers for the record layers and the person-window builder
  perception/       the alert policy, its loss, its training loop and its calibration
  exposure/         balance, the family selection rule and the exposure model
  overlap/          the certified-overlap gate and its effective-sample-size trajectory
  estimands/        the interventional analogue, its diagnostics and the window sweep
  estimators/       Eq. (2), the singly-robust pair, targeted learning, the ablation grid
  forest/           the honest-split causal forest and its gradient test
  falsification/    the negative-control panel, the E-value and the bias-factor curve
  operating_point/  the causal benefit curve and the abstention band
  strata/           the printed strata, the performance-gap ceiling, heterogeneity
  inference/        cluster-robust standard errors, the bootstrap, alternative estimands
  evaluation/       discrimination, the three-arm reader comparison, the ceiling
  workflow/         procedure metrics and the latency accounting of Table A1
  reporting/        table assembly and the pre-specified success criteria
  multiplicity/     Holm step-down and Benjamini-Hochberg
  verification/     both verification layers, the claim map and the integrity manifest
  analysis/         the end-to-end pipeline that composes all of the above
  utils/            configuration, seeding, logging, atomic IO, column scaling
  cli/              one command per stage
tests/              the suite, and the literal rows its tables are built from
THIRD_PARTY_NOTICES.txt   copyright and licence of the dependencies
dataset_urls.txt          the data endpoints, and why there is only one kind of entry
claim_to_code.json        every manuscript claim against the code that carries it
verification_report.json  both verification layers with the numbers they produced
integrity_manifest.json   SHA-256 of every tracked file
```

## Installation

Python 3.11 or newer.

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The suite, the linters, the formatter and the type checker come from
`requirements-dev.txt`; `.pre-commit-config.yaml` wires them into a commit hook:

```
pip install -r requirements-dev.txt
pre-commit install
```

With conda:

```
conda env create -f environment.yml
conda activate fluoroscopy-alert-takeover
```

With Docker:

```
docker build -t fluoroscopy-alert-takeover .
```

The image carries no accelerator runtime. The offline causal estimation is the stage the
container is sized for, and the manuscript reports that stage at 18.4 minutes per 10,000
procedures on eight CPU cores (Table A1). Train the alert policy outside the image.

## Data

There is one cohort and no open dataset. The record-level cohort is a de-identified
multi-site set of robot-assisted peripheral vascular procedures held under site
data-sharing agreements; the Data availability statement says the raw data cannot be shared
and may be requested from the corresponding author. No URL describes it, so none is
asserted. `dataset_urls.txt` records that fact and nothing else.

`scripts/prepare_data.sh <cohort-root>` checks the root against the layout below and prints
what is missing; it never writes into it.

```
<cohort-root>/
  site_registry.csv              site, region, annual_procedure_volume
  procedures.csv                 one row per procedure: case mix, operating-report outcomes,
                                 workflow quantities, integration cost
  reader_study.csv               the three-arm observational reader comparison
  navigation/<procedure_id>.jsonl   timestamped events: alert, score, takeover, hand_back,
                                    period_completed, stage, exchange, device
  cine/<procedure_id>.csv           frame index: time, subtraction quality, the spatial
                                    predicate label, the annotation flag
  frames/<procedure_id>.npy         the fluoroscopic frames
  masks/<procedure_id>.npy          the reference masks the predicate is scored against
```

The cohort builder reads the layers above and writes the analytic person-window table. It
is the only place that touches record-level data; every later stage works on the table.

Sizes and hashes are not published here because they describe the restricted cohort. The
manifest in `integrity_manifest.json` covers this release tree, not the cohort.

## Analysis

Every command takes one experiment file and any number of `key=value` overrides.

```
python -m fluoroscopy_alert_takeover.cli.prepare   --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.gate      --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.estimate  --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.decompose --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.falsify   --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.strata    --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.operating_point --experiment configs/experiment/primary.yaml
python -m fluoroscopy_alert_takeover.cli.tables    --experiment configs/experiment/primary.yaml
```

`scripts/launch_analysis.sh` runs the eight stages in order. Each stage writes its own
record under the run directory, so a stage that cannot proceed names itself rather than
failing the run silently.

The alert policy is trained separately, where the cine frames are staged:

```
python -m fluoroscopy_alert_takeover.cli.train_alert --experiment configs/experiment/primary.yaml
```

The reported experiments are:

| file | what it reproduces |
|---|---|
| `primary.yaml` | Table 1: both causal contrasts, the decomposition, the alert-model block |
| `ablation_estimator_family.yaml` | Table 2, the four estimator rows |
| `ablation_adjustment_set.yaml` | Table 2, the four adjustment-set rows |
| `ablation_nuisance_learner.yaml` | Table 2, the three nuisance-family rows |
| `ablation_cross_fitting.yaml` | Table 2, the single-fold row |
| `ablation_certified_gate.yaml` | Table 2, the ungated row and the positivity-deficit price |
| `sensitivity_bias_factor.yaml` | Table 4 Panel C |
| `sensitivity_inference.yaml` | Table 4 Panel E |
| `supplementary_site_validation.yaml` | Table A2 |
| `supplementary_threshold_sweep.yaml` | Table A3 Panel A |
| `supplementary_volume_tertiles.yaml` | Table A3 Panel B |
| `smoke.yaml` | nothing reported; the unit tests only |

## What the release can and cannot show here

The cohort is restricted, so the cohort-dependent quantities cannot be produced in this
tree. They are recorded as BLOCKED in `verification_report.json` rather than omitted:

- the alert model's AUROC 0.918, AUPRC 0.674, Brier 0.132 and predicate Dice 0.809;
- the printed contrasts of Table 1, +3.2 pp on takeover and +2.6 pp on completion;
- the certified region and its retained share;
- the per-stage latencies of Table A1, which are wall-clock on the deployment configuration.

Everything that does not need the cohort is executed. The forward and backward passes, the
parameter update, the checkpoint round trip and a single-batch overfit run without any data
at all. The estimator stack is checked against identities that hold by construction: the
decomposition must be additive, `Psi(a, g_a)` must collapse to the AIPW arm mean of Eq. (2),
and Eq. (2) with a flat outcome regression must equal the hand-written inverse-probability
terms. The gate, the two correction procedures, the interval constructions, the
heterogeneity statistic and the Dice coefficient are checked against closed forms written
independently of the code they test. Nothing in this tree produces data, so the estimators
are never checked against a number that the same tree also produced.

## Verification

```
scripts/run_verification.sh
```

The command runs the suite, then both verification layers, then refreshes
`claim_to_code.json`, `verification_report.json` and `integrity_manifest.json`.

Layer one maps every numbered equation, algorithm box and printed table to the code that
carries it, and recomputes the relations the manuscript's own printed values must satisfy.
Where those relations do not hold, the finding is reported with `failure_owner` set to
`manuscript`: it is a discrepancy inside the paper's tables, not a defect of this release,
and the two are kept apart so one cannot be mistaken for the other. Layer two executes the
release and records PASS, FAIL, NOT_RUN or BLOCKED per check, with the numbers each check
produced.

The audit currently raises eight findings inside the printed values, all of them recorded
in `verification_report.json` with the arithmetic that produced them:

- the largest unadjusted standardised mean difference in Table 1 is 0.290 (lesion length),
  not the 0.213 the text attributes to anatomic complexity;
- the complexity classes within each exposure group sum to 4,133 and 4,681, against the
  group sizes of 4,503 and 5,361;
- the three site strata sum to 10,792, against an analytic cohort of 9,864;
- the E-value implied by the printed outcome rates is 2.142, not the printed 2.06;
- the bias factor at the null crossing implied by the printed risk difference is 1.165, not
  the printed 1.41;
- the conventional I-squared for Cochran Q = 2.04 on two degrees of freedom is 1.96%, not
  the printed 21.4%;
- the printed site estimates span 2.6 to 3.6, which the stated leave-one-site-out range of
  3.0 to 3.5 does not contain;
- the largest difference among the printed subgroup estimates is 2.7 points, not the 3.8
  the maximum-gap row reports.

Two further claims are NOT_RUN because the manuscript names an instrument without
defining it: the 94.1% alert-policy coverage and the per-stage latency attribution. Four
execution checks are BLOCKED because they need the private cohort: the alert model's
discrimination and Dice, the Table 1 contrasts, the certified region, and the deployment
latencies.

## Compute budget

The manuscript reports deployment and offline figures, not a development budget:

- deployment: one mid-range accelerator per site at batch size 1; the end-to-end alert path
  is 54.7 ms median with a 3.4 GB peak, and the state-space alternative is 27.4 ms with
  2.4 GB (Table A1);
- the deployable budget is defined by the observed alert-to-action latency, median 2,400 ms
  with an interquartile range of 1,600 to 3,800 ms; the end-to-end path consumes 2.3% of it
  at the median;
- offline causal estimation: 18.4 minutes per 10,000 procedures on eight CPU cores.

No GPU type, count, VRAM figure or wall-clock total for model development is reported
anywhere in the manuscript, so none is stated here. The alert-policy training settings in
`configs/alert/default.yaml` are this release's own and are listed as engineering defaults
in `verification_report.json`.

## Reproducing the reported numbers

A machine with the cohort staged and the hardware of Table A1 can run
`scripts/launch_analysis.sh` and compare the emitted records against the manuscript. The
tolerance a reader should expect is set by the cohort: the printed intervals are the
manuscript's own, and this release reproduces the estimator that produced them rather than
the sampling noise inside them. Every deviation between this release and the manuscript is
recorded in `verification_report.json` under `engineering_defaults` when it is a value the
manuscript does not print, and as a manuscript finding when it is one it does.

## Licence

Apache License 2.0. See `LICENSE` and `THIRD_PARTY_NOTICES.txt`.
