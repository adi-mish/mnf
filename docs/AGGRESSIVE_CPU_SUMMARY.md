# Aggressive CPU sweep summary

This file summarizes `docs/aggressive_cpu_sweeps.json`, generated with
`1,000` seeds and `8` worker processes.

## Active Design

| Noise | Mean Measurements | Stable Rate | Accuracy | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 0.000 | 22.000 | 1.000 | 1.000 | 1.000 |
| 0.005 | 22.002 | 1.000 | 1.000 | 1.000 |
| 0.010 | 22.002 | 1.000 | 1.000 | 1.000 |
| 0.020 | 26.851 | 1.000 | 1.000 | 1.000 |
| 0.030 | 52.566 | 1.000 | 0.983 | 0.991 |
| 0.040 | 80.290 | 1.000 | 0.915 | 0.955 |

## Matched-Budget Baselines

At noise `0.02`, budget `64`, active design reaches stable labels in every
seed with accuracy `1.000`; uniform repeats are stable in `0.917` of seeds, and
random repeats in `0.125`.

At noise `0.03`, budget `64`, active design is stable in every seed with
accuracy `0.983`; uniform repeats have accuracy `1.000` but are stable in only
`0.003` of seeds. This is a useful warning: the current stability rule and
truth accuracy are not the same object.

At budget `256`, uniform repeats recover `1.000` accuracy and stability at
noise `0.02`, `0.03`, and `0.04`; active design stops earlier but is less
accurate at higher noise. The active rule is sample-efficient, not uniformly
best under all budgets.

## Noisy Recovery

The abstaining classifier keeps covered-label accuracy at `1.000` across all
tested noise levels by abstaining more aggressively as noise increases:

| Noise | Hard Accuracy | All-Correct Rate | Abstention Rate | Covered Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 0.010 | 1.000 | 1.000 | 0.003 | 1.000 |
| 0.020 | 0.995 | 0.933 | 0.172 | 1.000 |
| 0.030 | 0.942 | 0.422 | 0.423 | 1.000 |
| 0.050 | 0.694 | 0.012 | 0.707 | 1.000 |
| 0.080 | 0.417 | 0.000 | 0.944 | 1.000 |
| 0.120 | 0.186 | 0.000 | 0.997 | 1.000 |

## Interpretation

The CPU-local result is positive but not simplistic. Active repeated
measurement improves stability at small budgets, but high-noise correctness
still needs either better statistical calibration or more uniform measurement.
The abstention result is cleaner: when the margin is overwhelmed by noise, the
system can preserve correctness on covered labels by refusing to overclaim.
