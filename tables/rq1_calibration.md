# RQ1: post-hoc calibration vs. failure detection on old classes (25 runs)

Pooled old-class test samples after the last task; mean over seeds. ECE, AUROC, AURC, sel. risk @90 % coverage in %.

| dataset | method | buffer | variant | n | acc old | calibrator | conf old | ECE old | AUROC old | AURC old | sel90 old | AUROC all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | none | 86.6 | 19.41 | 77.23 | 16.07 | 28.75 | 80.17 |
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | ts | 79.9 | 12.84 | 77.23 | 16.07 | 28.74 | 80.17 |
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | task_ts | 95.4 | 28.15 | 78.67 | 20.24 | 26.06 | 67.08 |
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | oldnew_ts | 95.4 | 28.16 | 78.41 | 20.43 | 26.03 | 66.81 |
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | oracle_ts | 67.1 | 4.17 | 77.11 | 16.10 | 28.73 | nan |
| seq-cifar10 | derpp | 500 | amp | 3 | 67.2 | oracle_task_ts | 68.4 | 3.66 | 87.29 | 11.47 | 26.15 | nan |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | none | 86.1 | 19.75 | 75.99 | 18.05 | 29.53 | 79.76 |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | ts | 81.0 | 14.63 | 75.88 | 18.08 | 29.50 | 79.69 |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | task_ts | 95.5 | 29.12 | 80.92 | 18.46 | 27.06 | 69.58 |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | oldnew_ts | 95.6 | 29.15 | 80.28 | 19.14 | 27.03 | 69.00 |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | oracle_ts | 65.5 | 4.97 | 75.55 | 18.19 | 29.52 | nan |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 66.4 | oracle_task_ts | 67.3 | 3.83 | 88.08 | 11.59 | 26.85 | nan |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | none | 85.5 | 22.51 | 73.71 | 24.26 | 33.12 | 76.52 |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | ts | 75.9 | 12.86 | 73.61 | 24.35 | 33.16 | 76.46 |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | task_ts | 91.4 | 28.36 | 82.34 | 20.63 | 30.69 | 72.73 |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | oldnew_ts | 91.2 | 28.22 | 83.51 | 17.50 | 30.69 | 73.39 |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | oracle_ts | 62.2 | 5.86 | 73.36 | 24.51 | 33.12 | nan |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 63.0 | oracle_task_ts | 64.8 | 4.84 | 90.51 | 12.72 | 30.44 | nan |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | none | 93.1 | 43.46 | 71.45 | 31.30 | 48.14 | 78.18 |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | ts | 83.6 | 34.03 | 72.23 | 30.85 | 47.76 | 78.84 |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | task_ts | 91.7 | 42.14 | 90.31 | 23.41 | 45.32 | 84.45 |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | oldnew_ts | 91.7 | 42.14 | 90.31 | 23.41 | 45.32 | 84.45 |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | oracle_ts | 53.6 | 5.59 | 75.31 | 28.98 | 47.43 | nan |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 49.6 | oracle_task_ts | 55.9 | 10.88 | 94.07 | 19.21 | 44.91 | nan |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | none | 93.8 | 42.33 | 72.06 | 29.41 | 46.18 | 77.63 |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | ts | 84.4 | 32.96 | 72.88 | 28.93 | 45.90 | 78.33 |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | task_ts | 92.3 | 40.89 | 91.49 | 20.56 | 43.15 | 84.70 |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | oldnew_ts | 92.4 | 40.97 | 91.04 | 21.38 | 43.11 | 84.18 |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | oracle_ts | 55.0 | 5.09 | 76.26 | 26.81 | 45.47 | nan |
| seq-cifar10 | er | 500 | fp32 | 2 | 51.5 | oracle_task_ts | 57.2 | 10.29 | 94.46 | 17.73 | 42.86 | nan |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | none | 51.2 | 19.26 | 72.90 | 50.23 | 65.66 | 75.83 |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | ts | 58.5 | 26.50 | 72.85 | 50.23 | 65.66 | 75.80 |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | task_ts | 80.8 | 48.81 | 86.79 | 41.85 | 64.47 | 80.76 |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | oldnew_ts | 81.5 | 49.48 | 87.12 | 41.94 | 64.47 | 80.43 |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | oracle_ts | 36.0 | 7.59 | 72.87 | 50.29 | 65.67 | nan |
| seq-cifar100 | derpp | 500 | amp | 3 | 32.0 | oracle_task_ts | 37.2 | 5.32 | 85.19 | 41.93 | 64.56 | nan |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | none | 64.7 | 15.61 | 80.08 | 26.92 | 46.98 | 80.95 |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | ts | 65.5 | 16.34 | 80.07 | 26.92 | 46.98 | 80.95 |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | task_ts | 88.5 | 39.38 | 85.54 | 25.41 | 45.48 | 79.63 |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | oldnew_ts | 89.1 | 39.92 | 84.95 | 26.36 | 45.46 | 78.62 |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | oracle_ts | 52.3 | 6.38 | 80.12 | 26.92 | 47.00 | nan |
| seq-cifar100 | derpp | 2000 | amp | 3 | 49.1 | oracle_task_ts | 52.8 | 4.47 | 85.18 | 24.27 | 46.08 | nan |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | none | 83.0 | 68.44 | 53.61 | 84.61 | 84.95 | 67.35 |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | ts | 63.6 | 49.03 | 55.93 | 83.95 | 84.76 | 69.03 |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | task_ts | 70.7 | 56.14 | 92.99 | 63.54 | 83.83 | 87.99 |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | oldnew_ts | 70.9 | 56.32 | 93.80 | 63.29 | 83.83 | 88.25 |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | oracle_ts | 23.2 | 8.65 | 60.68 | 82.24 | 84.60 | nan |
| seq-cifar100 | er | 500 | amp | 3 | 14.6 | oracle_task_ts | 24.7 | 10.31 | 91.25 | 64.14 | 83.83 | nan |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | none | 83.1 | 49.30 | 69.94 | 49.73 | 63.95 | 73.95 |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | ts | 69.8 | 35.97 | 71.44 | 48.95 | 63.71 | 75.21 |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | task_ts | 80.6 | 46.81 | 91.44 | 38.46 | 62.47 | 86.20 |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | oldnew_ts | 80.8 | 47.00 | 91.21 | 39.41 | 62.47 | 85.81 |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | oracle_ts | 36.9 | 3.51 | 73.69 | 47.68 | 63.51 | nan |
| seq-cifar100 | er | 2000 | amp | 3 | 33.8 | oracle_task_ts | 39.4 | 6.75 | 90.79 | 36.03 | 62.47 | nan |

## Pre-registered decision (paired over all runs vs. `none`)

- `ts`: ECE_old relative reduction +17.2 % (runs improved: 20/25); AUROC_old change +0.59 (range -0.35 .. +2.81; p(change > 1) = 0.986) -> no calibration gain
- `task_ts`: ECE_old relative reduction -49.6 % (runs improved: 11/25); AUROC_old change +14.90 (range -3.76 .. +39.86; p(change > 1) = 0.000) -> detection improves
- `oldnew_ts`: ECE_old relative reduction -50.5 % (runs improved: 11/25); AUROC_old change +14.89 (range -4.46 .. +40.88; p(change > 1) = 0.000) -> detection improves
- `oracle_ts`: ECE_old relative reduction +77.8 % (runs improved: 25/25); AUROC_old change +2.01 (range -1.07 .. +7.92; p(change > 1) = 0.039) -> detection improves  [upper bound, not a method]
- `oracle_task_ts`: ECE_old relative reduction +78.5 % (runs improved: 25/25); AUROC_old change +17.62 (range +3.54 .. +38.81; p(change > 1) = 0.000) -> detection improves  [upper bound, not a method]

Claim 'calibration and failure detection come apart': NOT SUPPORTED (0/3 legal calibrators)
