| dataset | method | buffer | variant | n | old-class errors predicted as current task (%) | uniform-error baseline (%) | task-wise TS gain in AUROC_all |
|---|---|---|---|---|---|---|---|
| seq-cifar10 | derpp | 500 | amp | 3 | 34.8 | 22.2 | -13.09 |
| seq-cifar10 | derpp | 500 | amp+lr@35/45 | 3 | 37.9 | 22.2 | -10.18 |
| seq-cifar10 | derpp | 500 | fp32 | 2 | 46.1 | 22.2 | -3.79 |
| seq-cifar10 | er | 500 | amp+lr@35/45 | 3 | 71.8 | 22.2 | +6.27 |
| seq-cifar10 | er | 500 | fp32 | 2 | 73.2 | 22.2 | +7.07 |
| seq-cifar100 | derpp | 500 | amp | 3 | 54.2 | 10.1 | +4.93 |
| seq-cifar100 | derpp | 2000 | amp | 3 | 38.2 | 10.1 | -1.32 |
| seq-cifar100 | er | 500 | amp | 3 | 88.4 | 10.1 | +20.65 |
| seq-cifar100 | er | 2000 | amp | 3 | 72.9 | 10.1 | +12.24 |

Spearman(recency share, task-wise TS gain) over 25 runs: rho = +0.95, p = 0.0000
