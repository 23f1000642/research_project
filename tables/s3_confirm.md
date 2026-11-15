14/14 planned runs found

| run | recency share | AUROC_all none / ts / task_ts | ECE_all none / ts / task_ts |
|---|---|---|---|
| seq-cifar100-er-b500-s3-448e1569 | 91.9 | 66.8 / 68.2 / 88.6 | 65.4 / 51.8 / 56.4 |
| seq-cifar100-derpp-b500-s3-5eae9ae1 | 62.5 | 74.3 / 74.1 / 82.4 | 19.7 / 29.9 / 46.7 |
| seq-cifar100-er-b2000-s3-f91d8385 | 79.0 | 73.8 / 75.2 / 88.2 | 46.5 / 36.0 / 43.6 |
| seq-cifar100-derpp-b2000-s3-9a1c22a4 | 44.3 | 81.3 / 81.3 / 80.5 | 16.6 / 16.9 / 34.7 |
| seq-cifar100-er-b500-s4-69956daa | 88.2 | 66.9 / 68.4 / 88.8 | 61.3 / 44.2 / 51.1 |
| seq-cifar100-derpp-b500-s4-2303d83e | 57.5 | 75.0 / 75.0 / 81.4 | 19.0 / 26.7 / 44.7 |
| seq-cifar100-er-b2000-s4-eac70d7f | 71.3 | 74.7 / 76.0 / 86.4 | 44.9 / 32.5 / 42.5 |
| seq-cifar100-derpp-b2000-s4-28388ffe | 60.8 | 76.8 / 76.8 / 81.7 | 24.6 / 20.1 / 34.9 |
| seq-cifar100-mir-b2000-s0-a67eb829 | 70.3 | 74.9 / 76.2 / 86.0 | 43.8 / 30.6 / 41.1 |
| seq-cifar100-icarl-b2000-s0-7ddd3c3e | 10.2 | 75.7 / 77.6 / 76.9 | 24.7 / 8.9 / 12.4 |
| seq-cifar100-mir-b2000-s1-5e909996 | 75.5 | 73.5 / 74.7 / 87.1 | 46.2 / 33.6 / 42.8 |
| seq-cifar100-icarl-b2000-s1-d71809c7 | 11.5 | 75.8 / 77.5 / 77.0 | 24.3 / 8.2 / 11.4 |
| seq-cifar100-mir-b2000-s2-44a40a9f | 74.4 | 73.8 / 75.1 / 87.3 | 44.4 / 32.0 / 40.9 |
| seq-cifar100-icarl-b2000-s2-a6fcc07d | 12.9 | 75.2 / 76.4 / 76.2 | 23.1 / 9.4 / 11.9 |

C1 global TS, AUROC_all change +1.02 (TOST p = 0.5331): NOT CONFIRMED
C2 task-wise TS worsens ECE_all in 4/14 runs (sign test p = 0.9713): NOT CONFIRMED
C3 Spearman(recency share, task-wise TS gain) = +0.95 (p = 0.0000): CONFIRMED
C4 oracle global TS: ECE_old +83.0 % lower, AUROC_old +3.04: NOT CONFIRMED
