# Evaluation report -- r2_W1_s43

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4290        0.4053        0.4512 0.8254    0.8141    0.8368 0.1895   1.0000         0
     B 68575       1336      0.0195     0.4237        0.3994        0.4486 0.8180    0.8055    0.8307 0.1640   1.0000         0
    S7 40867       1259      0.0308     0.3844        0.3580        0.4098 0.7817    0.7674    0.7968 0.1995   1.0000         0
   S10 26435         77      0.0029     0.3766        0.2658        0.4928 0.7406    0.6773    0.8023 0.0179   1.0000         0
  S10+ 26598        240      0.0090     0.4667        0.4105        0.5277 0.8264    0.7959    0.8546 0.0942   1.0000         0
   A·C 73300       1662      0.0227     0.4260        0.4038        0.4497 0.8212    0.8092    0.8328 0.1871   1.0000         0
   B·C 68575       1336      0.0195     0.4199        0.3936        0.4451 0.8118    0.7988    0.8250 0.1608   1.0000         0
  S7·C 40867       1259      0.0308     0.3813        0.3556        0.4062 0.7778    0.7627    0.7937 0.1984   1.0000         0
 S10·C 26435         77      0.0029     0.3506        0.2467        0.4512 0.7271    0.6637    0.7918 0.0158   1.0000         0
S10+·C 26598        240      0.0090     0.4625        0.4052        0.5240 0.8171    0.7860    0.8469 0.0928   1.0000         0
```

**C - A gap: -0.0030** (robust); A = 0.4290, C = 0.4260.
**S7·C - S7 gap: -0.0032** (the C-A gap with the length cue removed).

## Length-shortcut diagnostics

Duration-only baseline (what record length alone scores):
```
view     n  challenge  auroc
   A 73300     0.0756 0.6465
   B 68575     0.0790 0.6767
```

Length-cue gap: mean predicted probability of CODE-15 negatives by length group; `gap = short - long` should be ~0 for a model that ignores length:
```
view  neg_mean_short  neg_mean_mid  neg_mean_long    gap  neg_n_short  neg_n_long
   A          0.5630        0.4613         0.4734 0.0895   39608.0000  26358.0000
   C          0.5564        0.4672         0.4750 0.0814   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5299
   PTB-XL  4399     0.0000     0.0417
SaMi-Trop   326   326.0000     0.7238
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.4625     0.3846 0.6512
 (25.0, 33.0] 7152         30     0.4736     0.5333 0.8446
 (33.0, 41.0] 7710        148     0.4887     0.5946 0.8807
 (41.0, 48.0] 7177        203     0.5005     0.4483 0.8206
 (48.0, 55.0] 7453        226     0.4946     0.4513 0.8559
 (55.0, 61.0] 7071        254     0.4985     0.4882 0.8440
 (61.0, 67.0] 7198        223     0.5037     0.3677 0.8193
 (67.0, 73.0] 7367        217     0.5185     0.3687 0.7977
 (73.0, 79.0] 7312        190     0.5364     0.2684 0.7346
(79.0, 300.0] 7265        158     0.5397     0.3228 0.7259
```
