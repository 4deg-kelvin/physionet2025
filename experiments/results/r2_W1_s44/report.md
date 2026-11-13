# Evaluation report -- r2_W1_s44

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 patient-level resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4422        0.4127        0.4700 0.8440    0.8312    0.8559 0.1866   1.0000         0
     B 68575       1336      0.0195     0.4401        0.4055        0.4730 0.8347    0.8180    0.8501 0.1592   1.0000         0
    S7 40867       1259      0.0308     0.3805        0.3510        0.4120 0.8053    0.7876    0.8232 0.1944   1.0000         0
   S10 26435         77      0.0029     0.2987        0.2059        0.4334 0.7752    0.7221    0.8288 0.0187   1.0000         0
  S10+ 26598        240      0.0090     0.4708        0.4053        0.5398 0.8464    0.8196    0.8732 0.1002   1.0000         0
   A·C 73300       1662      0.0227     0.4386        0.4089        0.4644 0.8409    0.8282    0.8534 0.1850   1.0000         0
   B·C 68575       1336      0.0195     0.4326        0.3995        0.4657 0.8289    0.8119    0.8445 0.1565   1.0000         0
  S7·C 40867       1259      0.0308     0.3805        0.3486        0.4117 0.8041    0.7855    0.8217 0.1959   1.0000         0
 S10·C 26435         77      0.0029     0.2987        0.1899        0.4322 0.7412    0.6798    0.8034 0.0140   1.0000         0
S10+·C 26598        240      0.0090     0.4708        0.3983        0.5362 0.8368    0.8063    0.8663 0.0966   1.0000         0
```

**C - A gap: -0.0036** (robust); A = 0.4422, C = 0.4386.
**S7·C - S7 gap: +0.0000** (the C-A gap with the length cue removed).

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
   A          0.5772        0.4678         0.4722 0.1050   39608.0000  26358.0000
   C          0.5707        0.4722         0.4774 0.0933   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5390
   PTB-XL  4399     0.0000     0.0478
SaMi-Trop   326   326.0000     0.7874
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.4455     0.4615 0.7553
 (25.0, 33.0] 7152         30     0.4607     0.5333 0.8408
 (33.0, 41.0] 7710        148     0.4802     0.6081 0.9040
 (41.0, 48.0] 7177        203     0.5024     0.4483 0.8495
 (48.0, 55.0] 7453        226     0.5003     0.4513 0.8643
 (55.0, 61.0] 7071        254     0.5122     0.4528 0.8591
 (61.0, 67.0] 7198        223     0.5215     0.3587 0.8470
 (67.0, 73.0] 7367        217     0.5429     0.3502 0.7837
 (73.0, 79.0] 7312        190     0.5667     0.2947 0.7509
(79.0, 300.0] 7265        158     0.5769     0.2342 0.7625
```
