# Evaluation report -- r2_W1_s42

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4122        0.3895        0.4367 0.8376    0.8273    0.8479 0.1825   1.0000         0
     B 68575       1336      0.0195     0.4192        0.3950        0.4439 0.8345    0.8227    0.8463 0.1670   1.0000         0
    S7 40867       1259      0.0308     0.3757        0.3501        0.4013 0.7981    0.7852    0.8126 0.1973   1.0000         0
   S10 26435         77      0.0029     0.3377        0.2286        0.4458 0.7752    0.7190    0.8259 0.0275   1.0000         0
  S10+ 26598        240      0.0090     0.4417        0.3790        0.4979 0.8326    0.8046    0.8617 0.0825   1.0000         0
   A·C 73300       1662      0.0227     0.4013        0.3772        0.4253 0.8341    0.8230    0.8446 0.1791   1.0000         0
   B·C 68575       1336      0.0195     0.4087        0.3822        0.4332 0.8291    0.8175    0.8408 0.1635   1.0000         0
  S7·C 40867       1259      0.0308     0.3709        0.3458        0.3958 0.7971    0.7838    0.8118 0.1987   1.0000         0
 S10·C 26435         77      0.0029     0.3117        0.2077        0.4167 0.7562    0.6967    0.8103 0.0164   1.0000         0
S10+·C 26598        240      0.0090     0.4208        0.3602        0.4781 0.8253    0.7968    0.8543 0.0751   1.0000         0
```

**C - A gap: -0.0108** (robust); A = 0.4122, C = 0.4013.
**S7·C - S7 gap: -0.0048** (the C-A gap with the length cue removed).

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
   A          0.6317        0.5217         0.5346 0.0971   39608.0000  26358.0000
   C          0.6256        0.5285         0.5393 0.0863   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5954
   PTB-XL  4399     0.0000     0.1237
SaMi-Trop   326   326.0000     0.7689
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.4990     0.2308 0.8348
 (25.0, 33.0] 7152         30     0.5163     0.5667 0.8492
 (33.0, 41.0] 7710        148     0.5364     0.5473 0.8776
 (41.0, 48.0] 7177        203     0.5555     0.4187 0.8361
 (48.0, 55.0] 7453        226     0.5543     0.4602 0.8493
 (55.0, 61.0] 7071        254     0.5666     0.4606 0.8566
 (61.0, 67.0] 7198        223     0.5794     0.3587 0.8235
 (67.0, 73.0] 7367        217     0.6025     0.3226 0.7744
 (73.0, 79.0] 7312        190     0.6312     0.3053 0.7466
(79.0, 300.0] 7265        158     0.6412     0.2658 0.7781
```
