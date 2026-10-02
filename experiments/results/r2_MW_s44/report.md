# Evaluation report -- r2_MW_s44 (selected alpha1.00)

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.3977        0.3751        0.4211 0.8152    0.8036    0.8264 0.1668   1.0000         0
     B 68575       1336      0.0195     0.3885        0.3636        0.4130 0.7986    0.7848    0.8121 0.1353   1.0000         0
    S7 40867       1259      0.0308     0.3662        0.3423        0.3912 0.7948    0.7806    0.8091 0.1942   1.0000         0
   S10 26435         77      0.0029     0.3636        0.2615        0.4750 0.7661    0.7084    0.8203 0.0165   1.0000         0
  S10+ 26598        240      0.0090     0.4667        0.4033        0.5253 0.8406    0.8116    0.8670 0.0943   1.0000         0
   A·C 73300       1662      0.0227     0.3917        0.3685        0.4139 0.8162    0.8050    0.8274 0.1659   1.0000         0
   B·C 68575       1336      0.0195     0.3817        0.3574        0.4065 0.7986    0.7848    0.8120 0.1341   1.0000         0
  S7·C 40867       1259      0.0308     0.3693        0.3444        0.3946 0.7963    0.7824    0.8107 0.1943   1.0000         0
 S10·C 26435         77      0.0029     0.3117        0.2235        0.4348 0.7622    0.7043    0.8169 0.0160   1.0000         0
S10+·C 26598        240      0.0090     0.4542        0.3926        0.5156 0.8426    0.8147    0.8681 0.0942   1.0000         0
```

**C - A gap: -0.0060** (robust); A = 0.3977, C = 0.3917.
**S7·C - S7 gap: +0.0032** (the C-A gap with the length cue removed).

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
   A          0.5626        0.5341         0.5526 0.0100   39608.0000  26358.0000
   C          0.5586        0.5364         0.5507 0.0079   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5619
   PTB-XL  4399     0.0000     0.4963
SaMi-Trop   326   326.0000     0.7812
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.4898     0.2308 0.6295
 (25.0, 33.0] 7152         30     0.5089     0.5667 0.8333
 (33.0, 41.0] 7710        148     0.5296     0.5068 0.8617
 (41.0, 48.0] 7177        203     0.5509     0.4433 0.8132
 (48.0, 55.0] 7453        226     0.5617     0.4115 0.8313
 (55.0, 61.0] 7071        254     0.5739     0.4528 0.8406
 (61.0, 67.0] 7198        223     0.5779     0.3184 0.8184
 (67.0, 73.0] 7367        217     0.5894     0.3410 0.7553
 (73.0, 79.0] 7312        190     0.6005     0.2526 0.7074
(79.0, 300.0] 7265        158     0.6103     0.2532 0.7382
```
