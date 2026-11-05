# Evaluation report -- r4_MWR_s44

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.3965        0.3751        0.4211 0.8267    0.8157    0.8370 0.1695   1.0000         0
     B 68575       1336      0.0195     0.3907        0.3653        0.4170 0.8158    0.8031    0.8281 0.1358   1.0000         0
    S7 40867       1259      0.0308     0.3749        0.3495        0.3982 0.8070    0.7941    0.8200 0.1972   1.0000         0
   S10 26435         77      0.0029     0.3766        0.2763        0.4857 0.7722    0.7099    0.8279 0.0188   1.0000         0
  S10+ 26598        240      0.0090     0.4833        0.4228        0.5455 0.8484    0.8242    0.8740 0.1138   1.0000         0
   A·C 73300       1662      0.0227     0.3995        0.3770        0.4231 0.8241    0.8130    0.8346 0.1683   1.0000         0
   B·C 68575       1336      0.0195     0.3847        0.3601        0.4113 0.8121    0.7989    0.8245 0.1345   1.0000         0
  S7·C 40867       1259      0.0308     0.3757        0.3512        0.4003 0.8051    0.7919    0.8177 0.1964   1.0000         0
 S10·C 26435         77      0.0029     0.3506        0.2571        0.4699 0.7649    0.7014    0.8211 0.0176   1.0000         0
S10+·C 26598        240      0.0090     0.4833        0.4215        0.5494 0.8487    0.8242    0.8735 0.1087   1.0000         0
```

**C - A gap: +0.0030** (robust); A = 0.3965, C = 0.3995.
**S7·C - S7 gap: +0.0008** (the C-A gap with the length cue removed).

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
   A          0.6005        0.5986         0.5998 0.0007   39608.0000  26358.0000
   C          0.6006        0.5987         0.6000 0.0006   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6018
   PTB-XL  4399     0.0000     0.5977
SaMi-Trop   326   326.0000     0.7090
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5898     0.4615 0.8208
 (25.0, 33.0] 7152         30     0.5933     0.5333 0.7723
 (33.0, 41.0] 7710        148     0.5964     0.5405 0.8614
 (41.0, 48.0] 7177        203     0.6001     0.4335 0.8087
 (48.0, 55.0] 7453        226     0.6032     0.4292 0.8332
 (55.0, 61.0] 7071        254     0.6062     0.4528 0.8555
 (61.0, 67.0] 7198        223     0.6052     0.3004 0.8076
 (67.0, 73.0] 7367        217     0.6080     0.3226 0.7568
 (73.0, 79.0] 7312        190     0.6085     0.2737 0.7171
(79.0, 300.0] 7265        158     0.6103     0.2658 0.7437
```
