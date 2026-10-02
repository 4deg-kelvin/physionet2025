# Evaluation report -- r2_MW_s42 (selected alpha0.50)

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.3809        0.3580        0.4026 0.8146    0.8040    0.8259 0.1612   1.0000         0
     B 68575       1336      0.0195     0.3765        0.3510        0.4024 0.8033    0.7901    0.8165 0.1385   1.0000         0
    S7 40867       1259      0.0308     0.3678        0.3434        0.3937 0.8018    0.7892    0.8158 0.1994   1.0000         0
   S10 26435         77      0.0029     0.3506        0.2500        0.4578 0.7580    0.6960    0.8123 0.0177   1.0000         0
  S10+ 26598        240      0.0090     0.4417        0.3810        0.5060 0.8324    0.8050    0.8583 0.0883   1.0000         0
   A·C 73300       1662      0.0227     0.3779        0.3568        0.4004 0.8144    0.8037    0.8262 0.1625   1.0000         0
   B·C 68575       1336      0.0195     0.3750        0.3498        0.4011 0.8030    0.7900    0.8163 0.1401   1.0000         0
  S7·C 40867       1259      0.0308     0.3709        0.3465        0.3970 0.8023    0.7897    0.8161 0.2005   1.0000         0
 S10·C 26435         77      0.0029     0.3506        0.2381        0.4533 0.7559    0.6925    0.8123 0.0176   1.0000         0
S10+·C 26598        240      0.0090     0.4375        0.3755        0.4978 0.8323    0.8047    0.8586 0.0901   1.0000         0
```

**C - A gap: -0.0030** (robust); A = 0.3809, C = 0.3779.
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
   A          0.6096        0.5976         0.6039 0.0058   39608.0000  26358.0000
   C          0.6083        0.5973         0.6033 0.0050   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6102
   PTB-XL  4399     0.0000     0.5792
SaMi-Trop   326   326.0000     0.7844
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5458     0.2308 0.6967
 (25.0, 33.0] 7152         30     0.5579     0.4667 0.7764
 (33.0, 41.0] 7710        148     0.5704     0.5135 0.8655
 (41.0, 48.0] 7177        203     0.5861     0.4335 0.8221
 (48.0, 55.0] 7453        226     0.5994     0.4248 0.8249
 (55.0, 61.0] 7071        254     0.6148     0.4370 0.8515
 (61.0, 67.0] 7198        223     0.6252     0.3049 0.8033
 (67.0, 73.0] 7367        217     0.6461     0.3088 0.7339
 (73.0, 79.0] 7312        190     0.6647     0.2474 0.7253
(79.0, 300.0] 7265        158     0.6845     0.2975 0.7519
```
