# Evaluation report -- r4_W1RS_s42

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4049        0.3823        0.4261 0.7816    0.7684    0.7951 0.1741   1.0000         0
     B 68575       1336      0.0195     0.4087        0.3831        0.4352 0.7941    0.7802    0.8071 0.1603   1.0000         0
    S7 40867       1259      0.0308     0.3828        0.3583        0.4081 0.7534    0.7386    0.7696 0.1931   1.0000         0
   S10 26435         77      0.0029     0.3247        0.2093        0.4263 0.6807    0.6115    0.7457 0.0154   1.0000         0
  S10+ 26598        240      0.0090     0.4083        0.3476        0.4696 0.7690    0.7332    0.8002 0.0777   1.0000         0
   A·C 73300       1662      0.0227     0.4019        0.3773        0.4235 0.7772    0.7638    0.7907 0.1698   1.0000         0
   B·C 68575       1336      0.0195     0.4019        0.3769        0.4289 0.7858    0.7717    0.7988 0.1543   1.0000         0
  S7·C 40867       1259      0.0308     0.3781        0.3535        0.4044 0.7504    0.7353    0.7672 0.1909   1.0000         0
 S10·C 26435         77      0.0029     0.2857        0.1899        0.3968 0.6725    0.6034    0.7374 0.0154   1.0000         0
S10+·C 26598        240      0.0090     0.4042        0.3401        0.4667 0.7711    0.7379    0.8027 0.0765   1.0000         0
```

**C - A gap: -0.0030** (robust); A = 0.4049, C = 0.4019.
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
   A          0.6173        0.6110         0.6118 0.0055   39608.0000  26358.0000
   C          0.6170        0.6114         0.6123 0.0047   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6160
   PTB-XL  4399     0.0000     0.6157
SaMi-Trop   326   326.0000     0.6613
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.6095     0.3846 0.6900
 (25.0, 33.0] 7152         30     0.6106     0.5333 0.8013
 (33.0, 41.0] 7710        148     0.6128     0.4865 0.8250
 (41.0, 48.0] 7177        203     0.6147     0.4138 0.7635
 (48.0, 55.0] 7453        226     0.6166     0.4381 0.8019
 (55.0, 61.0] 7071        254     0.6184     0.4685 0.7944
 (61.0, 67.0] 7198        223     0.6185     0.3408 0.7692
 (67.0, 73.0] 7367        217     0.6199     0.3594 0.7506
 (73.0, 79.0] 7312        190     0.6205     0.2737 0.6825
(79.0, 300.0] 7265        158     0.6210     0.2468 0.6730
```
