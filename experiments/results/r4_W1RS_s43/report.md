# Evaluation report -- r4_W1RS_s43

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4194        0.3971        0.4428 0.8240    0.8134    0.8344 0.1764   1.0000         0
     B 68575       1336      0.0195     0.4266        0.4008        0.4536 0.8291    0.8172    0.8405 0.1585   1.0000         0
    S7 40867       1259      0.0308     0.3725        0.3502        0.3988 0.7980    0.7845    0.8116 0.1906   1.0000         0
   S10 26435         77      0.0029     0.3247        0.2162        0.4286 0.7369    0.6745    0.7965 0.0166   1.0000         0
  S10+ 26598        240      0.0090     0.4375        0.3765        0.5020 0.8301    0.8014    0.8561 0.0951   1.0000         0
   A·C 73300       1662      0.0227     0.4176        0.3942        0.4399 0.8190    0.8086    0.8296 0.1724   1.0000         0
   B·C 68575       1336      0.0195     0.4207        0.3949        0.4479 0.8214    0.8091    0.8331 0.1528   1.0000         0
  S7·C 40867       1259      0.0308     0.3717        0.3481        0.3966 0.7944    0.7807    0.8076 0.1872   1.0000         0
 S10·C 26435         77      0.0029     0.3247        0.2286        0.4384 0.7228    0.6593    0.7839 0.0153   1.0000         0
S10+·C 26598        240      0.0090     0.4625        0.3925        0.5184 0.8293    0.7997    0.8556 0.0920   1.0000         0
```

**C - A gap: -0.0018** (robust); A = 0.4194, C = 0.4176.
**S7·C - S7 gap: -0.0008** (the C-A gap with the length cue removed).

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
   A          0.6070        0.6004         0.6010 0.0060   39608.0000  26358.0000
   C          0.6067        0.6009         0.6014 0.0053   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6058
   PTB-XL  4399     0.0000     0.6056
SaMi-Trop   326   326.0000     0.6708
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5969     0.3077 0.8263
 (25.0, 33.0] 7152         30     0.5993     0.4667 0.7924
 (33.0, 41.0] 7710        148     0.6023     0.5135 0.8468
 (41.0, 48.0] 7177        203     0.6050     0.4631 0.8306
 (48.0, 55.0] 7453        226     0.6066     0.4690 0.8479
 (55.0, 61.0] 7071        254     0.6085     0.4252 0.8259
 (61.0, 67.0] 7198        223     0.6085     0.3408 0.7982
 (67.0, 73.0] 7367        217     0.6109     0.3456 0.7712
 (73.0, 79.0] 7312        190     0.6109     0.2842 0.7270
(79.0, 300.0] 7265        158     0.6127     0.2595 0.7153
```
