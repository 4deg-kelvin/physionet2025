# Evaluation report -- r4_W1RS_s43

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 patient-level resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4194        0.3918        0.4452 0.8240    0.8112    0.8359 0.1764   1.0000         0
     B 68575       1336      0.0195     0.4266        0.3942        0.4598 0.8291    0.8141    0.8432 0.1585   1.0000         0
    S7 40867       1259      0.0308     0.3725        0.3398        0.4053 0.7980    0.7816    0.8144 0.1906   1.0000         0
   S10 26435         77      0.0029     0.3247        0.2054        0.4384 0.7369    0.6671    0.8015 0.0166   1.0000         0
  S10+ 26598        240      0.0090     0.4375        0.3796        0.5041 0.8301    0.8013    0.8604 0.0951   1.0000         0
   A·C 73300       1662      0.0227     0.4176        0.3872        0.4424 0.8190    0.8061    0.8310 0.1724   1.0000         0
   B·C 68575       1336      0.0195     0.4207        0.3881        0.4557 0.8214    0.8061    0.8352 0.1528   1.0000         0
  S7·C 40867       1259      0.0308     0.3717        0.3379        0.4036 0.7944    0.7781    0.8109 0.1872   1.0000         0
 S10·C 26435         77      0.0029     0.3247        0.2073        0.4421 0.7228    0.6503    0.7925 0.0153   1.0000         0
S10+·C 26598        240      0.0090     0.4625        0.3969        0.5277 0.8293    0.8005    0.8600 0.0920   1.0000         0
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
