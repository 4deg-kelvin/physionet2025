# Evaluation report -- r4_MWR_s43

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4146        0.3921        0.4384 0.8151    0.8040    0.8279 0.1774   1.0000         0
     B 68575       1336      0.0195     0.4079        0.3823        0.4346 0.8031    0.7897    0.8163 0.1447   1.0000         0
    S7 40867       1259      0.0308     0.3916        0.3671        0.4170 0.7969    0.7836    0.8118 0.2041   1.0000         0
   S10 26435         77      0.0029     0.3766        0.2742        0.4941 0.7735    0.7077    0.8311 0.0270   1.0000         0
  S10+ 26598        240      0.0090     0.4792        0.4154        0.5376 0.8496    0.8217    0.8749 0.1147   1.0000         0
   A·C 73300       1662      0.0227     0.4134        0.3897        0.4350 0.8141    0.8030    0.8261 0.1767   1.0000         0
   B·C 68575       1336      0.0195     0.4072        0.3808        0.4331 0.8015    0.7877    0.8148 0.1439   1.0000         0
  S7·C 40867       1259      0.0308     0.3916        0.3654        0.4154 0.7968    0.7836    0.8120 0.2039   1.0000         0
 S10·C 26435         77      0.0029     0.3636        0.2674        0.4857 0.7636    0.6941    0.8247 0.0255   1.0000         0
S10+·C 26598        240      0.0090     0.4792        0.4148        0.5381 0.8468    0.8186    0.8716 0.1127   1.0000         0
```

**C - A gap: -0.0012** (robust); A = 0.4146, C = 0.4134.
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
   A          0.6296        0.6267         0.6285 0.0011   39608.0000  26358.0000
   C          0.6297        0.6271         0.6287 0.0010   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6307
   PTB-XL  4399     0.0000     0.6261
SaMi-Trop   326   326.0000     0.7251
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.6159     0.3077 0.6902
 (25.0, 33.0] 7152         30     0.6190     0.4333 0.7585
 (33.0, 41.0] 7710        148     0.6229     0.5473 0.8552
 (41.0, 48.0] 7177        203     0.6267     0.4581 0.7711
 (48.0, 55.0] 7453        226     0.6300     0.4292 0.8107
 (55.0, 61.0] 7071        254     0.6343     0.4724 0.8469
 (61.0, 67.0] 7198        223     0.6350     0.3453 0.8051
 (67.0, 73.0] 7367        217     0.6387     0.3594 0.7584
 (73.0, 79.0] 7312        190     0.6420     0.2421 0.7404
(79.0, 300.0] 7265        158     0.6446     0.2785 0.7107
```
