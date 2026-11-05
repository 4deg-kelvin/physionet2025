# Evaluation report -- r4_W1RS_s44

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4224        0.4009        0.4461 0.8370    0.8269    0.8475 0.1901   1.0000         0
     B 68575       1336      0.0195     0.4289        0.4009        0.4537 0.8462    0.8350    0.8565 0.1733   1.0000         0
    S7 40867       1259      0.0308     0.3836        0.3597        0.4092 0.8091    0.7960    0.8226 0.2087   1.0000         0
   S10 26435         77      0.0029     0.4026        0.2976        0.5125 0.7750    0.7126    0.8282 0.0193   1.0000         0
  S10+ 26598        240      0.0090     0.4792        0.4215        0.5394 0.8341    0.8059    0.8596 0.0941   1.0000         0
   A·C 73300       1662      0.0227     0.4212        0.3976        0.4436 0.8339    0.8236    0.8445 0.1857   1.0000         0
   B·C 68575       1336      0.0195     0.4251        0.3988        0.4525 0.8398    0.8285    0.8504 0.1672   1.0000         0
  S7·C 40867       1259      0.0308     0.3813        0.3556        0.4065 0.8078    0.7948    0.8215 0.2072   1.0000         0
 S10·C 26435         77      0.0029     0.3896        0.2857        0.4941 0.7574    0.6976    0.8145 0.0181   1.0000         0
S10+·C 26598        240      0.0090     0.4792        0.4181        0.5375 0.8298    0.8010    0.8551 0.0903   1.0000         0
```

**C - A gap: -0.0012** (robust); A = 0.4224, C = 0.4212.
**S7·C - S7 gap: -0.0024** (the C-A gap with the length cue removed).

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
   A          0.6260        0.6167         0.6178 0.0082   39608.0000  26358.0000
   C          0.6257        0.6175         0.6186 0.0071   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6239
   PTB-XL  4399     0.0000     0.6231
SaMi-Trop   326   326.0000     0.6800
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.6132     0.3077 0.7873
 (25.0, 33.0] 7152         30     0.6160     0.5333 0.8282
 (33.0, 41.0] 7710        148     0.6193     0.5811 0.8621
 (41.0, 48.0] 7177        203     0.6219     0.4089 0.8235
 (48.0, 55.0] 7453        226     0.6244     0.4469 0.8578
 (55.0, 61.0] 7071        254     0.6272     0.4488 0.8519
 (61.0, 67.0] 7198        223     0.6273     0.3722 0.8210
 (67.0, 73.0] 7367        217     0.6294     0.3641 0.7798
 (73.0, 79.0] 7312        190     0.6307     0.2737 0.7399
(79.0, 300.0] 7265        158     0.6317     0.2911 0.7535
```
