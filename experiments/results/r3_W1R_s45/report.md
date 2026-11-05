# Evaluation report -- r3_W1R_s45

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4134        0.3910        0.4367 0.8224    0.8108    0.8335 0.1917   1.0000         0
     B 68575       1336      0.0195     0.4184        0.3938        0.4440 0.8236    0.8122    0.8357 0.1730   1.0000         0
    S7 40867       1259      0.0308     0.3892        0.3658        0.4145 0.7884    0.7744    0.8037 0.2121   1.0000         0
   S10 26435         77      0.0029     0.3377        0.2353        0.4471 0.7662    0.7084    0.8180 0.0182   1.0000         0
  S10+ 26598        240      0.0090     0.4542        0.3937        0.5111 0.8152    0.7832    0.8439 0.0960   1.0000         0
   A·C 73300       1662      0.0227     0.4128        0.3897        0.4359 0.8198    0.8084    0.8316 0.1876   1.0000         0
   B·C 68575       1336      0.0195     0.4177        0.3897        0.4399 0.8194    0.8072    0.8325 0.1682   1.0000         0
  S7·C 40867       1259      0.0308     0.3924        0.3676        0.4167 0.7881    0.7742    0.8031 0.2101   1.0000         0
 S10·C 26435         77      0.0029     0.3377        0.2297        0.4416 0.7504    0.6869    0.8061 0.0179   1.0000         0
S10+·C 26598        240      0.0090     0.4542        0.3925        0.5143 0.8108    0.7773    0.8408 0.0939   1.0000         0
```

**C - A gap: -0.0006** (robust); A = 0.4134, C = 0.4128.
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
   A          0.6230        0.6120         0.6139 0.0090   39608.0000  26358.0000
   C          0.6220        0.6124         0.6142 0.0079   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6204
   PTB-XL  4399     0.0000     0.6111
SaMi-Trop   326   326.0000     0.6745
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.6097     0.2308 0.7544
 (25.0, 33.0] 7152         30     0.6121     0.5000 0.8180
 (33.0, 41.0] 7710        148     0.6152     0.5541 0.8678
 (41.0, 48.0] 7177        203     0.6177     0.4286 0.8326
 (48.0, 55.0] 7453        226     0.6197     0.4513 0.8410
 (55.0, 61.0] 7071        254     0.6224     0.4606 0.8413
 (61.0, 67.0] 7198        223     0.6231     0.3587 0.8111
 (67.0, 73.0] 7367        217     0.6252     0.3594 0.7659
 (73.0, 79.0] 7312        190     0.6273     0.2579 0.7196
(79.0, 300.0] 7265        158     0.6288     0.2848 0.7430
```
