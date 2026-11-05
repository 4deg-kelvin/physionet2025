# Evaluation report -- r4_MWR_s42

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4073        0.3845        0.4296 0.8179    0.8070    0.8286 0.1741   1.0000         0
     B 68575       1336      0.0195     0.4012        0.3748        0.4270 0.8061    0.7925    0.8187 0.1415   1.0000         0
    S7 40867       1259      0.0308     0.3852        0.3606        0.4099 0.7953    0.7814    0.8093 0.1940   1.0000         0
   S10 26435         77      0.0029     0.3506        0.2500        0.4604 0.7565    0.6900    0.8149 0.0183   1.0000         0
  S10+ 26598        240      0.0090     0.4667        0.4025        0.5257 0.8463    0.8188    0.8714 0.1084   1.0000         0
   A·C 73300       1662      0.0227     0.4079        0.3840        0.4300 0.8167    0.8059    0.8276 0.1728   1.0000         0
   B·C 68575       1336      0.0195     0.3982        0.3738        0.4242 0.8041    0.7907    0.8165 0.1392   1.0000         0
  S7·C 40867       1259      0.0308     0.3820        0.3575        0.4060 0.7950    0.7813    0.8091 0.1931   1.0000         0
 S10·C 26435         77      0.0029     0.3636        0.2500        0.4714 0.7477    0.6821    0.8067 0.0174   1.0000         0
S10+·C 26598        240      0.0090     0.4667        0.4034        0.5251 0.8442    0.8156    0.8684 0.1071   1.0000         0
```

**C - A gap: +0.0006** (robust); A = 0.4073, C = 0.4079.
**S7·C - S7 gap: -0.0032** (the C-A gap with the length cue removed).

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
   A          0.6175        0.6160         0.6162 0.0012   39608.0000  26358.0000
   C          0.6175        0.6161         0.6164 0.0011   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.6177
   PTB-XL  4399     0.0000     0.6153
SaMi-Trop   326   326.0000     0.6686
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.6112     0.2308 0.7383
 (25.0, 33.0] 7152         30     0.6128     0.4667 0.7767
 (33.0, 41.0] 7710        148     0.6149     0.5541 0.8519
 (41.0, 48.0] 7177        203     0.6167     0.4581 0.7999
 (48.0, 55.0] 7453        226     0.6186     0.4336 0.8212
 (55.0, 61.0] 7071        254     0.6201     0.4567 0.8357
 (61.0, 67.0] 7198        223     0.6199     0.3229 0.8053
 (67.0, 73.0] 7367        217     0.6212     0.3548 0.7565
 (73.0, 79.0] 7312        190     0.6213     0.2474 0.7171
(79.0, 300.0] 7265        158     0.6219     0.2532 0.7350
```
