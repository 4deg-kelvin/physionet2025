# Evaluation report -- r2_M1_s44

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4085        0.3882        0.4327 0.8063    0.7931    0.8189 0.1590   1.0000         0
     B 68575       1336      0.0195     0.3937        0.3697        0.4207 0.7913    0.7770    0.8051 0.1231   1.0000         0
    S7 40867       1259      0.0308     0.3844        0.3614        0.4094 0.7781    0.7628    0.7939 0.1879   1.0000         0
   S10 26435         77      0.0029     0.2468        0.1594        0.3514 0.6958    0.6261    0.7668 0.0165   1.0000         0
  S10+ 26598        240      0.0090     0.4500        0.3882        0.5135 0.7863    0.7494    0.8219 0.0984   1.0000         0
   A·C 73300       1662      0.0227     0.4061        0.3860        0.4309 0.8074    0.7946    0.8198 0.1597   1.0000         0
   B·C 68575       1336      0.0195     0.3922        0.3675        0.4178 0.7922    0.7782    0.8059 0.1235   1.0000         0
  S7·C 40867       1259      0.0308     0.3860        0.3626        0.4087 0.7803    0.7648    0.7957 0.1890   1.0000         0
 S10·C 26435         77      0.0029     0.2338        0.1519        0.3377 0.6743    0.6062    0.7464 0.0157   1.0000         0
S10+·C 26598        240      0.0090     0.4458        0.3872        0.5148 0.7816    0.7448    0.8179 0.0979   1.0000         0
```

**C - A gap: -0.0024** (robust); A = 0.4085, C = 0.4061.
**S7·C - S7 gap: +0.0016** (the C-A gap with the length cue removed).

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
   A          0.5065        0.4719         0.4596 0.0469   39608.0000  26358.0000
   C          0.5036        0.4718         0.4556 0.0480   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.4917
   PTB-XL  4399     0.0000     0.3253
SaMi-Trop   326   326.0000     0.7383
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.4588     0.2308 0.7040
 (25.0, 33.0] 7152         30     0.4600     0.4333 0.7895
 (33.0, 41.0] 7710        148     0.4689     0.6014 0.8667
 (41.0, 48.0] 7177        203     0.4819     0.4680 0.8482
 (48.0, 55.0] 7453        226     0.4800     0.4602 0.8524
 (55.0, 61.0] 7071        254     0.4852     0.4213 0.8341
 (61.0, 67.0] 7198        223     0.4877     0.3318 0.7987
 (67.0, 73.0] 7367        217     0.4950     0.3088 0.7603
 (73.0, 79.0] 7312        190     0.5018     0.2211 0.7007
(79.0, 300.0] 7265        158     0.5107     0.2152 0.7375
```
