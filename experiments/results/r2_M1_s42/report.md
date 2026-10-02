# Evaluation report -- r2_M1_s42

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4043        0.3811        0.4271 0.8255    0.8153    0.8356 0.1651   1.0000         0
     B 68575       1336      0.0195     0.3960        0.3714        0.4225 0.8142    0.8016    0.8267 0.1390   1.0000         0
    S7 40867       1259      0.0308     0.3828        0.3571        0.4076 0.8022    0.7879    0.8161 0.2005   1.0000         0
   S10 26435         77      0.0029     0.3766        0.2740        0.5000 0.7554    0.6884    0.8126 0.0257   1.0000         0
  S10+ 26598        240      0.0090     0.4875        0.4274        0.5492 0.8366    0.8068    0.8628 0.0943   1.0000         0
   A·C 73300       1662      0.0227     0.4025        0.3814        0.4247 0.8275    0.8179    0.8377 0.1651   1.0000         0
   B·C 68575       1336      0.0195     0.3990        0.3733        0.4225 0.8170    0.8045    0.8295 0.1395   1.0000         0
  S7·C 40867       1259      0.0308     0.3805        0.3561        0.4054 0.8031    0.7889    0.8169 0.1992   1.0000         0
 S10·C 26435         77      0.0029     0.3766        0.2667        0.4858 0.7533    0.6845    0.8106 0.0184   1.0000         0
S10+·C 26598        240      0.0090     0.4875        0.4279        0.5485 0.8367    0.8080    0.8632 0.0892   1.0000         0
```

**C - A gap: -0.0018** (robust); A = 0.4043, C = 0.4025.
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
   A          0.5964        0.5542         0.5645 0.0319   39608.0000  26358.0000
   C          0.5931        0.5540         0.5561 0.0370   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5867
   PTB-XL  4399     0.0000     0.5242
SaMi-Trop   326   326.0000     0.7917
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5188     0.2308 0.6460
 (25.0, 33.0] 7152         30     0.5332     0.5333 0.8093
 (33.0, 41.0] 7710        148     0.5487     0.5068 0.8754
 (41.0, 48.0] 7177        203     0.5657     0.4433 0.8393
 (48.0, 55.0] 7453        226     0.5759     0.4381 0.8403
 (55.0, 61.0] 7071        254     0.5902     0.4528 0.8533
 (61.0, 67.0] 7198        223     0.5987     0.3274 0.8160
 (67.0, 73.0] 7367        217     0.6187     0.2995 0.7542
 (73.0, 79.0] 7312        190     0.6364     0.2579 0.7385
(79.0, 300.0] 7265        158     0.6559     0.2722 0.7484
```
