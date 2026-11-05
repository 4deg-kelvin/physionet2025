# Evaluation report -- r3_W1R_s46

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4398        0.4151        0.4608 0.8353    0.8255    0.8457 0.1932   1.0000         0
     B 68575       1336      0.0195     0.4386        0.4105        0.4634 0.8318    0.8199    0.8429 0.1664   1.0000         0
    S7 40867       1259      0.0308     0.3916        0.3663        0.4173 0.8004    0.7874    0.8150 0.2043   1.0000         0
   S10 26435         77      0.0029     0.3117        0.2152        0.4198 0.7400    0.6771    0.7982 0.0166   1.0000         0
  S10+ 26598        240      0.0090     0.4292        0.3696        0.4895 0.8281    0.7997    0.8550 0.0969   1.0000         0
   A·C 73300       1662      0.0227     0.4302        0.4069        0.4525 0.8287    0.8180    0.8391 0.1913   1.0000         0
   B·C 68575       1336      0.0195     0.4266        0.4006        0.4514 0.8230    0.8102    0.8347 0.1641   1.0000         0
  S7·C 40867       1259      0.0308     0.3876        0.3620        0.4125 0.7957    0.7828    0.8093 0.2056   1.0000         0
 S10·C 26435         77      0.0029     0.2727        0.1842        0.3820 0.7211    0.6600    0.7803 0.0154   1.0000         0
S10+·C 26598        240      0.0090     0.4333        0.3722        0.4955 0.8208    0.7910    0.8486 0.0958   1.0000         0
```

**C - A gap: -0.0096** (robust); A = 0.4398, C = 0.4302.
**S7·C - S7 gap: -0.0040** (the C-A gap with the length cue removed).

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
   A          0.5791        0.5734         0.5746 0.0045   39608.0000  26358.0000
   C          0.5786        0.5735         0.5747 0.0039   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5789
   PTB-XL  4399     0.0000     0.5716
SaMi-Trop   326   326.0000     0.6657
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5709     0.4615 0.8401
 (25.0, 33.0] 7152         30     0.5725     0.4333 0.7800
 (33.0, 41.0] 7710        148     0.5758     0.5878 0.8541
 (41.0, 48.0] 7177        203     0.5785     0.4926 0.8485
 (48.0, 55.0] 7453        226     0.5803     0.4735 0.8494
 (55.0, 61.0] 7071        254     0.5821     0.4606 0.8491
 (61.0, 67.0] 7198        223     0.5811     0.3498 0.8157
 (67.0, 73.0] 7367        217     0.5821     0.3687 0.7892
 (73.0, 79.0] 7312        190     0.5822     0.2842 0.7129
(79.0, 300.0] 7265        158     0.5830     0.2848 0.7333
```
