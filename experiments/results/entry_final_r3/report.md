# Evaluation report -- entry_final_r3

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 patient-level resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4543        0.4250        0.4824 0.8447    0.8325    0.8564 0.2057   1.0000         0
     B 68575       1336      0.0195     0.4521        0.4180        0.4862 0.8391    0.8230    0.8541 0.1780   1.0000         0
    S7 40867       1259      0.0308     0.4162        0.3846        0.4496 0.8080    0.7901    0.8249 0.2168   1.0000         0
   S10 26435         77      0.0029     0.3636        0.2332        0.4949 0.7649    0.7031    0.8262 0.0186   1.0000         0
  S10+ 26598        240      0.0090     0.4708        0.4019        0.5410 0.8536    0.8263    0.8799 0.1101   1.0000         0
   A·C 73300       1662      0.0227     0.4477        0.4184        0.4772 0.8409    0.8284    0.8528 0.2025   1.0000         0
   B·C 68575       1336      0.0195     0.4461        0.4125        0.4813 0.8336    0.8173    0.8489 0.1743   1.0000         0
  S7·C 40867       1259      0.0308     0.4114        0.3785        0.4450 0.8059    0.7879    0.8230 0.2166   1.0000         0
 S10·C 26435         77      0.0029     0.3636        0.2333        0.5000 0.7452    0.6795    0.8079 0.0174   1.0000         0
S10+·C 26598        240      0.0090     0.4667        0.4015        0.5339 0.8459    0.8176    0.8744 0.1058   1.0000         0
```

**C - A gap: -0.0066** (robust); A = 0.4543, C = 0.4477.
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
   A          0.5919        0.5846         0.5858 0.0061   39608.0000  26358.0000
   C          0.5914        0.5849         0.5861 0.0053   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5910
   PTB-XL  4399     0.0000     0.5828
SaMi-Trop   326   326.0000     0.6739
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5816     0.3077 0.8207
 (25.0, 33.0] 7152         30     0.5834     0.5333 0.8265
 (33.0, 41.0] 7710        148     0.5866     0.5946 0.8666
 (41.0, 48.0] 7177        203     0.5890     0.4778 0.8480
 (48.0, 55.0] 7453        226     0.5912     0.4735 0.8586
 (55.0, 61.0] 7071        254     0.5938     0.4882 0.8666
 (61.0, 67.0] 7198        223     0.5935     0.3901 0.8455
 (67.0, 73.0] 7367        217     0.5956     0.3963 0.7910
 (73.0, 79.0] 7312        190     0.5964     0.2947 0.7431
(79.0, 300.0] 7265        158     0.5984     0.3228 0.7505
```
