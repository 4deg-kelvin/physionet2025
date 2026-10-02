# Evaluation report -- r2_MW_s43 (selected alpha1.00)

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.4091        0.3872        0.4320 0.8081    0.7953    0.8200 0.1767   1.0000         0
     B 68575       1336      0.0195     0.3937        0.3696        0.4208 0.7889    0.7752    0.8027 0.1418   1.0000         0
    S7 40867       1259      0.0308     0.3868        0.3619        0.4114 0.7847    0.7709    0.7987 0.2042   1.0000         0
   S10 26435         77      0.0029     0.3766        0.2740        0.4872 0.7807    0.7178    0.8373 0.0285   1.0000         0
  S10+ 26598        240      0.0090     0.4958        0.4275        0.5570 0.8391    0.8096    0.8692 0.1166   1.0000         0
   A·C 73300       1662      0.0227     0.4097        0.3874        0.4337 0.8082    0.7956    0.8205 0.1753   1.0000         0
   B·C 68575       1336      0.0195     0.3967        0.3730        0.4243 0.7887    0.7748    0.8023 0.1407   1.0000         0
  S7·C 40867       1259      0.0308     0.3884        0.3645        0.4138 0.7855    0.7712    0.7992 0.2040   1.0000         0
 S10·C 26435         77      0.0029     0.3636        0.2597        0.4730 0.7759    0.7120    0.8317 0.0226   1.0000         0
S10+·C 26598        240      0.0090     0.4875        0.4231        0.5513 0.8393    0.8091    0.8694 0.1115   1.0000         0
```

**C - A gap: +0.0006** (robust); A = 0.4091, C = 0.4097.
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
   A          0.5953        0.5755         0.5852 0.0101   39608.0000  26358.0000
   C          0.5920        0.5764         0.5834 0.0087   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5947
   PTB-XL  4399     0.0000     0.4155
SaMi-Trop   326   326.0000     0.8245
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5161     0.3077 0.7483
 (25.0, 33.0] 7152         30     0.5430     0.5000 0.7516
 (33.0, 41.0] 7710        148     0.5666     0.5676 0.8608
 (41.0, 48.0] 7177        203     0.5885     0.4335 0.7899
 (48.0, 55.0] 7453        226     0.5884     0.4336 0.8366
 (55.0, 61.0] 7071        254     0.5944     0.4606 0.8366
 (61.0, 67.0] 7198        223     0.5966     0.3094 0.8009
 (67.0, 73.0] 7367        217     0.6077     0.3318 0.7503
 (73.0, 79.0] 7312        190     0.6189     0.2737 0.7207
(79.0, 300.0] 7265        158     0.6327     0.2785 0.7351
```
