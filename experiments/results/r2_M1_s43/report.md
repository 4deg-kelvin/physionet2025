# Evaluation report -- r2_M1_s43

## Views (TPR@5% = `challenge`; 95% bootstrap CIs, 1000 patient-level resamples)

* `A`: full held-out (source and length cues available)
* `B`: CODE-15 only (source cue removed; length cue still available)
* `S7`: CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY
* `S10`: CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)
* `S10+`: CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)
* `·C`: the same view on Test C (organiser-style perturbations)

```
  view     n  positives  prevalence  challenge  challenge_lo  challenge_hi  auroc  auroc_lo  auroc_hi  auprc  ceiling  n_failed
     A 73300       1662      0.0227     0.3899        0.3635        0.4164 0.8055    0.7916    0.8192 0.1651   1.0000         0
     B 68575       1336      0.0195     0.3757        0.3431        0.4109 0.7871    0.7695    0.8051 0.1267   1.0000         0
    S7 40867       1259      0.0308     0.3789        0.3504        0.4125 0.7784    0.7587    0.7974 0.2060   1.0000         0
   S10 26435         77      0.0029     0.3247        0.2142        0.4359 0.7474    0.6784    0.8156 0.0249   1.0000         0
  S10+ 26598        240      0.0090     0.4625        0.4008        0.5252 0.8268    0.7932    0.8585 0.1222   1.0000         0
   A·C 73300       1662      0.0227     0.3941        0.3675        0.4223 0.8059    0.7916    0.8196 0.1657   1.0000         0
   B·C 68575       1336      0.0195     0.3817        0.3485        0.4175 0.7878    0.7705    0.8056 0.1285   1.0000         0
  S7·C 40867       1259      0.0308     0.3852        0.3538        0.4178 0.7787    0.7599    0.7973 0.2043   1.0000         0
 S10·C 26435         77      0.0029     0.3247        0.2112        0.4348 0.7405    0.6698    0.8100 0.0232   1.0000         0
S10+·C 26598        240      0.0090     0.4458        0.3882        0.5146 0.8217    0.7883    0.8540 0.1196   1.0000         0
```

**C - A gap: +0.0042** (robust); A = 0.3899, C = 0.3941.
**S7·C - S7 gap: +0.0064** (the C-A gap with the length cue removed).

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
   A          0.5756        0.5380         0.5467 0.0289   39608.0000  26358.0000
   C          0.5738        0.5426         0.5428 0.0309   39608.0000  26358.0000
```

## Per-source mean predicted probability (source-shortcut diagnostic)
```
   source     n  positives  mean_prob
 CODE-15% 68575  1336.0000     0.5676
   PTB-XL  4399     0.0000     0.3937
SaMi-Trop   326   326.0000     0.8121
```

## Score by age decile (age-shortcut diagnostic)
```
   age_decile    n  positives  mean_prob  challenge  auroc
(1.999, 25.0] 7595         13     0.5010     0.3077 0.7770
 (25.0, 33.0] 7152         30     0.5281     0.4667 0.7617
 (33.0, 41.0] 7710        148     0.5479     0.5000 0.8683
 (41.0, 48.0] 7177        203     0.5665     0.4433 0.8074
 (48.0, 55.0] 7453        226     0.5629     0.4115 0.8374
 (55.0, 61.0] 7071        254     0.5665     0.4409 0.8203
 (61.0, 67.0] 7198        223     0.5658     0.3049 0.7901
 (67.0, 73.0] 7367        217     0.5743     0.3088 0.7709
 (73.0, 79.0] 7312        190     0.5803     0.2368 0.7207
(79.0, 300.0] 7265        158     0.5916     0.2405 0.7335
```
