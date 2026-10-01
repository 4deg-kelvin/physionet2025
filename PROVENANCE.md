# Provenance

Every idea in the `chagas/` package, and where it came from.  The rule the team set is
that anything that is not the team's own work carries a citation in the docstring of the
code that implements it; this file is the index.

PLAN.md section 7 is the source of this table; the "implemented in" column is what was
actually built.

## The team's own work (do not attribute these to a paper)

| Idea | Where it came from | Implemented in |
|---|---|---|
| ECG-FM backbone as the Chagas feature extractor | `fm.ECGFMFeatureExtractor` | `chagas/model.py::ECGFMFeatureExtractor` |
| Forward-hook capture of **all 12** transformer layer outputs, bypassing the fairseq classification wrapper's `.detach()` | `fm.ECGFMFeatureExtractor._make_hook` | `chagas/model.py::ECGFMFeatureExtractor._make_hook` |
| Learnable **static softmax** aggregation over the 12 layers (zeros-initialised → uniform) | `fm.ECGFMFeatureExtractor.layer_weights` | `chagas/model.py::ECGFMFeatureExtractor._aggregate` (still the default) |
| Masked mean pooling over time | `fm.ECGFMFeatureExtractor.forward` | `chagas/model.py::masked_mean` |
| `Dropout → Linear(768,256) → ReLU → Dropout → Linear(256,1)` head with age/sex side input | `fm.FMChagasClassifier` | `chagas/model.py::ChagasModule` |
| Per-record per-lead z-score before windowing, constant leads → 0 | `utils.zscore_per_record` | `chagas/preprocessing.py::zscore_per_record` |
| Enforced ECG-FM lead order with a 12-lead guard | `utils.preprocess_signal` | `chagas/preprocessing.py::reorder_leads` |
| Powerline interference augmentation (50/60 Hz ± jitter, harmonics at ½ and ⅓, identical across leads) | `augmentations.ECGAugmentations._add_powerline_interference` (Andy Smithwick, Sept 2025) | `chagas/augment.py::powerline` |
| Random crop + temporal shift augmentation | `augmentations._random_crop_and_shift` | `chagas/augment.py::crop_shift` |
| RBBB / 1dAVb-informed soft labels | `dataloader.ECGDataset._compute_soft_label` | `chagas/labels.py::_team_rbbb_soft_label` (`rbbb_scheme: "team_rbbb"`) |
| Auxiliary multitask heads on CODE-15 comorbidity labels | dev-tree `fm/train_multitask_fm.py` | `chagas/model.py` aux head + `chagas/losses.py::masked_aux_bce` |
| `TimeLimitCallback` for the organisers' 72 h ceiling | `fm.TimeLimitCallback` | `chagas/training.py::TimeLimitCallback` (made budget-aware) |
| `NotImplementedError` as the "do not emit predictions" tripwire | `team_code.run_model`, `custom_helper_code.get_patient_info` | `chagas/inference.py::run_model`, `chagas/labels.py::parse_patient_info` |
| Patient-grouped held-out split and the Test A / Test B framing | `fixbranch_test/build_split.py`, `score.py` | `chagas/training.py::split_entries`, `experiments/eval_report.py` |

## Borrowed ideas

| Idea | Source | Implemented in |
|---|---|---|
| Soft labels 0.8 / 0.2 for CODE-15, 1.0 SaMi-Trop, 0.0 PTB-XL; `pos_weight = 5` with 5× oversampling; two-stage frozen → full fine-tuning (2e-4 then 2e-5, effective batch 64, ~2,000 stage-1 steps); FiLM demographics; model selection on top-5% TPR | Van Santvliet, Nguyen, Vandenberk & De Vos, *Detecting Chagas Disease Using a Vision Transformer-based ECG Foundation Model*, CinC 2025, DOI 10.22489/CinC.2025.019 | `chagas/labels.py::soft_label`, `chagas/losses.py::weighted_soft_bce`, `chagas/dataset.py::ChagasDataModule`, `chagas/training.py` |
| Multi-report soft labels `c = 0.51·y + 0.49·(n_pos/n_total)`; RBBB-informed bump for CODE-15 negatives (+0.15 / +0.25, capped 0.49); margin-ranking loss on hard negatives; baseline wander / Gaussian noise / amplitude scaling / time warp / segment masking parameters; Butterworth band-pass | Hong, Chou & González, *Managing Label Uncertainty in the Detection of Chagas Disease from the ECG*, CinC 2025, DOI 10.22489/CinC.2025.100 | `chagas/labels.py::MultiReportStats`, `_hong_rbbb_bump`, `chagas/losses.py::hard_negative_ranking_loss`, `chagas/augment.py`, `chagas/preprocessing.py::clean_signal` |
| Source-stratified label smoothing; "upsampling is harmful under label noise" (the `oversample: 1.0` ablation); early stopping on the challenge score with patience 3 | *Reliability-aware hierarchical learning for Chagas disease screening from 12-lead ECGs*, Physiol. Meas. 2026, doi:10.1088/1361-6579/ae98e0 | `chagas/config.py::LabelCfg`, `TrainCfg.oversample`, `TrainCfg.early_stop_patience` |
| Input-dependent gate over layers (PMA); the finding that middle layers carry most information | Nguyen et al., *Exploiting a Mixture-of-Layers in an ECG Foundation Model*, arXiv:2509.00102 | `chagas/model.py` `aggregation: "pma"` (default OFF; the team's static softmax stays default) |
| 500 Hz / per-record z-score pretraining recipe; full fine-tuning beats linear probing | McKeen et al., *ECG-FM: an open electrocardiogram foundation model*, JAMIA Open 2025, arXiv:2408.05178 | `chagas/preprocessing.py`, `chagas/training.py` two-stage schedule |
| RBBB / LAFB-type conduction disease as the ECG signature of Chagas; seed ensembling | Jidling et al., PLoS NTD 2023, 17(7):e0011118 | auxiliary head choice; `TrainCfg.ensemble_members` |
| Layer-wise learning-rate decay (discriminative fine-tuning) | Howard & Ruder 2018 (ULMFiT); Clark et al. 2020 (ELECTRA) | `chagas/model.py::ChagasModule.param_groups` |
| FiLM conditioning (`γ ⊙ x + β`, identity-initialised) | Perez, Strub, de Vries, Dumoulin & Courville, *FiLM: Visual Reasoning with a General Conditioning Layer*, AAAI 2018 | `chagas/model.py::DemographicFiLM` |
| Domain-adversarial training with a gradient-reversal layer | Ganin et al., JMLR 2016; applied to Chagas by "CinCo Amigos", CinC 2025 (rank 7, best ELSA-Brasil score in the top 15) | `chagas/losses.py::grad_reverse`, `chagas/model.py` domain head (default OFF) |
| Multitask learning as a regulariser | Caruana, *Multitask Learning*, Mach. Learn. 1997 | `chagas/losses.py::masked_aux_bce` |
| The test-time perturbation types (noise, device filters, resampling), cohort facts and the challenge metric | Reyna et al., *Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025*, CinC 2025 / arXiv:2510.02202; <https://moody-challenge.physionet.org/2025/> | `chagas/augment.py` stage R, `experiments/build_test_C.py`, `chagas/metrics.py` |
| `resample_poly`, `sosfiltfilt`, `iirnotch`, Butterworth design | `scipy.signal` — standard DSP, no paper | `chagas/preprocessing.py`, `chagas/augment.py` |
| The closed-form expectation used to make the challenge score fast | Derived here from the organisers' own definition; the reference Monte-Carlo implementation is `helper_code.compute_challenge_score` | `chagas/metrics.py::challenge_score` |

## Deliberate departures from the cited sources

* **`crop_shift`** does the temporal shift by randomising the *crop start* rather than
  prepending zeros as the team's `augmentations.py` does.  The padding mask requires the
  valid region to be a prefix of the window (fairseq-signals' own token-level mask is
  prefix-shaped), and leading zeros would be counted as signal.  `edge_mask` still
  provides zeroed edges inside the valid region.
* **`baseline_wander` and `gaussian_noise` amplitudes** are relative to the record's
  standard deviation, not absolute.  Hong et al. state `a = 0.1` and `σ = 0.01`, which
  are values for normalised signal; these augmentations run *before* the per-record
  z-score, where an absolute amplitude would mean something different for each cohort.
* **The static softmax stays the default aggregation.**  Nguyen et al.'s input-dependent
  gate is implemented but off, because the team's version is the one the existing results
  were measured with and the ladder has not yet earned the change.

## Round 2 (PLAN_R2.md, 2026-09-30 / 10-01)

| Idea | Origin | Implemented in |
|---|---|---|
| **Discovery** that CODE-15 record length predicts the label (7.34 s vs 10.24 s, 12.7x prevalence ratio), and the **within-length evaluation** (views S7 / S10 / S10+, duration-only baseline, length-cue gap) | **This project** (analysis of `physionet2025/signal_lengths.csv` and saved predictions, 2026-09-30; DECISIONS.md D-2026-09-30-2) | `chagas/lengths.py`, `experiments/scoring.py`, `experiments/eval_report.py`, `experiments/aggregate.py`, `experiments/rescore_all.py` |
| ECG-FM backbone, 12-layer aggregation, MLP head (unchanged); "random padding or random windowing" named as future work | **The team** (`fm.py`); Nguyen et al., CinC 2025, DOI 10.22489/CinC.2025.416 | `chagas/model.py` (unchanged core) |
| CODE records are 7–10 s long | Ribeiro et al., *Automatic diagnosis of the 12-lead ECG using a deep neural network*, Nat. Commun. 2020, 11:1760 | `chagas/lengths.py` docstring |
| 5 s segments at 500 Hz as ECG-FM's pretraining input | McKeen et al., *ECG-FM: an open electrocardiogram foundation model*, JAMIA Open 2025; arXiv:2408.05178 | `PreprocessCfg.window_mode`, `chagas/preprocessing.py::window_starts/extract_windows/random_window` |
| Random 5 s training crops, window averaging at inference | FjordNet, CinC 2025, DOI 10.22489/CinC.2025.091 | `chagas/preprocessing.py::random_window`, `extract_windows`; `chagas/model.py::segment_mean` |
| Shortcut learning; balancing / subsampling groups to remove a spurious cue (frequency matching of controls to cases on a confounder) | Geirhos et al., *Shortcut learning in deep neural networks*, Nat. Mach. Intell. 2020; Idrissi et al., *Simple data balancing achieves competitive worst-group-accuracy*, CLeaR 2022; Sagawa et al., *Distributionally robust neural networks*, ICLR 2020 | `DataCfg.match_negatives_on`, `chagas/training.py::matched_negative_quota`, `_matched_negatives` |
| WiSE-FT weight interpolation; LP-FT / feature-distortion analysis | Wortsman et al., *Robust fine-tuning of zero-shot models*, CVPR 2022; Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*, ICLR 2022 | `chagas/wise_ft.py`, `chagas/training.py::_wise_ft_select`, `TrainCfg.wise_ft*` |
| A frozen encoder beat a fine-tuned one on the hidden data | Wang et al. (CinCo Amigos), CinC 2025, DOI 10.22489/CinC.2025.158; Nguyen et al. 2025 (416) | motivation in `chagas/wise_ft.py`; tie-break towards smaller alpha (D-2026-10-01-2) |
| Deep ensembles; multi-member ensembles for Chagas | Lakshminarayanan, Pritzel & Blundell, NeurIPS 2017; Nejedly et al. (ISIBrno), CinC 2025, DOI 10.22489/CinC.2025.132; Jidling et al., PLoS NTD 2023 | `chagas/inference.py::ChagasPredictor` (unchanged), `experiments/aggregate.py` seed ensemble |
| Hidden cohorts are ~10 s; cohort facts | Reyna et al., CinC 2025 / arXiv:2510.02202 | `chagas/lengths.py` docstring, S10 view |
| Percentile bootstrap; weighted (multiplicity) form on pre-sorted scores | Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993 (standard); the weighted closed form is derived here from `chagas/metrics.py` | `experiments/scoring.py::RankedScores`, `bootstrap_weights` |

Engineering that is ours and carries no citation: the durations builder and its full cross-check, batched inference
(`experiments/predict_batched.py`), `run_info.json`, the multi-window collate.
