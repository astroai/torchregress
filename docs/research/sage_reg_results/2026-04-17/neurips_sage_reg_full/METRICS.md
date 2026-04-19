# SAGE-Reg full run — metric digest

- Generated: `2026-04-18T01:22:30.144498+00:00`
- Run root: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full`

## Year direct (by method × unlabeled fraction)

### SupervisedOnly

| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |
|------------------:|----:|-----:|------:|-----:|
| 0.25 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.8915620446205139 |
| 0.5 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.8915620446205139 |
| 1.0 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.8915620446205139 |

### MeanTeacher

| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |
|------------------:|----:|-----:|------:|-----:|
| 0.25 | 4.137241363525391 | 0.46131378412246704 | 0.617584228515625 | 0.8873944878578186 |
| 0.5 | 4.068221092224121 | 0.4612753689289093 | 0.615325927734375 | 0.8864114284515381 |
| 1.0 | 3.976048231124878 | 0.4606039822101593 | 0.619049072265625 | 0.8877015113830566 |

### ConfidenceWeightedPseudoLabel

| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |
|------------------:|----:|-----:|------:|-----:|
| 0.25 | 4.657982349395752 | 0.4772152900695801 | 0.508209228515625 | 0.8946952223777771 |
| 0.5 | 4.865184307098389 | 0.48633673787117004 | 0.4912109375 | 0.9131101965904236 |
| 1.0 | 3.969097137451172 | 0.4770316481590271 | 0.514007568359375 | 0.9008802771568298 |

### SAGE-Reg

| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |
|------------------:|----:|-----:|------:|-----:|
| 0.25 | 2.587252616882324 | 0.460395872592926 | 0.6676025390625 | 0.9041783809661865 |
| 0.5 | 2.2187914848327637 | 0.45747408270835876 | 0.686859130859375 | 0.9020062685012817 |
| 1.0 | 2.0032472610473633 | 0.4581788182258606 | 0.7059326171875 | 0.9056075811386108 |

### PiModelConsistency

| UnlabeledFraction | NLL | CRPS | Cov90 | RMSE |
|------------------:|----:|-----:|------:|-----:|
| 0.25 | 3.7223167419433594 | 0.45355600118637085 | 0.6353759765625 | 0.8756581544876099 |
| 0.5 | 3.5586936473846436 | 0.45351162552833557 | 0.64508056640625 | 0.8788394927978516 |
| 1.0 | 3.740816116333008 | 0.45510467886924744 | 0.63262939453125 | 0.8778514266014099 |

## Multiseed (tuned row, aggregate)

| Benchmark | Seeds | SAGE−Sup (mean) | SAGE−Sup (std) | Conf−Sup (mean) |
|-----------|------:|----------------:|---------------:|----------------:|
| higgs_public | 6 | -47.52406255404154 | 36.13920998893806 | 405.07711426417035 |
| year | 6 | 0.06360183159510295 | 0.12004423032266728 | 1.593176821867625 |

## OpenML diamonds multiseed (tuned Year row, aggregate)

| Benchmark | Seeds | SAGE−Sup (mean) | SAGE−Sup (std) | Conf−Sup (mean) |
|-----------|------:|----------------:|---------------:|----------------:|
| openml_diamonds | 6 | 0.02040032660200571 | 0.031852842805191926 | -0.024864997093876202 |

## Labeled budget sweep (collated NLL)

| n_labeled | NLL sup | NLL SAGE | Δ(SAGE−sup) |
|----------:|--------:|---------:|------------:|
| 1024 | 3.3430068492889404 | 2.361638307571411 | -0.9813685417175293 |
| 2048 | 2.9337880611419678 | 2.4886417388916016 | -0.4451463222503662 |
| 4096 | 2.1346349716186523 | 2.0032472610473633 | -0.13138771057128906 |
| 8192 | 1.7821369171142578 | 1.8850387334823608 | 0.10290181636810303 |
| 16384 | 1.3609564304351807 | 1.4978646039962769 | 0.1369081735610962 |
| 32768 | 1.2064917087554932 | 1.3256044387817383 | 0.11911273002624512 |

## Run manifest (phase paths)

- `year_direct`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/sage/year_direct`
- `multiseed`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/sage/multiseed`
- `year_labeled_sweep`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/year_labeled_sweep`
- `multiseed_year_nl2048`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/multiseed_year_nl2048`
- `catboost`: `docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/catboost`
- `tabred`: `docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/tabred`
- `synthetic`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/synthetic`
- `backbone`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/backbone`
- `ablations`: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/ablations`
- `shifts`: `/Users/fabbros/src/torchregress/data/shifts/solar/README.txt`
- `image_rebuttal`: `skipped`
- `openml_diamonds_multiseed`: `docs/research/sage_reg_results/2026-04-17/neurips_sage_reg_full/openml_diamonds`
- `tabred_fetch`: `manual_posthoc_probe_quick`

