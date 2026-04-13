# Tabular paper bundle — metric digest

- Generated: `2026-04-12T05:04:54.320016+00:00`
- Bundle dir: `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle`

## SAGE year direct (by method × unlabeled fraction)

### SupervisedOnly

| UnlabeledFraction | NLL | CRPS | Cov90 | CalibMAE | RMSE |
|------------------:|----:|-----:|------:|---------:|-----:|
| 0.25 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.0640101432800293 | 0.8915620446205139 |
| 0.5 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.0640101432800293 | 0.8915620446205139 |
| 1.0 | 2.1346349716186523 | 0.4479001462459564 | 0.75518798828125 | 0.0640101432800293 | 0.8915620446205139 |

### ConfidenceWeightedPseudoLabel

| UnlabeledFraction | NLL | CRPS | Cov90 | CalibMAE | RMSE |
|------------------:|----:|-----:|------:|---------:|-----:|
| 0.25 | 4.657982349395752 | 0.4772152900695801 | 0.508209228515625 | 0.12651945650577545 | 0.8946952223777771 |
| 0.5 | 4.865184307098389 | 0.48633673787117004 | 0.4912109375 | 0.1308336704969406 | 0.9131101965904236 |
| 1.0 | 3.969097137451172 | 0.4770316481590271 | 0.514007568359375 | 0.12850020825862885 | 0.9008802771568298 |

### SAGE-Reg

| UnlabeledFraction | NLL | CRPS | Cov90 | CalibMAE | RMSE |
|------------------:|----:|-----:|------:|---------:|-----:|
| 0.25 | 2.587252616882324 | 0.460395872592926 | 0.6676025390625 | 0.09309884905815125 | 0.9041783809661865 |
| 0.5 | 2.2187914848327637 | 0.45747408270835876 | 0.686859130859375 | 0.09163721650838852 | 0.9020062685012817 |
| 1.0 | 2.0032472610473633 | 0.4581788182258606 | 0.7059326171875 | 0.08921123296022415 | 0.9056075811386108 |

## SAGE multiseed (supervised gap, aggregate over seeds)

| Benchmark | Seeds | SAGE−Sup (mean) | SAGE−Sup (std) | Conf−Sup (mean) |
|-----------|------:|----------------:|---------------:|----------------:|
| higgs_public | 3 | -74.45306825637817 | 53.451839328232865 | 434.406413714091 |
| year | 3 | 0.10186127821604411 | 0.06959456341858436 | 1.7276114225387573 |

## Other SPT summaries (paths from artifact manifest)

- **synthetic:** `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle/spt/full/synthetic_competing_methods_full.json`
- **tabular_small:** `/Users/fabbros/src/torchregress/docs/research/sage_reg_results/2026-04-11/tabular_paper_bundle/spt/full/tabular_competing_methods_full.json`

## SPT large-tabular track (Gaussian family, key methods)

| Method | NLL | CRPS | Cov90 | Width90 |
|--------|----:|-----:|------:|--------:|
| SourceGaussian | 2.7756375716059836 | 1.565695881843567 | 1.0 | 20.995376586914062 |
| SPTRegGaussian | 2.9365162763002957 | 1.8363659381866455 | 1.0 | 24.701122283935547 |
| TargetRefitSmallGaussian | 2.0665755522274436 | 0.7883169651031494 | 1.0 | 10.215767860412598 |
| RawSplitConformalGaussian | 2.7756375716059836 | 1.5609984397888184 | 0.915283203125 | 1.4269418716430664 |
| FeatureStatNormGaussian | 2.5998182261159353 | 1.3488633632659912 | 1.0 | 17.502334594726562 |
| PriorTransportGaussian | 2.7756375716059836 | 1.566078782081604 | 1.0 | 20.995376586914062 |
| SPTRegBinnedPDF | 1.9604624193089886 | 1.7150415182113647 | 0.943359375 | 2.41935658454895 |
| SPTTransportGaussian | 2.9365162763002957 | 1.8429841995239258 | 1.0 | 24.701122283935547 |
| SignificantSubspaceGaussian | 2.7300963978457906 | 1.4957623481750488 | 1.0 | 20.062103271484375 |
| SourceBinnedPDF | 1.9429357080277563 | 1.6971534490585327 | 1.0 | 7.341760635375977 |

