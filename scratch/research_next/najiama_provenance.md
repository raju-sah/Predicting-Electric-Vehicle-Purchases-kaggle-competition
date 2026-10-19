# Najiama artifact provenance — Sep 19, 2026

## Verified sources and artifacts

**Pure LGBM V6:** The current [first-party LGBM notebook](https://www.kaggle.com/code/najiama/pure-lgbm-model-cv-0-94607-lb-0-94638) explicitly sets `Folds = 5`, then `StratifiedKFold(n_splits=Folds, shuffle=True, random_state=42)`. Target encoding is fitted within each outer training fold; sklearn `TargetEncoder.fit_transform` uses internal cross-fitting. Downloaded output log shows five completed folds, with AUCs 0.94513, 0.94573, 0.94694, 0.94640, 0.94617, and pooled 0.94607. Current notebook outputs `oof_LIGHTGBM.csv` and `test_LIGHTGBM.csv` have prediction arrays exactly equal to downloaded dataset `Pure LGBM_V6_oof.csv` and `Pure LGBM_V6_test.csv`. Comparison uses `np.array_equal` after pandas CSV parsing. Source notebook text in `najiama_train/`; outputs and log in `najiama_lgb_output/`.

V6 uses ordinary `LGBMClassifier`, no custom focal objective. Parameters: 20,000 maximum estimators, learning rate .02, depth 5, 32 leaves, minimum child samples 10, subsample .812763, column sample .30293, alpha .07094, lambda 2.03303, max bin 1024, seed42, AUC metric, 500-round early stopping. These are verified V6 settings, **not verified V5 settings**.

**XGBoost Triple-TE 5-fold:** Current [first-party XGB notebook](https://www.kaggle.com/code/najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639) likewise sets `Folds = 5`, shuffled `StratifiedKFold` seed42. Its markdown incorrectly retains a ten-fold headline; executable source and log both show five folds. Output log AUCs: 0.94523, 0.94578, 0.94704, 0.94644, 0.94624, pooled 0.94614. Downloaded `oof_XGBOOST.csv` and `test_XGBOOST.csv` prediction arrays exactly equal dataset `XGBoost_Triple_TE_5folds_oof.csv` and `XGBoost_Triple_TE_5folds_test.csv`. Objective is `binary:logistic`. Source in `najiama_xgb/`; output and log in `najiama_xgb_output/`.

## V5 remains unverified

The [dataset description](https://www.kaggle.com/datasets/najiama/s6e9-oof) explicitly claims all members were rerun with shuffled seed42 fivefold CV. However, its text specifically credits older V1 files and now the dataset also includes explicitly ten-fold XGB files, so that blanket description cannot establish V5-specific provenance. The current LGB source calls its feature version V3 while matching the V6 dataset predictions exactly; filenames/version labels alone are unreliable.

Historical CLI kernel pulls with `/5` and `/6` returned 403. No source-to-output link for V5 was found during this bounded investigation. Its objective and hyperparameters therefore remain unknown; do not label it focal or certify fivefold provenance. Exclude V5 from a strictly verified fixed-fold pool until its source or fold assignment can be established.

## Reproduction

Our own `compare_provenance.py` compares current first-party kernel output predictions against the downloaded dataset files using exact parsed-array equality. No downloaded notebook code was executed, no training or submissions performed. Existing project-level ID alignment and validation checks remain necessary before blending.

Whole-file SHA256 also matches between notebook outputs and corresponding dataset files (therefore IDs and CSV contents match too):

| Artifact | SHA256 |
|---|---|
| LGB V6 OOF | `5597ebfad7f443a7c8e3e7c77dc1ad9f10637761d537b6eb87a26de9269328bf` |
| LGB V6 test | `7c170fa7913a597a63f411fc579c28cc01c2fee98c4c1ed36157c8abd62a309d` |
| XGB 5fold OOF | `ffc1ffc1bd4163cc6c73777e2b2f4aeff698b957e7237d7cd666589d01a15ddd` |
| XGB 5fold test | `d5c2bc4ab1c653e8ef370ca2c6db5b4762d1b1eaefb58db1da516865dbb319fe` |
