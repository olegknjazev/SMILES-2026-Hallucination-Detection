# SMILES 2026 Hallucination Detection. Report File

## Reproducibility instructions

### VS Code

Just run solution.py

### Google Colab

Run first cell in colab.ipynb

## Final solution description

### splitting.py

- Changes: Added Stratified cross-validation with fixed 6 folds for splitting train-val and test datasets

- Reason: for calculation more reliable, objective and sustainable metrics.

### aggregation.py

- Changes: Change the feature extractor from taking the last layer to compute the mean activation over all real (non‑padding) tokens for every layer separately. Added a small set of geometric features: the L2 norm of every layer's mean vector, which are concatenated to the main feature.

- Reason: to obtain a stable representation of the entire response's content rather than a single possibly irrelevant token.

### probe.py

- Changes: Added PCA
 
- Reason: to simplify the data structure and improve the performance of predictive models.

## Final metrics comparison

|-----------------------| Before | After  |
|-----------------------|--------|--------|
| avg_train_accuracy    | 0.86   | 0.74   |
| avg_train_f1          | 0.91   | 0.84   |
| avg_train_auroc       | 0.99   | 1.0    |
| avg_val_auroc         | 0.67   | 0.54   |
| avg_test_accuracy     | 0.75   | 0.70   |
| avg_test_f1           | 0.84   | 0.82   |
| avg_test_auroc        | 0.74   | 0.6    |
| n_folds               | 1      | 6      |
| extract_time_s        | 167.15 | 171.32 |

## Experiments and failed attempts

- Using max pooling instead of mean pooling made the features too sensitive to outlier tokens and lowered F1 by 0.03‑0.04.

- Concatenating all 12 layers (creating a 6144‑dimensional vector) caused severe overfitting – the probe memorised the training set but failed on validation.

- Adding learning rate scheduling or using sequence length as a geometric feature had no measurable effect. I also tried using attention entropy as a weighting scheme, but accessing attention weights required modifying model.py (which was not allowed) and initial tests showed noisy, uncorrelated signals.