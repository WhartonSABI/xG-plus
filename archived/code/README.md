## Details for .ipynb files

This directory contains the analysis code for the project:

- `multigame.py`: cleans raw Premier League match data and constructs features from ball and player locations.
- `baseline.ipynb`: fits baseline logistic models using several feature subsets.
- `all-season_model.ipynb`: trains xG and xS models (XGBoost) and produces predicted xG, xS, and xG+ outputs for evaluation.
- `stability_test.ipynb`: runs a stability evaluation using 10 models; each model is trained on 90% of the training set and evaluated on the held-out portion.

For security, any AWS Access Key IDs and Secret Access Keys are redacted.
