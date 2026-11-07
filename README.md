# Decay Value Prediction

XGBoost predicts the base-10 logarithm of a decay branching fraction. The best run used one selected diagram and pseudo-Huber loss on all 3,449 rows in the training CSV. Pooled out-of-fold scores were **$R^2$ = 0.80135**, **RMSE = 0.69582 dex**, and **MAE = 0.47454 dex**.

## Run

Keep `All PDG Training Decays 2026.csv` and `All PDG Decays 2026.csv` beside the script and run from the repository root.

```bash
pip install numpy pandas xgboost==3.2.0 optuna==4.9.0 scikit-learn==1.6.1
python BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py
```

On Kaggle, attach the same two CSVs and paste the [standalone script](BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py) into a code cell. The output folder `xgb_new_pdg_all_unfiltered` contains held-out predictions, fold assignments, metrics, and the final model.

## CSV columns used

**Decay context** — `Decay Type`, `Is Decay Radiative`, `Decay Q Value`, `Decay P*`.

**Diagrams** — For each `n` in `1`, `2`, `3`, the script reads `Diagram n Topology`, `Diagram n Quark Transition`, `Diagram n Effective CKM Matrix Element`, and `Diagram n Relative Amplitude`. Relative amplitude ranks the available diagrams. Only the selected diagram contributes topology, transition, and CKM features.

**Particle fields** — Join each prefix and suffix below with one space to get the exact CSV column name. `Intermidiate Particle` follows the CSV spelling.

| Prefixes | Suffixes |
| --- | --- |
| `Initial Particle`, `Intermidiate Particle`, `Product Particle 1`, `Product Particle 2`, `Product Particle 3`, `Product Particle 4` | `Name`, `Multiplicity`, `Type`, `Mass`, `Width`, `Charge`, `Quantum C`, `Quantum G`, `Quantum I`, `Quantum J`, `Quantum P`, `Structure As Int` |

`Name` marks occupied slots. Product multiplicities expand repeated daughters. Initial and intermediate multiplicities are checked for validity. `Structure As Int` supplies net `u`, `d`, `s`, `c`, and `b` counts. The encoder converts the inputs into 1,285 features, including derived phase-space and quantum-number balance features.

| File | Other columns read | Role |
| --- | --- | --- |
| `All PDG Training Decays 2026.csv` | `Decay Value` | Training target and source lookup |
| `All PDG Training Decays 2026.csv` | `Decay Description` | Source lookup and reporting |
| `All PDG Training Decays 2026.csv` | `Decay Value Error` | Uncertainty audit |
| `All PDG Decays 2026.csv` | `PDG ID`, `Decay Description`, `Decay Value` | Match each training row to its original PDG record for fold grouping |

## Validation

The score comes from five source-grouped outer folds. Each outer fold runs 40 Optuna trials using three grouped inner folds. Target scaling is fitted on the training split. Related channels with different source IDs and decays of the same parent species can still appear across folds. The reported score includes the original flagged rows and measures performance under this split protocol.
