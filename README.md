# Decay Value Prediction

[نسخهٔ فارسی](README.fa.md)

XGBoost predicts the base-10 logarithm of a decay branching fraction. The best observed run used one selected diagram, pseudo-Huber loss with `huber_slope = 0.8`, and 100 Optuna trials per outer fold. It reached **$R^2$ = 0.80143** on held-out predictions for all 3,449 training rows.

## Run

Keep `All PDG Training Decays 2026.csv` and `All PDG Decays 2026.csv` beside the script. Install the benchmark dependencies and run the 100-trial GPU configuration from the repository root.

```bash
pip install numpy pandas xgboost==3.2.0 optuna==4.9.0 scikit-learn==1.6.1
python BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py --device cuda --trials 100 --output xgb_new_pdg_gpu100
```

For the default 40-trial CPU run, execute the script without arguments. On Kaggle, attach the same two CSVs, enable a GPU, paste the [standalone script](BDecay_XGBoost_NewPDG_All_Unfiltered_Optuna.py) into a code cell, and set `n_trials` to `100` and `device` to `'cuda'` in `Config`. Outputs include held-out predictions, fold assignments, metrics, and the final model.

## Results

| Run | Trials per outer fold | Pseudo-Huber slope | Out-of-fold $R^2$ | RMSE (dex) | MAE (dex) |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU baseline | 40 | Fixed at 0.8 | 0.80135 | 0.69582 | 0.47454 |
| GPU control | 40 | Fixed at 0.8 | 0.79897 | 0.69999 | 0.47596 |
| **GPU, longer search** | **100** | **Fixed at 0.8** | **0.80143** | **0.69569** | **0.47157** |
| GPU, slope tuning | 100 | Tuned in the inner folds | 0.79799 | 0.70168 | 0.47417 |

The 100-trial fixed-slope run improved $R^2$ by 0.00246 over the same-device control, which used the first 40 trials of that search. The gain over the earlier CPU baseline was about 0.00007.

### CPU comparison on all rows and on B decays

Both runs used CPU, 40 Optuna trials per outer fold, the same 1,285 selected-diagram features, and `huber_slope = 0.8`. The B-only run restricted training and inner validation to open-bottom mesons and baryons. Both models were evaluated on the same 751 open-bottom rows using the same outer fold assignments. All 120 flagged rows in this cohort were retained.

| Training scope | Held-out rows scored | Out-of-fold $R^2$ | RMSE (dex) | MAE (dex) |
| --- | --- | ---: | ---: | ---: |
| Open-bottom only (751 rows) | 751 open-bottom rows | 0.74609 | 0.63591 | 0.46307 |
| All particles (3,449 rows) | All 3,449 rows | 0.80135 | 0.69582 | 0.47454 |
| All particles (3,449 rows) | The same 751 open-bottom rows | **0.75530** | **0.62427** | **0.45879** |

The last two rows use predictions from the same all-particle run. On the same 751 open-bottom rows, training on all particles raised $R^2$ by 0.00921. The B-only run completed in 12.2 minutes, including the separate deployment search.

#### How the B rows are held out

In each outer fold, the all-particle model is fitted from scratch in one training stage on both B rows and other decay rows from the training portion of that fold. The B-only model uses the same training B rows. Both fits and their Optuna searches use only rows from the training portion.

| Fold 1 example | Training rows | Held-out rows used for the B score |
| --- | --- | --- |
| B-only model | 606 B rows | 145 B rows |
| All-particle model | The same 606 B rows + 2,153 other decay rows | The same 145 B rows |

Each B row contributes one held-out prediction per model. The B scores pool those predictions across all five outer folds.

## CSV columns used

### `All PDG Training Decays 2026.csv`

**Decay context**

```text
Decay Type
Is Decay Radiative
Decay Q Value
Decay P*
```

**Diagram inputs**

```text
Diagram 1 Topology
Diagram 1 Quark Transition
Diagram 1 Effective CKM Matrix Element
Diagram 1 Relative Amplitude
Diagram 2 Topology
Diagram 2 Quark Transition
Diagram 2 Effective CKM Matrix Element
Diagram 2 Relative Amplitude
Diagram 3 Topology
Diagram 3 Quark Transition
Diagram 3 Effective CKM Matrix Element
Diagram 3 Relative Amplitude
```

**Initial Particle**

```text
Initial Particle Multiplicity
Initial Particle Type
Initial Particle Mass
Initial Particle Width
Initial Particle Charge
Initial Particle Quantum C
Initial Particle Quantum G
Initial Particle Quantum I
Initial Particle Quantum J
Initial Particle Quantum P
Initial Particle Structure As Int
```

**Intermidiate Particle** (CSV spelling)

```text
Intermidiate Particle Multiplicity
Intermidiate Particle Type
Intermidiate Particle Mass
Intermidiate Particle Width
Intermidiate Particle Charge
Intermidiate Particle Quantum C
Intermidiate Particle Quantum G
Intermidiate Particle Quantum I
Intermidiate Particle Quantum J
Intermidiate Particle Quantum P
Intermidiate Particle Structure As Int
```

**Product Particle 1**

```text
Product Particle 1 Multiplicity
Product Particle 1 Type
Product Particle 1 Mass
Product Particle 1 Width
Product Particle 1 Charge
Product Particle 1 Quantum C
Product Particle 1 Quantum G
Product Particle 1 Quantum I
Product Particle 1 Quantum J
Product Particle 1 Quantum P
Product Particle 1 Structure As Int
```

**Product Particle 2**

```text
Product Particle 2 Multiplicity
Product Particle 2 Type
Product Particle 2 Mass
Product Particle 2 Width
Product Particle 2 Charge
Product Particle 2 Quantum C
Product Particle 2 Quantum G
Product Particle 2 Quantum I
Product Particle 2 Quantum J
Product Particle 2 Quantum P
Product Particle 2 Structure As Int
```

**Product Particle 3**

```text
Product Particle 3 Multiplicity
Product Particle 3 Type
Product Particle 3 Mass
Product Particle 3 Width
Product Particle 3 Charge
Product Particle 3 Quantum C
Product Particle 3 Quantum G
Product Particle 3 Quantum I
Product Particle 3 Quantum J
Product Particle 3 Quantum P
Product Particle 3 Structure As Int
```

**Product Particle 4**

```text
Product Particle 4 Multiplicity
Product Particle 4 Type
Product Particle 4 Mass
Product Particle 4 Width
Product Particle 4 Charge
Product Particle 4 Quantum C
Product Particle 4 Quantum G
Product Particle 4 Quantum I
Product Particle 4 Quantum J
Product Particle 4 Quantum P
Product Particle 4 Structure As Int
```

Product multiplicities expand repeated daughters. Initial and intermediate multiplicities are checked for validity. The `Structure As Int` fields supply net `u`, `d`, `s`, `c`, and `b` counts. The encoder produces 1,285 features, including derived phase-space and quantum-number balance features.

**Presence check during preprocessing** — The code checks whether a particle name is present to mark the corresponding slot as occupied.

**Target, grouping, and audit columns**

- `Decay Value` is the training target and also helps match the source record.
- `Decay Description` is used for source lookup and reporting.
- `Decay Value Error` is used for the uncertainty audit.

### `All PDG Decays 2026.csv`

```text
PDG ID
Decay Description
Decay Value
```

The script matches each training row by `Decay Description` and `Decay Value` to recover its `PDG ID` for grouped validation.

## How the diagram is selected for each row

1. Consider diagram slots 1, 2, and 3 whose `Topology` field is present.
2. Among those slots, choose the highest finite `Relative Amplitude`. Ties go to the lowest slot number. For example, if all three topologies are present and their amplitudes are `0.20`, `0.65`, and `0.15`, the code selects Diagram 2.
3. If none of the available slots has a finite amplitude, choose the first slot with a topology. If every topology is missing, the diagram features are marked unknown.
4. Use the selected slot's first-stage topology, first quark transition, and supplied effective CKM matrix element. The relative amplitude is used only to rank the diagrams.

## Validation

The reported scores come from five source-grouped outer folds. Each Optuna study uses three grouped inner folds. Target scaling is fitted on the training split. Related channels with different source IDs and decays of the same parent species can still appear across folds. The scores include the original flagged rows and measure performance under this split protocol.
