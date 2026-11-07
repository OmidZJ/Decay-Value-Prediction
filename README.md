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
Initial Particle Name
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
Intermidiate Particle Name
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
Product Particle 1 Name
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
Product Particle 2 Name
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
Product Particle 3 Name
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
Product Particle 4 Name
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

Particle names identify occupied slots. Product multiplicities expand repeated daughters. Initial and intermediate multiplicities are checked for validity. The `Structure As Int` fields supply net `u`, `d`, `s`, `c`, and `b` counts. The encoder produces 1,285 features, including derived phase-space and quantum-number balance features.

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

The score comes from five source-grouped outer folds. Each outer fold runs 40 Optuna trials using three grouped inner folds. Target scaling is fitted on the training split. Related channels with different source IDs and decays of the same parent species can still appear across folds. The reported score includes the original flagged rows and measures performance under this split protocol.
