"""Standalone all-row XGBoost + Optuna counterpart to unfiltered new-PDG ParT.

Every row of the Training CSV is retained, including all reported issue flags
and zero/missing uncertainties. On the audited file this is 3449 rows across
178 parent labels, including 751 open-bottom decays and 633 flagged rows.
Flags are report-only and never select rows, features, weights or splits.
Same particle/context information, selected-diagram projection, and fixed
permutation-invariant tabular encoding as the filtered XGBoost version.

Each outer fold runs an independent Optuna study using only grouped inner
validation. Target scaling is fitted on the training rows of each fit. No outer
early stopping, global preprocessing fit, or reuse of outer scores for tuning.
Final deployment uses a separate full-data grouped study and fits all rows.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

@dataclass
class Config:
    data_path: str = 'All PDG Training Decays 2026.csv'
    source_data_path: str = 'All PDG Decays 2026.csv'
    validation_grouping: str = 'source'
    output_dir: str = '/kaggle/working/xgb_new_pdg_all_unfiltered' if Path('/kaggle').exists() else 'xgb_new_pdg_all_unfiltered'
    seed: int = 42
    outer_folds: int = 5
    inner_folds: int = 3
    n_trials: int = 40
    min_trees: int = 150
    max_trees: int = 1800
    n_jobs: int = 4
    device: str = 'cpu'  # 'cuda' for XGBoost >=2 with a compatible GPU
    objective: str = 'reg:pseudohubererror'
    huber_delta_normalized: float = .8
    max_daughters: int = 10
    log_prediction_min: float = -12.
    log_prediction_max: float = 0.
    train_full_data_model: bool = True



PREFIXES = ['Initial Particle', 'Intermidiate Particle'] + [f'Product Particle {i}' for i in range(1, 5)]

EXPANDED_FLAVOR_COLUMNS = (7, 9)

QUANTUM_SPECS = [
    ('Type', 1, 0, 7, 4), ('Quantum J', 2, 0, 12, 3),
    ('Quantum P', 1, 1, 3, 2), ('Charge', 1, 2, 5, 3),
    ('Quantum C', 1, 1, 3, 2), ('Quantum G', 1, 1, 3, 2),
    ('Quantum I', 2, 0, 5, 2),
    ('Number of u Quarks', 1, 2, 6, 2), ('Number of d Quarks', 1, 2, 5, 2),
    ('Number of s Quarks', 1, 2, 6, 2), ('Number of c Quarks', 1, 2, 5, 3),
    ('Number of b Quarks', 1, 2, 5, 3),
]

CTX_CATEGORIES = [('Decay Type', 7), ('Decay Topology', 16), ('Decay Quark Transition', 11)]

CTX_CONTINUOUS = ['Decay Q Value', 'Decay P*', 'CKM Matrix Element']

HADRON_TYPE_IDS = (1, 2)

LEPTON_TYPE_ID = 3

TYPE_CODES = {'Meson': 1, 'Baryon': 2, 'Lepton': 3, 'Boson': 4, 'Quark': 5}

DECAY_TYPE_CODES = {'Semi-Leptonic': 1, 'Leptonic': 2, 'Hadronic': 3}

TOPOLOGY_CODES = {'Tree': 1, 'Hadronic Tree': 2, 'Color-Allowed Tree (T)': 3,
                  'Color-Suppressed Tree (C)': 4, 'Exchange (E)': 5, 'QCD Penguin (P)': 6,
                  'Electroweak Penguin (P_EW)': 7, 'Electromagnetic Penguin (P_gamma)': 8,
                  'Annihilation (A)': 9, 'Semileptonic Tree': 10, 'Semileptonic (Tau) Tree': 11,
                  'Leptonic Tree': 12, 'Strong Decay': 13, 'Electromagnetic Decay': 14,
                  'Box Diagram': 15, 'W-Exchange': 16}

TRANSITION_CODES = {'b -> c': 1, 'b -> u': 2, 'b -> s': 3, 'b -> d': 4,
                    'c -> s': 5, 'c -> d': 6, 's -> u': 7, 'u -> d': 8,
                    'd -> u': 9, "q -> q'": 10, 'W -> q1 q2': 11}

FRACTIONAL_CHARGE_IDS = {7: -2 / 3, 8: -1 / 3, 9: 1 / 3, 10: 2 / 3}

ISSUE_COLUMNS = ['Issue Final Multiplicity', 'Issue Identical Inputs', 'Issue Target Type',
                 'Issue Charge', 'Issue Self Conjugate']

AUDITED_TRAINING_SHA256 = '9e1a8b592f610a7eb2b7305feb6eedf8c887f1b304c9df9a984539deff1d7548'

CONFIRMED_MULTIPLICITY_RECORDS = {489, 1129, 1139, 1239, 1247, 1248, 1424, 1516, 1654, 1727, 2559, 2654}

def banner(message):
    print('\n' + '=' * 94 + '\n' + message + '\n' + '=' * 94, flush=True)

def parse_number(value):
    """Missing/ambiguous values stay unknown, never silently become physical zero."""
    if value is None or pd.isna(value):
        return float('nan')
    if isinstance(value, (float, int, np.number)):
        return float(value) if np.isfinite(value) else float('nan')
    s = str(value).strip().replace('−', '-').strip('[]()\'"; ')
    frac = re.fullmatch(r'([+-]?\d+)\s*/\s*(\d+)', s)
    if frac:
        denominator = float(frac.group(2))
        return float(frac.group(1)) / denominator if denominator else float('nan')
    # Limits, ranges, alternatives, and "DEFINED AS 1" are not central values.
    if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', s):
        return float('nan')
    result = float(s)
    return result if math.isfinite(result) else float('nan')

def numeric_column(df, column):
    return df[column].map(parse_number).to_numpy(dtype=np.float64, copy=True)

def encode_category(values, multiplier, offset, max_id, label):
    known = np.isfinite(values)
    scaled = np.where(known, values * multiplier + offset, 0.)
    rounded = np.rint(scaled)
    invalid = known & ((np.abs(scaled - rounded) > 1e-5) | (rounded < 0) | (rounded > max_id))
    if invalid.any():
        raise ValueError(f'{label}: unsupported category values {np.unique(values[invalid])[:8]}; extend its vocabulary explicitly.')
    # ID 0 is UNKNOWN; every ordinary numeric category is shifted by one.
    return np.where(known, rounded + 1, 0).astype(np.int64)

def encode_signed_quark_count(values, label):
    """Keep V5 IDs for -2..3; append ID 7 for -3 and reserve 0 for unknown."""
    known = np.isfinite(values)
    rounded = np.rint(np.where(known, values, 0.))
    invalid = known & ((np.abs(values - rounded) > 1e-5) | (rounded < -3) | (rounded > 3))
    if invalid.any():
        raise ValueError(f'{label}: signed quark counts must be integers in [-3, 3].')
    encoded = np.where(rounded == -3, 7, rounded + 3)
    return np.where(known, encoded, 0).astype(np.int64)

def encode_charge(values, label):
    """Preserve legacy integer IDs and append four fractional-charge categories."""
    fractional = np.zeros(len(values), dtype=bool)
    encoded_fractional = np.zeros(len(values), dtype=np.int64)
    for category, charge in FRACTIONAL_CHARGE_IDS.items():
        match = np.isfinite(values) & np.isclose(values, charge, atol=1e-6, rtol=0.)
        fractional |= match
        encoded_fractional[match] = category
    integer_values = np.where(fractional, np.nan, values)
    ordinary = encode_category(integer_values, 1, 2, 5, label)
    return np.where(fractional, encoded_fractional, ordinary)

def decode_constituents(value):
    """Decode ten SIGNED integer/symbolic fields; never cast the whole code to int.

    Negative tokens occupy two characters. X means unknown, never physical zero.
    This baseline derives net counts; it does not add a new constituent branch.
    The supplied rounded coefficients and multiplicities are not repaired.
    """
    if value is None or pd.isna(value):
        return np.full(10, np.nan)
    text = str(value).strip().strip("'\"")
    pieces = re.findall(r'[+-]?\d|[Xx]', text)
    if len(pieces) != 10 or ''.join(pieces) != text:
        raise ValueError(f'Unsupported constituent code {value!r}; expected ten signed digits/X fields.')
    return np.array([np.nan if token.lower() == 'x' else int(token) for token in pieces], dtype=np.float64)

def bottom_scope(raw):
    constituents = np.stack(raw['Initial Particle Structure As Int'].map(decode_constituents))
    net = constituents[:, 4] - constituents[:, 9]
    hadron = raw['Initial Particle Type'].isin(['Meson', 'Baryon']).to_numpy()
    return hadron & np.isfinite(net) & (np.abs(net) >= .5)

def adapt_new_schema(raw):
    """Deterministic, target-independent adapter. No particle/label correction.

    Reuse the single-context V5 network with the diagram having the largest
    provided heuristic weight (ties: lowest slot; missing weights: first known
    topology). Weights are selectors only, not probabilities or target factors.
    A cascade topology/transition contributes its first stage; supplied CKM is
    retained. Unrecognized annotations become UNKNOWN with an audit count.
    """
    df = raw.copy()
    report = {'context_projection': 'largest provided diagram weight; first-stage labels',
              'unknown_type_entries': 0, 'unknown_topology_rows': 0, 'unknown_transition_rows': 0}
    names = raw[[f'Diagram {i} Topology' for i in range(1, 4)]].notna().to_numpy()
    weights = np.stack([numeric_column(raw, f'Diagram {i} Relative Amplitude') for i in range(1, 4)], axis=1)
    usable = np.where(names & np.isfinite(weights), weights, -np.inf)
    chosen = usable.argmax(axis=1)
    missing_weight = ~np.isfinite(usable).any(axis=1)
    chosen[missing_weight] = names[missing_weight].argmax(axis=1)
    no_diagram = ~names.any(axis=1)
    topology = np.full(len(raw), np.nan)
    transition = np.full(len(raw), np.nan)
    ckm = np.full(len(raw), np.nan)
    for pos, (_, row) in enumerate(raw.iterrows()):
        if no_diagram[pos]:
            continue
        slot = int(chosen[pos]) + 1
        top = str(row[f'Diagram {slot} Topology']).split(' -> [', 1)[0].strip()
        value = row[f'Diagram {slot} Quark Transition']
        trans = '' if pd.isna(value) else str(value).split(';', 1)[0].strip().replace('→', '->')
        topology[pos] = TOPOLOGY_CODES.get(top, np.nan)
        transition[pos] = TRANSITION_CODES.get(trans, np.nan)
        ckm[pos] = parse_number(row[f'Diagram {slot} Effective CKM Matrix Element'])
        report['unknown_topology_rows'] += int(not np.isfinite(topology[pos]))
        report['unknown_transition_rows'] += int(not np.isfinite(transition[pos]))
    df['Original Decay Type'] = raw['Decay Type']
    df['Selected Diagram'] = np.where(no_diagram, 0, chosen + 1)
    df['Decay Type'] = raw['Decay Type'].map(DECAY_TYPE_CODES)
    df['Decay Topology'], df['Decay Quark Transition'], df['CKM Matrix Element'] = topology, transition, ckm
    for prefix in PREFIXES:
        active = raw[prefix + ' Name'].notna().to_numpy()
        values = np.stack(raw[prefix + ' Structure As Int'].map(decode_constituents))
        net = values[:, :5] - values[:, 5:]
        net[~active] = np.nan
        for j, flavor in enumerate(['u', 'd', 's', 'c', 'b']):
            df[prefix + f' Number of {flavor} Quarks'] = net[:, j]
        df[prefix + ' Type'] = raw[prefix + ' Type'].map(TYPE_CODES)
        report['unknown_type_entries'] += int((active & df[prefix + ' Type'].isna()).sum())
    return df, report

def read_reference_data(path):
    raw = pd.read_csv(path, low_memory=False)
    required = ['Decay Description', 'Decay Value', 'Decay Value Error', 'Initial Particle Type',
                'Initial Particle Structure As Int']
    missing = [c for c in required if c not in raw]
    if missing:
        raise ValueError('New PDG schema required; missing: ' + ', '.join(missing))
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    flags = annotate_issues(raw, digest)
    scope = np.ones(len(raw), dtype=bool)
    selected = raw.loc[scope].copy()
    selected['_source_row'] = np.flatnonzero(scope)
    selected['Decay Value'] = selected['Decay Value'].map(parse_number)
    selected['Decay Value Error'] = selected['Decay Value Error'].map(parse_number)
    valid = selected['Decay Value'].between(0, 1, inclusive='right') & selected['Decay Description'].notna() & selected['Decay Description'].str.strip().ne('')
    if not valid.all():
        raise ValueError('Invalid selected target/description; no row was silently removed: ' + str(selected.loc[~valid, '_source_row'].tolist()[:12]))
    if len(selected) < 12:
        raise ValueError('Need at least 12 training rows.')
    for column in flags:
        selected[column] = flags.loc[scope, column].to_numpy()
    # Reporting fields only. They do not filter rows or enter model features.
    selected['Original Parent Type'] = selected['Initial Particle Type']
    selected['Evaluation Cohort'] = np.where(bottom_scope(selected), 'Open-bottom hadron', 'Other parents')
    adapted, adapter_audit = adapt_new_schema(selected)
    audit = {'csv_rows': len(raw), 'csv_columns': len(raw.columns), 'rows_retained': len(selected),
             'rows_outside_parent_scope': int((~scope).sum()), 'rows_excluded_by_issue_filter': 0,
             'rows_excluded_by_uncertainty_filter': 0, 'parent_filter_applied': False,
             'scope_definition': 'Every row of the Training CSV; all parent species, including hidden-bottom and free-quark records.',
             'parent_counts': selected['Initial Particle Name'].value_counts().to_dict(),
             'parent_type_counts': selected['Original Parent Type'].value_counts(dropna=False).to_dict(),
             'evaluation_cohort_counts': selected['Evaluation Cohort'].value_counts().to_dict(),
             'nonpositive_or_missing_uncertainty_rows': int((~selected['Decay Value Error'].gt(0)).sum()),
             'unique_descriptions': int(selected['Decay Description'].nunique()),
             'issue_counts': {c: int(selected[c].sum()) for c in ISSUE_COLUMNS + ['Any Identified Issue']},
             'known_multiplicity_audit_snapshot': digest == AUDITED_TRAINING_SHA256, 'adapter': adapter_audit}
    return adapted.reset_index(drop=True), audit

def retain_all_rows(reference, reference_audit):
    """Retain every row; flags only describe the population."""
    keep = np.ones(len(reference), dtype=bool)
    selected = reference.loc[keep].reset_index(drop=True).copy()
    if len(selected) < 12:
        raise ValueError('Need at least 12 training rows.')
    audit = copy.deepcopy(reference_audit)
    audit.update({'rows_retained': len(selected), 'rows_excluded_by_issue_filter': int((~keep).sum()),
                  'source_issue_counts': reference_audit['issue_counts'],
                  'issue_counts': {c: int(selected[c].sum()) for c in ISSUE_COLUMNS + ['Any Identified Issue']},
                  'scope_definition': 'Every Training CSV row, every parent species; audit flags and errors are report-only.',
                  'parent_counts': selected['Initial Particle Name'].value_counts().to_dict(),
                  'parent_type_counts': selected['Original Parent Type'].value_counts(dropna=False).to_dict(),
                  'evaluation_cohort_counts': selected['Evaluation Cohort'].value_counts().to_dict(),
                  'nonpositive_or_missing_uncertainty_rows': int((~selected['Decay Value Error'].gt(0)).sum()),
                  'unique_descriptions': int(selected['Decay Description'].nunique())})
    # Adapter counts in read_reference_data refer to the original population.
    audit['reference_adapter'] = audit.pop('adapter')
    return selected, audit

def read_training_data(path):
    reference, audit = read_reference_data(path)
    return retain_all_rows(reference, audit)

def export_row_selection(reference, output):
    columns = ['_source_row', 'Decay Description', 'Decay Value', 'Initial Particle Name',
               'Original Parent Type', 'Evaluation Cohort'] + ISSUE_COLUMNS + ['Any Identified Issue',
               'Multiplicity Audit Snapshot Available']
    rows = reference[columns].copy()
    rows['CSV Record'] = rows['_source_row']+1
    rows['Issue Notes'] = ['; '.join(c for c in ISSUE_COLUMNS if row[c]) for _, row in rows.iterrows()]
    rows.to_csv(output/'all_training_rows.csv', index=False)
    rows.loc[rows['Any Identified Issue']].to_csv(output/'flagged_rows_report_only.csv', index=False)


def comparison_splits(reference, retained, cfg, path):
    """Build full-population outer and inner GroupKFold plans, without exclusions."""
    full_groups, full_provenance, audit = validation_groups(reference, cfg, path)
    keep = np.ones(len(reference), dtype=bool)
    np.testing.assert_array_equal(retained['_source_row'], reference.loc[keep, '_source_row'])
    mapping = np.full(len(reference), -1, dtype=int)
    mapping[keep] = np.arange(len(retained))
    def restricted(indices):
        result = mapping[indices]
        return result[result >= 0]
    plan = []
    full_fold_ids = np.zeros(len(reference), dtype=int)
    for fold, (full_dev, full_test) in enumerate(GroupKFold(cfg.outer_folds).split(reference, groups=full_groups), 1):
        full_fold_ids[full_test] = fold
        development, test = restricted(full_dev), restricted(full_test)
        if len(development) == 0 or len(test) == 0:
            raise ValueError('An outer split is empty.')
        inner = []
        for train_pos, val_pos in GroupKFold(cfg.inner_folds).split(full_dev, groups=full_groups[full_dev]):
            train, val = restricted(full_dev[train_pos]), restricted(full_dev[val_pos])
            if len(train) == 0 or len(val) == 0:
                raise ValueError('An inner split is empty.')
            inner.append((train, val))
        plan.append((development, test, inner))
    groups = full_groups[keep]
    provenance = full_provenance.loc[keep].reset_index(drop=True)
    manifest = reference[['_source_row', 'Decay Description', 'Any Identified Issue']].copy()
    manifest['Validation Group'], manifest['Outer Fold'] = full_groups, full_fold_ids
    manifest['Included In Experiment'] = keep
    audit = dict(audit, n_reference_rows=len(reference), n_reference_groups=len(set(full_groups)),
                 n_groups=len(set(groups)), n_retained_rows=len(retained),
                 split_policy='Outer and inner GroupKFold on all Training rows; same source grouping as all-row ParT.')
    return groups, provenance, audit, plan, manifest

def tokenize(df, max_daughters=10):
    """Fixed vocabularies; no data-derived fitting occurs here.

    Continuous status: 0 missing, 1 positive numeric value, 2 explicitly zero.
    Mass zero is a valid numerical mass. Width zero has a state but no finite log.
    """
    required = [c for c, _ in CTX_CATEGORIES] + CTX_CONTINUOUS + ['Is Decay Radiative']
    for prefix in PREFIXES:
        required += [prefix + ' ' + suffix for suffix, *_ in QUANTUM_SPECS]
        required += [prefix + ' ' + suffix for suffix in ['Name', 'Mass', 'Width', 'Multiplicity']]
    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError('Dataset schema is missing: ' + ', '.join(missing[:10]))
    n, p = len(df), max_daughters + 2
    result = {
        'ctx_cat': np.zeros((n, 4), dtype=np.int64),
        'ctx_cont': np.zeros((n, 3), dtype=np.float32),
        'ctx_known': np.zeros((n, 3), dtype=bool),
        'part_cat': np.zeros((n, p, 12), dtype=np.int64),
        'part_cont': np.zeros((n, p, 2), dtype=np.float32),
        'part_status': np.zeros((n, p, 2), dtype=np.int64),
        'mask': np.ones((n, p), dtype=bool),
    }
    audit = {'quantum_unknown_source_entries': {}, 'positive_widths_below_old_floor': 0,
             'missing_source_widths': 0, 'reported_zero_source_widths': 0}
    for j, (col, maximum) in enumerate(CTX_CATEGORIES):
        result['ctx_cat'][:, j] = encode_category(numeric_column(df, col), 1, 0, maximum, col)
    bool_map = {'true': 2, '1': 2, '1.0': 2, 'false': 1, '0': 1, '0.0': 1}
    result['ctx_cat'][:, 3] = df['Is Decay Radiative'].map(
        lambda v: bool_map.get(str(v).strip().lower(), 0)).to_numpy(dtype=np.int64)
    for j, col in enumerate(CTX_CONTINUOUS):
        x = numeric_column(df, col)
        known = np.isfinite(x)
        if j == 2:
            known &= x > 0
            x = np.log10(np.where(known, x, 1.))
        result['ctx_cont'][:, j] = np.where(known, x, 0.)
        result['ctx_known'][:, j] = known
    extracted = []
    for prefix in PREFIXES:
        active = (df[prefix+' Name'].notna() & df[prefix+' Name'].astype(str).str.strip().ne('')).to_numpy()
        cats = np.zeros((n, 12), dtype=np.int64)
        cont = np.zeros((n, 2), dtype=np.float32)
        status = np.zeros((n, 2), dtype=np.int64)
        for j, (suffix, factor, offset, maximum, _) in enumerate(QUANTUM_SPECS):
            x = numeric_column(df, prefix+' '+suffix)
            x[~active] = np.nan
            if j == 3:
                cats[:, j] = encode_charge(x, prefix+' '+suffix)
            elif j in EXPANDED_FLAVOR_COLUMNS:
                cats[:, j] = encode_signed_quark_count(x, prefix+' '+suffix)
            else:
                cats[:, j] = encode_category(x, factor, offset, maximum, prefix+' '+suffix)
            audit['quantum_unknown_source_entries'][suffix] = audit['quantum_unknown_source_entries'].get(suffix, 0) + int((active & ~np.isfinite(x)).sum())
        for j, suffix in enumerate(['Mass', 'Width']):
            x = numeric_column(df, prefix+' '+suffix)
            valid = active & np.isfinite(x) & (x >= 0)
            positive = valid & (x > 0)
            status[positive, j] = 1
            status[valid & (x == 0), j] = 2
            if j == 0:
                cont[:, j] = np.where(valid, x, 0.)
            else:
                # Log in float64 first: even widths near 1e-28 keep their information.
                cont[:, j] = np.log10(np.where(positive, x, 1.))
                audit['positive_widths_below_old_floor'] += int((positive & (x < 1e-16)).sum())
                audit['missing_source_widths'] += int((active & ~valid).sum())
                audit['reported_zero_source_widths'] += int((active & valid & (x == 0)).sum())
        mult_values = numeric_column(df, prefix+' Multiplicity')
        mult = np.where(np.isfinite(mult_values), mult_values, 1.)
        if (active & ((mult < 1) | (np.abs(mult - np.rint(mult)) > 1e-5))).any():
            raise ValueError(f'{prefix}: multiplicities must be positive integers.')
        extracted.append((active, cats, cont, status, np.rint(mult).astype(int)))
    if not extracted[0][0].all():
        raise ValueError('Every row must have a named initial particle.')
    for slot in [0, 1]:
        active, cats, cont, status, _ = extracted[slot]
        result['part_cat'][:, slot] = cats
        result['part_cont'][:, slot] = cont
        result['part_status'][:, slot] = status
        result['mask'][:, slot] = ~active
    for row in range(n):
        cursor = 2
        for active, cats, cont, status, multiplicity in extracted[2:]:
            if not active[row]:
                continue
            end = cursor + multiplicity[row]
            if end > p:
                raise ValueError(f'Row {row}: daughter multiplicity exceeds max_daughters={max_daughters}; no truncation is allowed.')
            for key, value in [('part_cat', cats), ('part_cont', cont), ('part_status', status)]:
                result[key][row, cursor:end] = value[row]
            result['mask'][row, cursor:end] = False
            cursor = end
        if cursor == 2:
            raise ValueError(f'Row {row}: no active daughter particles.')
    audit['expanded_active_tokens'] = int((~result['mask']).sum())
    audit['expanded_daughter_counts'] = pd.Series((~result['mask'][:, 2:]).sum(1)).value_counts().sort_index().to_dict()
    audit['active_widths_below_old_floor'] = int(((result['part_cont'][:, :, 1] < -16) & (result['part_status'][:, :, 1] == 1) & ~result['mask']).sum())
    return result, audit

def metrics(y, pred):
    error = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    absolute = np.abs(error)
    r2 = float(r2_score(y, pred)) if len(y) > 1 and np.std(y) > 1e-12 else None
    correlation = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and np.std(y) > 1e-12 and np.std(pred) > 1e-12 else None
    return {'n': len(error), 'log10_R2': r2, 'pearson_r': correlation,
            'log10_MAE': float(absolute.mean()), 'log10_MedAE': float(np.median(absolute)),
            'log10_RMSE': float(np.sqrt(np.mean(error ** 2))),
            'geometric_mean_error_factor': float(10 ** absolute.mean()),
            'median_error_factor': float(10 ** np.median(absolute)),
            **{f'within_factor_{factor}': float((absolute <= math.log10(factor)).mean()) for factor in [2, 3, 5, 10]}}

def print_metrics(values, name):
    print(f"\n{name}: n={values['n']} | R2={values['log10_R2']:.5f} | MAE={values['log10_MAE']:.5f} dex | RMSE={values['log10_RMSE']:.5f} dex", flush=True)
    print(f"  Median factor={values['median_error_factor']:.3f}x | Geometric mean factor={values['geometric_mean_error_factor']:.3f}x | Within 2x={100*values['within_factor_2']:.2f}% | Within 10x={100*values['within_factor_10']:.2f}%", flush=True)

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')

def discover_file(requested, default_name, adjacent=None):
    direct = Path(requested)
    if direct.is_file():
        return direct.resolve()
    if requested != default_name:
        raise FileNotFoundError(f'Explicit file not found: {requested}')
    if adjacent is not None and (Path(adjacent) / default_name).is_file():
        return (Path(adjacent) / default_name).resolve()
    local = Path('new dataset 30-10-2026') / default_name
    if local.is_file():
        return local.resolve()
    matches = sorted(Path('/kaggle/input').rglob(default_name)) if Path('/kaggle/input').exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f'Multiple {default_name} files found. Set an explicit path: ' + str(matches))
    raise FileNotFoundError(f'Attach {default_name} to Kaggle or set its Config/CLI path explicitly.')

def resolve_path(cfg):
    return discover_file(cfg.data_path, Config().data_path)

def validation_groups(df, cfg, training_path):
    """Keep source expansions/conjugates together without removing any rows.

    Source lookup uses the stored target only to recover its record identifier;
    the identifier and target are never model input features. Both inner and
    outer folds use the same group definition. Exact descriptions are also
    unioned, including when their source records differ.
    """
    if cfg.validation_grouping == 'description':
        return df['Decay Description'].astype(str).to_numpy(), pd.DataFrame(index=df.index), {
            'grouping': 'description', 'source_grouping_enabled': False,
            'note': 'Historical-style grouping; related source expansions can cross folds. Not a family-held-out benchmark.'}
    if cfg.validation_grouping != 'source':
        raise ValueError('validation_grouping must be source or description.')
    source_path = discover_file(cfg.source_data_path, Config().source_data_path, Path(training_path).parent)
    source = pd.read_csv(source_path, low_memory=False)
    by_description = {key: group for key, group in source.groupby('Decay Description')}
    pids = []
    for _, row in df.iterrows():
        candidates = by_description.get(row['Decay Description'])
        if candidates is None:
            raise ValueError('Cannot recover source ID for ' + row['Decay Description'])
        matched = candidates[np.isclose(candidates['Decay Value'], row['Decay Value'], rtol=1e-10, atol=0.)]
        if len(matched) != 1:
            raise ValueError('Ambiguous/missing source measurement for ' + row['Decay Description'])
        pids.append(str(matched.iloc[0]['PDG ID']))
    roots = [re.sub(r'(/\d{4}).*$', r'\1', pid) for pid in pids]
    parent = list(range(len(df)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for labels in [roots, df['Decay Description'].tolist()]:
        seen = {}
        for i, label in enumerate(labels):
            if label in seen:
                left, right = find(i), find(seen[label])
                parent[max(left, right)] = min(left, right)
            else:
                seen[label] = i
    groups = np.array([f'source_group_{find(i):05d}' for i in range(len(df))])
    provenance = pd.DataFrame({'Source PDG ID': pids, 'Base Source PDG ID': roots}, index=df.index)
    return groups, provenance, {'grouping': 'source', 'source_grouping_enabled': True,
                                'source_path': str(source_path), 'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                                'n_groups': len(set(groups)), 'note': 'Source/description groups; not complete decay-family holdout.'}

FROZEN_AUDIT_RECORDS = {'Issue Final Multiplicity': [489, 1129, 1139, 1239, 1247, 1248, 1424, 1516, 1654, 1727, 2559, 2654], 'Issue Identical Inputs': [12, 14, 15, 16, 17, 72, 73, 78, 79, 109, 134, 163, 164, 204, 205, 219, 264, 267, 268, 284, 285, 289, 294, 302, 328, 337, 346, 347, 348, 351, 359, 360, 361, 375, 376, 377, 380, 404, 409, 411, 412, 436, 439, 440, 445, 450, 451, 452, 453, 456, 457, 458, 489, 495, 497, 507, 511, 514, 523, 538, 556, 576, 593, 624, 625, 626, 627, 628, 648, 668, 674, 679, 680, 682, 705, 721, 741, 756, 813, 859, 875, 888, 889, 895, 896, 899, 902, 903, 937, 938, 939, 940, 946, 947, 948, 952, 955, 960, 961, 962, 963, 992, 1007, 1122, 1128, 1129, 1136, 1139, 1239, 1240, 1247, 1248, 1373, 1412, 1424, 1436, 1530, 1538, 1621, 1654, 1664, 1674, 1723, 1727, 1887, 1934, 2029, 2030, 2047, 2048, 2056, 2057, 2058, 2059, 2062, 2063, 2072, 2073, 2074, 2079, 2096, 2098, 2099, 2103, 2104, 2111, 2112, 2121, 2122, 2158, 2173, 2174, 2183, 2190, 2191, 2192, 2193, 2194, 2197, 2198, 2199, 2200, 2256, 2257, 2264, 2265, 2266, 2267, 2268, 2269, 2272, 2274, 2286, 2287, 2288, 2289, 2310, 2333, 2401, 2423, 2424, 2454, 2455, 2456, 2457, 2459, 2460, 2461, 2483, 2487, 2491, 2492, 2493, 2494, 2497, 2498, 2499, 2502, 2505, 2509, 2514, 2519, 2525, 2526, 2534, 2536, 2564, 2566, 2567, 2568, 2569, 2572, 2579, 2587, 2588, 2593, 2595, 2615, 2631, 2642, 2643, 2647, 2648, 2649, 2650, 2652, 2654, 2667, 2670, 2685, 2695, 2706, 2707, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2749, 2752, 2765, 2766, 2781, 2794, 2798, 2804, 2808, 2810, 2813, 2814, 2815, 2816, 2821, 2822, 2830, 2835, 2836, 2849, 2850, 2853, 2854, 2859, 2875, 2882, 2887, 2891, 2902, 2917, 2920, 2939, 2943, 2963, 2969, 2970, 2971, 2975, 2982, 3006, 3007, 3013, 3016, 3021, 3028, 3053, 3054, 3067, 3068, 3076, 3079, 3128, 3132, 3163, 3165, 3168, 3169, 3170, 3171, 3172, 3175, 3176, 3177, 3178, 3179, 3180, 3181, 3182, 3183, 3184, 3185, 3186, 3187, 3188, 3189, 3191, 3192, 3193, 3194, 3195, 3196, 3197, 3198, 3199, 3200, 3203, 3204, 3206, 3207, 3208, 3209, 3210, 3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3220, 3221, 3222, 3223, 3224, 3227, 3229, 3232, 3233, 3234, 3235, 3237, 3238, 3240, 3241, 3243, 3244, 3245, 3246, 3247, 3263, 3268, 3303, 3309, 3310, 3311, 3314, 3315, 3316, 3317, 3318, 3320, 3321, 3323, 3324, 3325, 3326, 3338, 3339, 3341, 3342, 3344, 3345, 3346, 3347, 3348, 3349, 3350, 3351, 3355, 3357, 3358, 3359, 3384, 3385, 3386, 3387, 3388, 3389, 3390, 3391, 3392, 3393, 3394, 3395, 3396, 3397, 3398, 3399, 3400, 3401, 3402, 3403, 3406, 3407, 3408, 3409, 3410, 3411, 3412, 3413, 3414, 3415, 3416, 3417, 3418, 3419, 3420, 3421, 3423, 3424, 3425, 3434, 3435, 3436, 3437, 3438, 3439, 3443], 'Issue Target Type': [556, 562, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 2421, 2422, 2423, 2426, 2433, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2808, 2809, 2837, 2842, 2857, 2858], 'Issue Charge': [98, 99, 118, 119, 135, 136, 139, 140, 221, 222, 240, 669, 686, 723, 776, 942, 943, 953, 954, 957, 958, 959, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 983, 1229, 1231, 1261, 1262, 2037, 2045, 2047, 2050, 2054, 2056, 2058, 2062, 2067, 2072, 2079, 2089, 2096, 2098, 2103, 2111, 2115, 2121, 2126, 2222, 2379, 2380, 2444, 2488, 2489, 2517, 2518, 2521, 2523, 2557, 2559, 2693, 2719, 2722, 2790, 2952, 2957, 3169, 3170, 3176, 3177, 3181, 3182, 3186, 3187, 3192, 3193, 3200, 3204, 3206, 3207, 3211, 3212, 3215, 3216, 3220, 3221], 'Issue Self Conjugate': [2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 49, 59, 62, 70, 71, 97, 176, 178, 188, 191, 194, 202, 210, 212, 213, 216, 353, 459, 460, 462, 739, 741, 743, 744, 938, 940, 941, 959, 962, 963, 1069, 1070, 1071, 1072, 1073, 1118, 1120, 1121, 1569, 1703, 1716, 1717, 1906, 1908, 1913, 1914, 1916, 1965, 1966, 2013, 2014, 2155, 2211, 2217, 2294, 2298, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2474, 2475, 2477, 2479, 2887, 3306, 3307, 3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320, 3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3420, 3421, 3422, 3423, 3424, 3425, 3426, 3427, 3428, 3429, 3430, 3431, 3432, 3433, 3434, 3435, 3436, 3437, 3438, 3439, 3440, 3441, 3444, 3445, 3446, 3447, 3448, 3449]}

def annotate_issues(raw, dataset_hash):
    """Use the PREVIOUSLY FROZEN audit; do not discover conflicts from test labels."""
    known_snapshot = dataset_hash == AUDITED_TRAINING_SHA256
    flags = pd.DataFrame(index=raw.index)
    records = np.arange(1, len(raw)+1)
    for column in ISSUE_COLUMNS:
        flags[column] = np.isin(records, FROZEN_AUDIT_RECORDS[column]) if known_snapshot else False
    flags['Any Identified Issue'] = flags[ISSUE_COLUMNS].any(axis=1)
    flags['Multiplicity Audit Snapshot Available'] = known_snapshot
    return flags


def canonicalize_tokens(tokens):
    """Sort complete daughter records by model-visible properties only."""
    result = {k: v.copy() for k, v in tokens.items()}
    for row in range(len(tokens['mask'])):
        def key(slot):
            return (int(tokens['mask'][row, slot]),
                    *tokens['part_cat'][row, slot].tolist(),
                    *tokens['part_status'][row, slot].tolist(),
                    *tokens['part_cont'][row, slot].tolist())
        order = [0, 1] + sorted(range(2, tokens['mask'].shape[1]), key=key)
        for name in ['part_cat', 'part_status', 'part_cont', 'mask']:
            result[name][row] = tokens[name][row, order]
    return result


def physics_features(t):
    """Raw-unit counterparts of ParT flow/highway information; no fitted stats."""
    cat, active = t['part_cat'], ~t['mask'][:, 2:]
    columns, names = [], []
    def add(name, values):
        columns.append(np.asarray(values, dtype=np.float32)); names.append(name)
    defects = []
    for index, name in [(3, 'charge'), (7, 'u'), (8, 'd'), (9, 's'), (10, 'c'), (11, 'b'), (1, 'spin')]:
        ids = cat[:, :, index]
        if index == 1:
            values = (ids.astype(float)-1)/2
        else:
            values = ids.astype(float)-3
            if index == 3:
                for category, charge in FRACTIONAL_CHARGE_IDS.items():
                    values = np.where(ids == category, charge, values)
            elif index in EXPANDED_FLAVOR_COLUMNS:
                values = np.where(ids == 7, -3, values)
        valid = (ids[:, 0] > 0) & ((ids[:, 2:] > 0) | ~active).all(1)
        defect = values[:, 0] - (values[:, 2:]*active).sum(1)
        defect = np.where(valid, defect, np.nan)
        defects.append(defect); add('flow_delta_'+name, defect)
    add('flow_total_abs_net_flavor_change', np.abs(np.stack(defects[1:6], 1)).sum(1))
    count = active.sum(1); add('daughter_count', count)
    q, momentum, log_ckm = t['ctx_cont'].T
    mass = t['part_cont'][:, 0, 0]
    valid = t['ctx_known'][:, :2].all(1) & (t['part_status'][:, 0, 0] > 0) & (mass > 0) & (q > 0) & (momentum > 0) & (count >= 2)
    safe_mass = np.maximum(mass, 1.)  # Match the original proxy; no new physics.
    phi = np.log10(np.clip(2*momentum/safe_mass, 1e-5, 2.)) + (count-2)*np.log10(np.clip(q/(4*np.pi*safe_mass), 1e-5, 1.))
    add('phase_space_proxy', np.where(valid, phi, np.nan))
    measured = t['ctx_known'][:, 2] & (log_ckm < -.1)
    add('ckm_measured_gate', measured)
    add('gated_log10_ckm', np.where(measured, log_ckm, 0.))
    add('semileptonic_annotation', t['ctx_cat'][:, 0] == 2)
    dcat = cat[:, 2:]
    kind, charge = dcat[:, :, 0], dcat[:, :, 3]
    leptons = active & (kind == LEPTON_TYPE_ID+1)
    neutral = leptons & (charge == 3)
    positive = leptons & (charge > 3)
    negative = leptons & (charge > 0) & (charge < 3)
    hadrons = active & np.isin(kind, [x+1 for x in HADRON_TYPE_IDS])
    known = ((kind > 0) | ~active).all(1) & ((charge > 0) | ~leptons).all(1)
    add('semileptonic_structure', np.where(known, hadrons.any(1) & neutral.any(1) & (positive | negative).any(1), np.nan))
    add('dilepton_structure', np.where(known, ~neutral.any(1) & positive.any(1) & negative.any(1), np.nan))
    return np.column_stack(columns), names


def feature_matrix(df, max_daughters=10):
    """Fixed-vocabulary encoding: no learned preprocessing or label inspection."""
    tokens, audit = tokenize(df, max_daughters)
    t = canonicalize_tokens(tokens)
    pieces, names = [], []
    def add(name, values):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 1:
            values = values[:, None]
        pieces.append(values)
        names.extend([name] if values.shape[1] == 1 else [f'{name}_{i}' for i in range(values.shape[1])])
    for j, (name, capacity) in enumerate(zip(['decay_type', 'topology', 'transition', 'radiative'], [9, 18, 13, 3])):
        add('context_'+name, np.eye(capacity, dtype=np.float32)[t['ctx_cat'][:, j]])
    for j, name in enumerate(['Q', 'P_star', 'log10_ckm']):
        add('context_'+name, np.where(t['ctx_known'][:, j], t['ctx_cont'][:, j], np.nan))
    capacities = [spec[3]+2 for spec in QUANTUM_SPECS]
    capacities[3] = 11
    for slot in range(max_daughters+2):
        role = 'parent' if slot == 0 else 'intermediate' if slot == 1 else f'daughter_rank_{slot-1}'
        add(role+'_present', ~t['mask'][:, slot])
        for j, spec in enumerate(QUANTUM_SPECS):
            suffix = re.sub('[^A-Za-z0-9_]', '_', spec[0])
            add(role+'_'+suffix, np.eye(capacities[j], dtype=np.float32)[t['part_cat'][:, slot, j]])
        for j, name in enumerate(['mass', 'log10_width']):
            status = t['part_status'][:, slot, j]
            known = status > 0 if j == 0 else status == 1
            add(role+'_'+name, np.where(known & ~t['mask'][:, slot], t['part_cont'][:, slot, j], np.nan))
            add(role+'_'+name+'_status', np.eye(3, dtype=np.float32)[status])
    physical, physical_names = physics_features(t)
    pieces.append(physical); names.extend(physical_names)
    matrix = np.concatenate(pieces, axis=1).astype(np.float32)
    if np.isinf(matrix).any() or len(set(names)) != len(names):
        raise AssertionError('Invalid feature matrix/names.')
    return matrix, names, audit


def target_stats(y, train):
    values = np.asarray(y)[train]
    return {'mean': float(values.mean()), 'scale': max(float(values.std()), 1e-6), 'n_training_rows': len(train)}


def make_model(params, cfg, seed):
    return xgb.XGBRegressor(**params, objective=cfg.objective,
                           **({'huber_slope': cfg.huber_delta_normalized} if cfg.objective == 'reg:pseudohubererror' else {}),
                           tree_method='hist', device=cfg.device, n_jobs=cfg.n_jobs,
                           random_state=seed, base_score=0., verbosity=0)


def fit_model(X, y, train, params, cfg, seed):
    stats = target_stats(y, train)
    model = make_model(params, cfg, seed)
    model.fit(X[train], (y[train]-stats['mean'])/stats['scale'])
    return model, stats


def predict_model(model, X, stats, cfg):
    raw = model.predict(X).astype(float)*stats['scale']+stats['mean']
    if not np.isfinite(raw).all():
        raise FloatingPointError('Nonfinite model predictions.')
    return np.clip(raw, cfg.log_prediction_min, cfg.log_prediction_max), raw


def validate_splits(splits, development, groups):
    counts = np.zeros(len(groups), dtype=int)
    for train, val in splits:
        if set(train) & set(val) or set(groups[train]) & set(groups[val]):
            raise AssertionError('Inner group overlap.')
        if set(train) | set(val) != set(development):
            raise AssertionError('Inner split does not cover the development population.')
        counts[val] += 1
    if not (counts[development] == 1).all():
        raise AssertionError('Incomplete/duplicated inner validation coverage.')


def search_parameters(X, y, groups, development, splits, cfg, seed, folder):
    """This function receives no outer test indices or outer test scores."""
    folder.mkdir(parents=True, exist_ok=True)
    validate_splits(splits, development, groups)
    history = []
    def objective(trial):
        params = {'n_estimators': trial.suggest_int('n_estimators', cfg.min_trees, cfg.max_trees, log=True),
                  'max_depth': trial.suggest_int('max_depth', 2, 7),
                  'learning_rate': trial.suggest_float('learning_rate', .008, .15, log=True),
                  'min_child_weight': trial.suggest_float('min_child_weight', .1, 20., log=True),
                  'subsample': trial.suggest_float('subsample', .65, 1.),
                  'colsample_bytree': trial.suggest_float('colsample_bytree', .55, 1.),
                  'reg_alpha': trial.suggest_float('reg_alpha', 1e-7, 5., log=True),
                  'reg_lambda': trial.suggest_float('reg_lambda', .1, 40., log=True),
                  'gamma': trial.suggest_float('gamma', 0., .5)}
        sse, n = 0., 0
        for inner, (train, val) in enumerate(splits, 1):
            model, stats = fit_model(X, y, train, params, cfg, seed+inner*100)
            pred, _ = predict_model(model, X[val], stats, cfg)
            error = pred-y[val]
            fold_sse = float(np.square(error).sum())
            sse += fold_sse; n += len(val)
            history.append({'trial': trial.number, 'inner_fold': inner, 'n_train': len(train), 'n_val': len(val),
                            'sse': fold_sse, 'target_mean': stats['mean'], 'target_scale': stats['scale']})
        return sse/n
    # Separate studies, fixed seeds, sequential trials, and no test-driven pruning.
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=seed),
                               pruner=optuna.pruners.NopPruner())
    def callback(study, trial):
        study.trials_dataframe().to_csv(folder/'trials.csv', index=False)
        pd.DataFrame(history).to_csv(folder/'inner_scores.csv', index=False)
        print(f'  Trial {trial.number+1}/{cfg.n_trials}: pooled inner RMSE={math.sqrt(trial.value):.5f}; '
              f'best={math.sqrt(study.best_value):.5f}', flush=True)
    study.optimize(objective, n_trials=cfg.n_trials, n_jobs=1, callbacks=[callback])
    write_json(folder/'best_parameters.json', {'params': study.best_params, 'inner_mse': study.best_value,
                                             'best_trial': study.best_trial.number})
    return study.best_params, study.best_value


def save_model(folder, model, stats, params, cfg, names, dataset_hash):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_model(folder/'model.ubj')
    write_json(folder/'metadata.json', {'format': 'new_pdg_unfiltered_xgb_optuna_v1', 'config': asdict(cfg),
                                      'target_stats': stats, 'params': params, 'feature_names': names,
                                      'dataset_sha256': dataset_hash, 'schema': 'fixed_one_hot_sorted_daughters_v1',
                                      'xgboost_version': xgb.__version__})


def predict_dataframe(raw, model_folder):
    folder = Path(model_folder)
    metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('format') != 'new_pdg_unfiltered_xgb_optuna_v1':
        raise ValueError('Unexpected checkpoint format.')
    cfg = Config(**metadata['config'])
    if len(raw) == 0:
        return pd.DataFrame(columns=['Predicted log10 Value', 'Predicted Decay Value', 'Unclipped log10 Value'], index=raw.index)
    adapted, _ = adapt_new_schema(raw)
    X, names, _ = feature_matrix(adapted, cfg.max_daughters)
    if names != metadata['feature_names']:
        raise ValueError('Feature schema mismatch.')
    model = xgb.XGBRegressor(); model.load_model(folder/'model.ubj')
    model.set_params(device=cfg.device, n_jobs=cfg.n_jobs)
    pred, unclipped = predict_model(model, X, metadata['target_stats'], cfg)
    return pd.DataFrame({'Predicted log10 Value': pred, 'Predicted Decay Value': 10.**pred,
                         'Unclipped log10 Value': unclipped}, index=raw.index)


def run_experiment(cfg=None):
    cfg = cfg or Config()
    if int(xgb.__version__.split('.')[0]) < 2:
        raise RuntimeError('XGBoost >=2.0 is required (device API and pseudo-Huber support).')
    if cfg.n_trials < 1 or cfg.min_trees < 1 or cfg.max_trees < cfg.min_trees:
        raise ValueError('Invalid Optuna/tree budget.')
    if cfg.log_prediction_min >= cfg.log_prediction_max:
        raise ValueError('Invalid prediction bounds.')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output = Path(cfg.output_dir); output.mkdir(parents=True, exist_ok=True)
    path = resolve_path(cfg); start = time.perf_counter()
    reference, reference_audit = read_reference_data(path)
    df, audit = retain_all_rows(reference, reference_audit)
    groups, provenance, group_audit, plan, manifest = comparison_splits(reference, df, cfg, path)
    X, names, encoding_audit = feature_matrix(df, cfg.max_daughters)
    y = np.log10(df['Decay Value'].to_numpy(float))
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    export_row_selection(reference, output)
    manifest.to_csv(output/'reference_fold_assignments.csv', index=False)
    inner_records = []
    for outer, (_, _, splits) in enumerate(plan, 1):
        for inner, (_, val) in enumerate(splits, 1):
            inner_records.extend({'outer_fold': outer, 'inner_validation_fold': inner,
                                  '_source_row': int(df['_source_row'].iloc[i])} for i in val)
    pd.DataFrame(inner_records).to_csv(output/'inner_fold_assignments.csv', index=False)
    write_json(output/'feature_names.json', names)
    write_json(output/'data_audit.json', dict(audit, encoding=encoding_audit, validation=group_audit, n_features=X.shape[1]))
    write_json(output/'run_config.json', {'config': asdict(cfg), 'dataset_sha256': dataset_hash,
                                        'xgboost': xgb.__version__, 'optuna': optuna.__version__,
                                        'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__})
    banner('XGBOOST + OPTUNA — ALL TRAINING ROWS, NO ISSUE OR UNCERTAINTY FILTERS')
    print(f'Rows={len(df)} | excluded={audit["rows_excluded_by_issue_filter"]} | features={X.shape[1]} | '
          f'outer x inner={cfg.outer_folds} x {cfg.inner_folds} | trials per study={cfg.n_trials}', flush=True)
    print(f'XGBoost={xgb.__version__} | Optuna={optuna.__version__} | sklearn={sklearn.__version__} | device={cfg.device}', flush=True)
    print('Fresh Optuna study inside each outer development set. No early stopping; tree count is an inner-search parameter.', flush=True)
    if not audit['known_multiplicity_audit_snapshot']:
        print('Prior issue snapshot unavailable for this file; false report flags do not certify clean rows.', flush=True)
    print('Every training row retained. All audit flags are report-only; no label repairs or uncertainty weights.', flush=True)
    print(f'Report-only flags: {audit["issue_counts"]}', flush=True)
    oof = np.full(len(df), np.nan); raw_oof = oof.copy(); folds = np.zeros(len(df), int); coverage = folds.copy()
    fold_results = []
    for fold, (development, test, splits) in enumerate(plan, 1):
        if set(groups[development]) & set(groups[test]):
            raise AssertionError('Outer group overlap.')
        banner(f'OUTER FOLD {fold}: development={len(development)}, untouched test={len(test)}')
        params, inner_mse = search_parameters(X, y, groups, development, splits, cfg, cfg.seed+fold*10000,
                                              output/'optuna'/f'outer_{fold}')
        model, stats = fit_model(X, y, development, params, cfg, cfg.seed+fold*100)
        pred, raw = predict_model(model, X[test], stats, cfg)
        oof[test], raw_oof[test], folds[test] = pred, raw, fold; coverage[test] += 1
        result = dict(metrics(y[test], pred), fold=fold, inner_mse=inner_mse, params=params,
                      unclipped_R2=metrics(y[test], raw)['log10_R2'], n_clipped=int((pred != raw).sum()))
        fold_results.append(result); print_metrics(result, f'OUTER {fold} TEST')
        save_model(output/'models'/f'outer_{fold}', model, stats, params, cfg, names, dataset_hash)
        write_json(output/'fold_metrics.json', fold_results)
    if not (coverage == 1).all() or not np.isfinite(oof).all():
        raise AssertionError('Each retained row needs exactly one outer held-out prediction.')
    results = df[['_source_row', 'Decay Description', 'Initial Particle Name', 'Original Parent Type', 'Evaluation Cohort']].copy()
    results['Validation Group'], results['Outer Fold'] = groups, folds
    results['True Decay Value'], results['True log10 Value'] = df['Decay Value'], y
    results['Predicted log10 Value'], results['Unclipped log10 Value'] = oof, raw_oof
    results['Predicted Decay Value'], results['Absolute log10 Error'] = 10.**oof, np.abs(oof-y)
    for c in list(provenance.columns) + ISSUE_COLUMNS + ['Any Identified Issue']:
        results[c] = provenance[c].to_numpy() if c in provenance else df[c].to_numpy()
    results.to_csv(output/'xgb_oof_predictions.csv', index=False)
    results[['_source_row', 'Outer Fold', 'Validation Group']].to_csv(output/'fold_assignments.csv', index=False)
    issue_metrics = []
    for field in ISSUE_COLUMNS + ['Any Identified Issue']:
        for flag, subset in results.groupby(field):
            issue_metrics.append({'issue': field, 'flag': bool(flag), **metrics(subset['True log10 Value'], subset['Predicted log10 Value'])})
    pd.DataFrame(issue_metrics).to_csv(output/'issue_subgroup_metrics.csv', index=False)
    population = []
    for field in ['Original Parent Type', 'Evaluation Cohort', 'Initial Particle Name']:
        for value, subgroup in results.groupby(field, dropna=False):
            population.append({'group_by': field, 'group_value': str(value),
                               **metrics(subgroup['True log10 Value'], subgroup['Predicted log10 Value'])})
    pd.DataFrame(population).to_csv(output/'population_metrics.csv', index=False)
    pooled = dict(metrics(y, oof), unclipped_R2=metrics(y, raw_oof)['log10_R2'], n_clipped=int((oof != raw_oof).sum()))
    print_metrics(pooled, 'POOLED OOF — ALL TRAINING ROWS')
    print(pd.DataFrame([r for r in population if r['group_by'] != 'Initial Particle Name'])[
        ['group_by', 'group_value', 'n', 'log10_R2', 'log10_MAE', 'log10_RMSE']].to_string(index=False), flush=True)
    summary = {'pooled_oof': pooled, 'folds': fold_results, 'n_rows': len(df), 'n_features': len(names),
               'population_metrics': population, 'issue_subgroup_metrics': issue_metrics, 'dataset_sha256': dataset_hash, 'config': asdict(cfg),
               'validation': group_audit, 'row_selection': 'All Training CSV rows; no parent, issue or uncertainty filters.', 'rows_excluded_by_issue_filter': 0, 'issue_counts': audit['issue_counts']}
    write_json(output/'metrics.json', summary)
    if cfg.train_full_data_model:
        banner('DEPLOYMENT: SEPARATE FULL-DATA OPTUNA SEARCH — NO INDEPENDENT TEST SCORE')
        full_groups, _, _ = validation_groups(reference, cfg, path)
        keep = np.ones(len(reference), dtype=bool)
        mapping = np.full(len(reference), -1, int); mapping[keep] = np.arange(len(df))
        splits = []
        for full_train, full_val in GroupKFold(cfg.inner_folds).split(reference, groups=full_groups):
            train, val = mapping[full_train], mapping[full_val]
            splits.append((train[train >= 0], val[val >= 0]))
        all_rows = np.arange(len(df))
        params, value = search_parameters(X, y, groups, all_rows, splits, cfg, cfg.seed+999999, output/'optuna'/'deployment')
        model, stats = fit_model(X, y, all_rows, params, cfg, cfg.seed+888888)
        save_model(output/'full_data_model', model, stats, params, cfg, names, dataset_hash)
        summary['deployment'] = {'params': params, 'inner_selection_mse': value, 'n_training_rows': len(df),
                                 'note': 'No separate test score; use nested OOF metrics for evaluation.'}
    summary['elapsed_seconds'] = time.perf_counter()-start
    write_json(output/'metrics.json', summary)
    banner('RUN COMPLETE')
    print(f'Rows={len(df)} | R2={pooled["log10_R2"]:.5f} | RMSE={pooled["log10_RMSE"]:.5f} | '
          f'elapsed={summary["elapsed_seconds"]/60:.1f} min | outputs={output.resolve()}', flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data'); parser.add_argument('--source-data'); parser.add_argument('--output')
    parser.add_argument('--trials', type=int); parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--seed', type=int); parser.add_argument('--smoke-test', action='store_true')
    args, _ = parser.parse_known_args()
    cfg = Config()
    for argument, field in [('data', 'data_path'), ('source_data', 'source_data_path'), ('output', 'output_dir'),
                            ('trials', 'n_trials'), ('device', 'device'), ('seed', 'seed')]:
        value = getattr(args, argument)
        if value is not None:
            setattr(cfg, field, value)
    if args.smoke_test:
        cfg.outer_folds = cfg.inner_folds = 2
        cfg.n_trials, cfg.min_trees, cfg.max_trees = 2, 8, 16
        if args.output is None:
            cfg.output_dir += '_smoke_test'
        banner('SMOKE TEST ONLY — NOT AN ACCURACY ESTIMATE')
    run_experiment(cfg)

