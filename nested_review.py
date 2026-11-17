"""Nested temporal search, reserving days 621 onward."""

import json
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from final_audit import metrics
from ferreyros_propension.data import PRICES, PROFILE, PROMOS, load_data
from ferreyros_propension.feature_research import monthly_trend_features


def main():
    start = perf_counter()
    out = Path('outputs/nested_review')
    out.mkdir(parents=True, exist_ok=True)
    data = load_data('Dataset/compras_data.csv')
    events, x = monthly_trend_features(data.loc[data.dia_visita.le(620)])
    y = events.incidencia_compra
    offer_derived = [c for c in x if 'precio' in c or 'promo' in c]
    excluded = set(PRICES + PROMOS + PROFILE + offer_derived)
    excluded.update(c for c in x if c.startswith('fase_'))
    excluded.update(['cliente_conocido', 'dia_relativo'])
    historical = [c for c in x if c not in excluded]
    choices = {
        'history': historical,
        'history_offers': historical + sorted(set(PRICES + PROMOS + offer_derived)),
        'all': historical + sorted(set(PRICES + PROMOS + offer_derived + PROFILE)),
    }
    for c in x.select_dtypes(include=['object', 'string']):
        x[c] = x[c].astype('category')

    def fit_and_predict(parameters, columns, half_life, cut, end):
        train = events.dia_visita.le(cut)
        valid = events.dia_visita.between(cut + 1, end)
        weights = np.exp2((events.loc[train, 'dia_visita'] - cut) / half_life)
        model = lgb.LGBMClassifier(**parameters, verbosity=-1, n_jobs=4, random_state=42)
        model.fit(x.loc[train, columns], y[train], sample_weight=weights)
        return train, valid, model.predict_proba(x.loc[train, columns])[:, 1], model.predict_proba(x.loc[valid, columns])[:, 1]

    def run(trial):
        group = trial.suggest_categorical('group', list(choices))
        half_life = trial.suggest_int('half_life', 90, 360)
        threshold = trial.suggest_float('threshold', 0.23, 0.36)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 150, 650, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.06, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 10),
            'max_depth': trial.suggest_int('max_depth', 2, 4),
            'min_child_samples': trial.suggest_int('min_child_samples', 200, 900),
            'reg_lambda': trial.suggest_float('reg_lambda', 10, 500, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 40, log=True),
        }
        fold_scores = []
        for cut, end in inner_folds:
            train, valid, train_p, p = fit_and_predict(params, choices[group], half_life, cut, end)
            validation = metrics(y[valid], p, threshold)
            fold_scores.append(min(validation['precision'], validation['recall']))
        trial.set_user_attr('threshold', threshold)
        trial.set_user_attr('fold_scores', fold_scores)
        return float(np.mean(fold_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []
    configurations = []
    for cut, end, inner_folds in [
        (404, 512, [(150, 220), (220, 296), (296, 404)]),
        (512, 620, [(220, 296), (296, 404), (404, 512)]),
    ]:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(run, n_trials=20, show_progress_bar=False)
        study.trials_dataframe().to_csv(out / f'trials_{cut}.csv', index=False)
        best = study.best_trial
        configurations.append({'cut': cut, 'parameters': best.params, 'inner_score': best.value})
        values = best.params.copy()
        group = values.pop('group')
        half_life = values.pop('half_life')
        threshold = values.pop('threshold')
        train, valid, train_p, p = fit_and_predict(values, choices[group], half_life, cut, end)
        for label, mask, probabilities in [('train', train, train_p), ('validation', valid, p)]:
            rows.append({'cut': cut, 'end': end, 'split': label, 'n': int(mask.sum()),
                         'threshold': threshold, **metrics(y[mask], probabilities, threshold)})
        print('FINISHED', cut, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    (out / 'configuration.json').write_text(json.dumps({'folds': configurations, 'seconds': perf_counter() - start}, indent=2))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print('SECONDS', round(perf_counter() - start, 1), flush=True)


if __name__ == '__main__':
    main()
