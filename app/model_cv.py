import os
from catboost import Pool, cv, CatBoostClassifier
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from feature_upload import load_features




if __name__ == "__main__":

  df = load_features()
  X = df.drop("target", axis=1)
  y = df["target"]

  cat_features = ["country", "city", "os", "source", "topic", "action", "hour", "gender", "exp_group", "is_weekend"]

  pool = Pool(data=X, label=y, cat_features=cat_features)

  n_splits = 3
  tscv = TimeSeriesSplit(n_splits=n_splits)
  # CatBoost ожидает список (train_idx, test_idx)
  folds = list(tscv.split(X, y))

  param_grid = {
        'loss_function': ['Logloss', 'CrossEntropy'],
        'eval_metric': ['AUC'],
        'iterations': [500, 1000, 1500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4],
        'depth': [4, 6, 8, 10],
        'random_seed': [42],
        'l2_leaf_reg': [1, 3, 5, 7, 9],
        'bagging_temperature': [0, 0.5, 1, 2],
        'border_count': [32, 64, 128],
        'task_type': ['GPU'],
        'devices': ['0']
    }

  grid = list(ParameterGrid(param_grid))
  max_runs = 50
  if len(grid) > max_runs:
    grid = grid[:max_runs]

  best_score = 0
  best_params = None
  best_iteration = None
  best_cv = None

  for params in grid:
    cv_results = cv(
          pool,
          params,
          # вместо fold_count используем заранее подготовленные временные фолды
          folds=folds,
          shuffle=False,  # для time series shuffle выключаем
          stratified=False,
          early_stopping_rounds=30,
          verbose=False,
      )
    metric_name = params['eval_metric']
    metric_column = f"test-{metric_name}-mean"
    score = cv_results[metric_column].max()
    if score > best_score:
        best_score = score
        best_iteration = cv_results[metric_column].idxmax()
        best_params = params
        best_cv = cv_results

  print('Лучший AUC:', best_score)
  print('Лучшие параметры:', best_params)
  print('Лучшая итерация:', best_iteration)

  # обучаем финальную модель на всех данных
  final_model = CatBoostClassifier(**best_params)
  final_model.fit(
      X,
      y,
      cat_features=cat_features,
      verbose=False,
      use_best_model=True,
  )

  base_dir = os.path.dirname(os.path.dirname(__file__))
  model_path = os.path.join(base_dir, "models", "catboost_model.cbm")
  
  final_model.save_model(model_path, format="cbm")

  # Лучшие параметры: {'bagging_temperature': 0, 'border_count': 32, 'depth': 4, 'devices': '0', 'eval_metric': 'AUC', 'iterations': 1500, 'l2_leaf_reg': 5, 'learning_rate': 0.4, 'loss_function': 'CrossEntropy', 'random_seed': 42, 'task_type': 'GPU'}
  # лучший auc - 0.91