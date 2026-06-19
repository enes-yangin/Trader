from typing import Dict, Tuple, Any, Optional
import optuna
from models.base_model import BaseModel
from models.linear_model import LinearModel
from models.xgb_model import XGBModel
from models.lstm_model import LSTMModel
from utils.config import MODEL, OPTIMIZATION
from utils.types import SplitDict, TrainResult
from utils.logger import get_logger

log = get_logger("optimizer")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_linear(sp: SplitDict, n_trials: int = OPTIMIZATION.linear_n_trials,
                     seed: int = OPTIMIZATION.optuna_seed) -> Tuple[Dict[str, Any], optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        mdl = LinearModel(alpha=alpha)
        mdl.train(sp["X_tr"], sp["y_tr"])
        m = mdl.evaluate(sp["X_val"], sp["y_val"])
        return m["rmse"]

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def optimize_xgb(sp: SplitDict, n_trials: int = OPTIMIZATION.xgb_n_trials,
                  seed: int = OPTIMIZATION.optuna_seed) -> Tuple[Dict[str, Any], optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        mdl = XGBModel(**params)
        mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
        m = mdl.evaluate(sp["X_val"], sp["y_val"])
        return m["rmse"]

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def optimize_lstm(sp: SplitDict, n_trials: int = OPTIMIZATION.lstm_n_trials,
                   seed: int = OPTIMIZATION.optuna_seed,
                   epochs: int = OPTIMIZATION.lstm_opt_epochs) -> Tuple[Dict[str, Any], optuna.Study]:
    def objective(trial: optuna.Trial) -> float:
        params = {
            "hidden_dim": trial.suggest_categorical("hidden_dim", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "epochs": epochs,
            "batch_size": MODEL.lstm_params["batch_size"],
        }
        mdl = LSTMModel(**params)
        mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
        m = mdl.evaluate(sp["X_val"], sp["y_val"])
        return m["rmse"]

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


_OPTIMIZERS = {
    "linear": optimize_linear,
    "xgboost": optimize_xgb,
    "lstm": optimize_lstm,
}

_DEFAULT_TRIALS = {
    "linear": OPTIMIZATION.linear_n_trials,
    "xgboost": OPTIMIZATION.xgb_n_trials,
    "lstm": OPTIMIZATION.lstm_n_trials,
}


def optimize_model(model_name: str, sp: SplitDict, n_trials: Optional[int] = None,
                    seed: int = OPTIMIZATION.optuna_seed, **kw: Any) -> Tuple[Dict[str, Any], optuna.Study]:
    name = model_name.lower()
    fn = _OPTIMIZERS.get(name)
    if fn is None:
        raise ValueError(f"Unknown model for optimization: {model_name}. "
                        f"Options: {list(_OPTIMIZERS)}")
    if n_trials is None:
        n_trials = _DEFAULT_TRIALS[name]
    log.info(f"Optimizing {name} ({n_trials} trials)")
    return fn(sp, n_trials=n_trials, seed=seed, **kw)


def build_optimized_model(model_name: str, sp: SplitDict, n_trials: Optional[int] = None,
                           seed: int = OPTIMIZATION.optuna_seed, **kw: Any
                           ) -> Tuple[BaseModel, TrainResult, Dict[str, Any], optuna.Study]:
    best_params, study = optimize_model(model_name, sp, n_trials=n_trials, seed=seed, **kw)
    name = model_name.lower()

    mdl: BaseModel
    if name == "linear":
        mdl = LinearModel(**best_params)
    elif name == "xgboost":
        mdl = XGBModel(**best_params)
    else:
        params = dict(best_params)
        params.setdefault("epochs", MODEL.lstm_params["epochs"])
        params.setdefault("batch_size", MODEL.lstm_params["batch_size"])
        mdl = LSTMModel(**params)

    res = mdl.train(sp["X_tr"], sp["y_tr"], sp["X_val"], sp["y_val"])
    res["test"] = mdl.evaluate(sp["X_test"], sp["y_test"])
    log.info(f"{name} optimized: best_val_rmse={study.best_value:.6f}, params={best_params}")
    return mdl, res, best_params, study
