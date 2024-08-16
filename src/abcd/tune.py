from functools import partial
import pickle
from typing import Callable
from optuna import Trial, create_study
from optuna.samplers import TPESampler

from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.utils import cleanup_checkpoints


def get_rnn_params(trial: Trial, config: Config) -> dict:
    return {
        "num_layers": trial.suggest_int(
            name="num_layers",
            low=config.model.num_layers["low"],
            high=config.model.num_layers["high"],
        ),
        "hidden_dim": trial.suggest_categorical(
            name="hidden_dim", choices=config.model.hidden_dim
        ),
    }


def get_embedding_params(trial: Trial, config: Config) -> dict:
    return {
        "refine": True,  # FIXME trial.suggest_categorical(name="refine", choices=[True, False]),
        "freeze": False,  # trial.suggest_categorical(name="freeze", choices=[True, False]),
    }


def get_shared_params(trial: Trial, config: Config) -> dict:
    return {
        "lr": trial.suggest_float(
            name="lr",
            low=config.optimizer.lr["low"],
            high=config.optimizer.lr["high"],
        ),
        "weight_decay": trial.suggest_float(
            name="weight_decay",
            low=config.optimizer.weight_decay["low"],
            high=config.optimizer.weight_decay["high"],
        ),
        "dropout": trial.suggest_float(
            name="dropout",
            low=config.model.dropout["low"],
            high=config.model.dropout["high"],
        ),
        # "method": trial.suggest_categorical(name="method", choices=["rnn", "mlp"]),
    }


def get_method_params(method: str) -> Callable:
    match method:
        case "rnn" | "mlp":
            return get_rnn_params
        case "embedding":
            return get_embedding_params
        case _:
            raise ValueError(
                f"Invalid method '{method}'. Choose from: 'rnn', 'mlp', or 'embedding'"
            )


# FIXME: current "rnn" and "mlp" methods are never reached
def objective(
    trial: Trial, config: Config, data_module, input_dim: int, output_dim: int
):
    method = "embedding"  # FIXME: method is hardcoded
    method_params = get_method_params(method=method)
    params = method_params(trial=trial, config=config)
    shared_params = get_shared_params(trial=trial, config=config)
    params.update(shared_params)
    model = make_model(
        method=method,
        input_dim=input_dim,
        output_dim=output_dim,
        momentum=config.optimizer.momentum,
        nesterov=config.optimizer.nesterov,
        **params,
    )
    trainer, checkpoint_callback = make_trainer(config=config)
    trainer.fit(model, datamodule=data_module)
    cleanup_checkpoints(config.filepaths.data.results.checkpoints, mode="min")
    return checkpoint_callback.best_model_score.item()  # type: ignore


def tune(config: Config, data_module, input_dim: int, output_dim: int):
    sampler = TPESampler(
        seed=config.random_seed,
        multivariate=True,
        n_startup_trials=config.n_trials // 2,
    )
    study = create_study(
        sampler=sampler,
        direction="minimize",
        study_name="ABCD",
    )
    objective_function = partial(
        objective,
        config=config,
        data_module=data_module,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    study.optimize(func=objective_function, n_trials=config.n_trials)
    with open(config.filepaths.data.results.study, "wb") as f:
        pickle.dump(study, f)
    return study
