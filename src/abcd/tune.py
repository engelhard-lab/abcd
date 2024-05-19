import pickle
from optuna import Trial, create_study
from optuna.samplers import TPESampler

from abcd.config import Config
from abcd.model import make_model, make_trainer
from abcd.utils import cleanup_checkpoints


def tune(config: Config, data_module, input_dim: int, output_dim: int):
    def objective(trial: Trial):
        params = {
            "hidden_dim": trial.suggest_categorical(
                name="hidden_dim", choices=config.model.hidden_dim
            ),
            "num_layers": trial.suggest_int(
                name="num_layers",
                low=config.model.num_layers["low"],
                high=config.model.num_layers["high"],
            ),
            "dropout": trial.suggest_float(
                name="dropout",
                low=config.model.dropout["low"],
                high=config.model.dropout["high"],
            ),
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
            "method": trial.suggest_categorical("method", ["rnn", "mlp"]),
        }
        model = make_model(
            # method=config.method,
            input_dim=input_dim,
            output_dim=output_dim,
            momentum=config.optimizer.momentum,
            nesterov=config.optimizer.nesterov,
            **params,
        )
        trainer, checkpoint_callback = make_trainer(config)
        trainer.fit(model, datamodule=data_module)
        cleanup_checkpoints(config.filepaths.checkpoints, mode="min")
        return checkpoint_callback.best_model_score.item()  # type: ignore

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
    study.optimize(func=objective, n_trials=config.n_trials)
    with open("data/studies/study.pkl", "wb") as f:
        pickle.dump(study, f)
    cleanup_checkpoints(config.filepaths.checkpoints, mode="min")
    return study
