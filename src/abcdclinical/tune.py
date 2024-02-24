from sklearn.metrics import mean_squared_error
from abcdclinical.dataset import drop_demographics, get_data
from optuna import Trial, create_study
from optuna.samplers import TPESampler
import statsmodels.api as sm

from abcdclinical.config import Config


def tune(config: Config):

    def objective(trial: Trial):
        (
            X_train,
            y_train,
            group_train,
            X_val,
            y_val,
            group_val,
            X_test,
            y_test,
            group_test,
        ) = get_data(
            config,
            regenerate=True,
            n_components=trial.suggest_int("n_components", low=1, high=122),
        )
        X_train, X_val, X_test = drop_demographics(X_train, X_val, X_test)
        model = sm.MixedLM(y_train, X_train, groups=group_train).fit(method="lbfgs")
        y_pred = model.predict(X_val)
        return float(mean_squared_error(y_val, y_pred))

    sampler = TPESampler(seed=config.random_seed)
    study = create_study(
        sampler=sampler,
        direction="minimize",
        study_name="ABCD",
    )
    study.optimize(func=objective, n_trials=config.n_trials)
    return study
