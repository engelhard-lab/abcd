from multiprocessing import cpu_count
from tomllib import load
import pickle
from lightning import seed_everything
from optuna import Study
from sklearn import set_config

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate_model
from abcd.preprocess import get_data
from abcd.config import Config
from abcd.model import make_model
from abcd.plots import plot
from abcd.tables import make_tables
from abcd.tune import tune


def main():
    set_config(transform_output="pandas")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    match config.target:
        case "binary":
            targets = ["p_factor"]
        case "multioutput":
            targets = config.labels.cbcl_labels
        case _:
            raise ValueError(
                f"Invalid target '{config.target}'. Choose from: 'p_factor' or 'multioutput'"
            )
    train, val, test = get_data(config, regenerate=config.regenerate)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
        targets=targets,
    )
    n_targets = len(targets)
    match config.task:
        case "classification":
            output_dim = config.n_quantiles
        case "regression":
            output_dim = n_targets
    input_dim = train.shape[1] - n_targets - 1  # -1 for src_subject_id
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        with open("data/studies/study.pkl", "rb") as f:
            study: Study = pickle.load(f)
    model = make_model(
        method=config.method,
        task=config.task,
        target=config.target,
        input_dim=input_dim,
        output_dim=output_dim,
        **study.best_params,
    )
    print(study.best_params)
    if config.evaluate:
        evaluate_model(config=config, model=model, data_module=data_module)
    if config.plot:
        plot(config=config, dataloader=data_module.test_dataloader())
    if config.tables:
        make_tables()


if __name__ == "__main__":
    main()
