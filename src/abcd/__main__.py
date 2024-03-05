from multiprocessing import cpu_count
from tomllib import load
import pickle
from lightning import seed_everything
from sklearn import set_config

from abcd.dataset import ABCDDataModule, RNNDataset
from abcd.evaluate import evaluate
from abcd.preprocess import get_data
from abcd.config import Config
from abcd.model import Network, make_trainer
from abcd.plots import plot
from abcd.tune import tune
from abcd.utils import cleanup_checkpoints, get_best_checkpoint


def main():
    set_config(transform_output="pandas")
    with open("config.toml", "rb") as f:
        config = Config(**load(f))
    seed_everything(config.random_seed)
    train, val, test = get_data(config, regenerate=config.regenerate)
    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=config.training.batch_size,
        num_workers=cpu_count(),
        dataset_class=RNNDataset,
        target="p_score",
    )
    input_dim = train.shape[1] - 2
    if config.tune:
        study = tune(
            config=config,
            data_module=data_module,
            input_dim=input_dim,
        )
        best_model_path = get_best_checkpoint(
            ckpt_folder=config.filepaths.checkpoints, mode="min"
        )
        model = Network.load_from_checkpoint(best_model_path)
    else:
        with open("data/studies/study.pkl", "rb") as f:
            study = pickle.load(f)
        if config.refit:
            model = Network(
                input_dim=input_dim,
                output_dim=1,
                momentum=config.optimizer.momentum,
                nesterov=config.optimizer.nesterov,
                **study.best_params,
            )
            trainer, _ = make_trainer(config)
            trainer.fit(model, datamodule=data_module)
            cleanup_checkpoints(config.filepaths.checkpoints, mode="min")
        else:
            best_model_path = get_best_checkpoint(
                ckpt_folder=config.filepaths.checkpoints, mode="min"
            )
            model = Network.load_from_checkpoint(best_model_path)
    print(study.best_params)

    data_module = ABCDDataModule(
        train=train,
        val=val,
        test=test,
        batch_size=500,
        num_workers=0,
        dataset_class=RNNDataset,
        target="p_score",
    )
    if config.evaluate:
        # test_dataloader = iter(data_module.test_dataloader())
        # X, _ = next(test_dataloader)
        # background, _ = next(test_dataloader)
        # explainer = GradientExplainer(model, background.to("mps:0"))
        # shap_values = explainer.shap_values(X.to("mps:0"))
        # save(shap_values, "data/results/shap_values.pt")
        evaluate(config=config, model=model, data_module=data_module)
    if config.plot:
        plot(data_module)
    # make_tables()


if __name__ == "__main__":
    main()
