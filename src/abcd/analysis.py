import numpy as np
import polars as pl


def make_coefficent_table(model):
    df = model.summary().tables[1]
    numerical_columns = ["Coef.", "Std.Err.", "z", "P>|z|", "[0.025", "0.975]"]
    for col in numerical_columns:
        df[col] = df[col].replace("", np.nan)
        df[col] = df[col].astype(float)
    df = df.reset_index().rename({"index": "Predictor"}, axis=1)
    df["Predictor"] = df["Predictor"].replace(
        {"intercept": "Intercept", "pca": "PC ", "eventname": "Event"}, regex=True
    )
    df = df[df["Predictor"] != "Group Var"]
    temp = (
        pl.from_pandas(df)
        .filter(pl.col("Predictor").is_in(["Intercept", "Event"]))
        .with_columns(pl.lit(None).cast(pl.Utf8).alias("index"))
        .select(pl.col("index"), pl.exclude("index"))
    )
    df = (
        pl.from_pandas(df)
        .filter(~pl.col("Predictor").is_in(["Intercept", "Event"]))
        .with_row_index(offset=1)
        .with_columns(pl.col("index").cast(pl.Utf8))
        .drop("Predictor")
        .with_columns(("PC " + pl.col("index")).alias("Predictor"))
        .select("index", "Predictor", pl.exclude("index", "Predictor"))
    )
    df = pl.concat([temp, df]).sort(pl.col("Coef.").abs(), descending=True)
    df.write_csv("data/results/coefficients.csv")
