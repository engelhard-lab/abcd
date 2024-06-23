import polars as pl


def p_factor_stats():
    df = (
        pl.read_csv("data/labels/p_factors.csv")
        .pivot(values="p_factor", columns="eventname", index="src_subject_id")
        .select(pl.col(pl.Float64))
        .to_pandas()
        .corr()
        .mean()
        .mean()
    )
    print(df)


def missingness_stats():
    df = pl.read_csv("data/analytic/dataset.csv")
    print(df)
    df = df.null_count()
    print(df)


if __name__ == "__main__":
    p_factor_stats()
    missingness_stats()
