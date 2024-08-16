import torch
from sentence_transformers import SentenceTransformer
import polars as pl
import polars.selectors as cs


def make_embeddings(sentences: list[str]) -> None:
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = torch.tensor(model.encode(sentences))  # type: ignore
    padding_embedding = torch.zeros(1, embeddings.shape[-1])
    embeddings = torch.concat([padding_embedding, embeddings])
    torch.save(embeddings, "data/embeddings.pt")


def make_nlp_table(sentences: list[str]) -> pl.DataFrame:
    return (
        pl.read_csv("data/variables.csv")
        .drop_nulls(subset=["response"])
        .with_columns(
            pl.col("response").str.replace_all(".", ";", literal=True).str.split(";")
        )
        .explode("response")
        .filter(pl.col("response").str.contains("="))
        .with_columns(
            pl.col("response").str.extract(r"(\d+)\s*=").cast(pl.Int32).alias("value")
        )
        .with_columns(pl.col("response").str.extract(r"=\s*(.*)"))
        .with_columns(
            (
                pl.col("respondent")
                + "; "
                + pl.col("question")
                + ": "
                + pl.col("response")
                + "."
            ).alias("nlp_input")
        )
        .with_row_index(offset=1)
        .with_columns(pl.Series(sentences).alias("nlp_output"))
        .filter(pl.col("variable").ne("race_ethnicity"))
    )


def pad_lists(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("index")
            .len()
            .over(["src_subject_id", "eventname"])
            .max()
            .alias("max_length")
        )
        .group_by("src_subject_id", "eventname")
        .agg(
            pl.col("index").extend_constant(
                value=0, n=pl.col("max_length").first() - pl.col("index").len()
            ),
            pl.col("y_{t+1}").first(),
        )
        .drop("max_length")
    )


# .select(cs.exclude(["src_subject_id", "eventname", "y_{t+1}"]))


def format_split(split: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    split = split.with_columns(cs.numeric().cast(pl.Int32)).melt(
        id_vars=["src_subject_id", "eventname", "y_t", "y_{t+1}"]
    )
    df = df.select(["variable", "value", "index"])
    split = (
        split.join(
            df,
            on=["variable", "value"],
            how="inner",
        )
        .pipe(pad_lists)
        .sort(["src_subject_id", "eventname"])
    )
    return split


def make_data():
    df = pl.read_csv("data/responses.csv")
    sentences = df["nlp_output"].to_list()
    make_embeddings(sentences=sentences)
    df = make_nlp_table(sentences=sentences)
    splits = {
        "train": pl.DataFrame(),
        "val": pl.DataFrame(),
        "test": pl.DataFrame(),
    }
    for name in splits.keys():
        split = pl.read_csv("data/analyses/metadata/analytic/" + name + ".csv")
        splits[name] = format_split(split, df)
    return splits
