from datasets import load_dataset
import pandas as pd

REPO_ID = "toqa66/hr-talent-mining-dataset"


def load_raw_dataset():
    dataset = load_dataset(
        REPO_ID,
        data_files="versions/hr_dataset_v0_raw.csv"
    )

    df = dataset["train"].to_pandas()

    return df