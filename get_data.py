import pandas as pd

# Import  data
df_features = pd.read_csv("venv/data/train_features.tsv", sep="\t")
df_labels = pd.read_csv("venv/data/train_labels.tsv", sep="\t")
df_features_validate = pd.read_csv("venv/data/valid_features.tsv", sep="\t")
df_labels_validate = pd.read_csv("venv/data/valid_labels.tsv", sep="\t")
df_features_test = pd.read_csv("venv/data/NEW_test_features.tsv", sep="\t")

# Getter functions allow data to be parsed to other files
def get_features():
    return df_features


def get_labels():
    labels = df_labels.iloc[:, 1]
    return labels


def get_validation_features():
    return df_features_validate


def get_validation_labels():
    labels_validate = df_labels_validate.iloc[:, 1]
    return labels_validate


def get_test_features():
    return df_features_test
