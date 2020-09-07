
import pandas as pd
from sklearn.dummy import DummyClassifier


df_features = pd.read_csv("venv/data/train_features.tsv", sep="\t")
df_labels = pd.read_csv("venv/data/train_labels.tsv", sep="\t")


dummy_classifier = DummyClassifier(strategy="most_frequent")
dummy_classifier.fit(df_features, df_labels["genres"])
x = dummy_classifier.predict(df_features)
for each in x:
    print(each)
