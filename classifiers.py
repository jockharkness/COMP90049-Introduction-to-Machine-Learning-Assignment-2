from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

import feature_extraction
import get_data

# Define array with the four classifier names
names = ["Naive Bayes", "Perceptron",
         "Decision Tree", "Neural Net"]

# Define an array with the four classifiers
classifiers = [GaussianNB(),
               Perceptron(),
               DecisionTreeClassifier(),
               MLPClassifier()]


# Import the data
df_features = feature_extraction.get_visual_data()[:5240]
df_labels = get_data.get_labels()
df_features_validate = feature_extraction.get_visual_data()[5240:5539]
df_labels_validate = get_data.get_validation_labels()

# Iterate through the classifiers, print their score
for name, clf in zip(names, classifiers):
    clf.fit(df_features, df_labels)
    score = clf.score(df_features_validate, df_labels_validate)
    print(score)
