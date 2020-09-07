import feature_extraction, get_data
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# Create index
visual = 0
audio = 1
titles = 2
tags = 3


# Get data
df_features = [feature_extraction.get_visual_data()[:5240],
               feature_extraction.get_audio_data()[:5240],
               feature_extraction.get_titles()[:5240],
               feature_extraction.get_tags()[:5240]]
df_labels = get_data.get_labels()
df_features_validate = [feature_extraction.get_visual_data()[5240:5539],
                        feature_extraction.get_audio_data()[5240:5539],
                        feature_extraction.get_titles()[5240:5539],
                        feature_extraction.get_tags()[5240:5539]]
df_labels_validate = get_data.get_validation_labels()


# Define alpha values (cost complexity pruning granularity) for testing
ccp_alphas = [x / 1000.0 for x in range(0, 10)]


# Build a classifier for each each of the alpha values
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(df_features[titles], df_labels)
    clfs.append(clf)

# Trim root node
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

# Define an array containing the scores for training
train_scores = [clf.score(df_features[titles], df_labels) for clf in clfs]
validate_scores = [clf.score(df_features_validate[titles], df_labels_validate) for clf in clfs]


# Plot the scores to show effect of pruning on accuracy

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Titles")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, validate_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()
