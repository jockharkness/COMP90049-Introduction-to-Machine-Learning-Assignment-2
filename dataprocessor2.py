import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Import datasets
df_features = pd.read_csv("venv/data/train_features.tsv", sep="\t")
df_labels = pd.read_csv("venv/data/train_labels.tsv", sep="\t")
df_features_validate = pd.read_csv("venv/data/valid_features.tsv", sep="\t")
df_labels_validate = pd.read_csv("venv/data/valid_labels.tsv", sep="\t")
df_features_test = pd.read_csv("venv/data/test_features.tsv", sep="\t")

# # Handle tags column
tags_original = df_features.iloc[:, 4]
tags_new = pd.DataFrame(tags_original)


tag_counts = tags_new['tag'].str.get_dummies(sep=',').sum()
pd.set_option('display.max_rows', None)
tag_counts = tag_counts.sort_values(axis='index', ascending=False)

# Remove tags which don't say anything about movie genre
remove_tags = ['bd-r', 'clv', 'betamax', 'dvd-video', 'erlend\'s_dvds', 'tumey\'s_dvds', 'bd-video', 'r',
               'dvd', 'seen_more_than_once', 'reviewed', 'imdb_top_250', 'national_film_registry', 'predictable',
               'dvd-ram', 'can\'t_remember',
               'adapted_from:book', 'based_on_a_book', 'boring', '70mm', 'less_than_300_ratings', 'franchise',
               'true_story', 'cinematography', 'soundtrack', 'overrated', 'remake', 'great_soundtrack', 'new_york_city',
               'bechdel_test:fail']

for each in remove_tags:
    tag_counts = tag_counts.drop(labels=each)

def reduce_tags(self):
    # Reduce to one tag
    split = self.split(',')
    for each in split:
        if each in tag_counts.index:
            return each
        elif each not in tag_counts.index:
            return 'N/A'


# Remove tags which do not specify genre and have high frequency
matches = [',', 'bd-r', 'clv', 'betamax', 'dvd-video', 'erlend\'s_dvds', 'tumey\'s_dvds', 'bd-video']


def remove_irrelevant_tags(self):
    if self in matches:
        return 'N/A'
    elif self is None:
        return 'N/A'
    else:
        return self


# Change tags to numerical values
def convert_tag_to_numerical(self):
    for i in range(len(tag_counts.index)):
        if self == tag_counts.index[i]:
            return i
        elif self == 'N/A':
            return 170



labels = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
          'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']


# Change genre to numerical value
def convert_genre_to_numerical(self):
    for i in range(len(labels)):
        if self == labels[i]:
            return i
        else:
            continue


# Remove errors in year column
def remove_errors(self):
    if self.isdigit() == False:
        # print(type(self))
        return 0
    else:
        return self


def get_features():
    features_final = df_features.iloc[:, 3:]
    features_final['year'] = features_final['year'].apply(remove_errors).apply(int)
    features_final['tag'] = features_final['tag'].apply(reduce_tags, remove_irrelevant_tags)
    features_final['tag'] = features_final['tag'].apply(convert_tag_to_numerical)
    return features_final
get_features()

def get_labels():
    labels_final = df_labels.iloc[:, 1]
    labels_final = labels_final.apply(convert_genre_to_numerical)
    return labels_final


def get_validation_features():
    validation_final = df_features_validate.iloc[:, 3:]
    validation_final['tag'] = validation_final['tag'].apply(reduce_tags, remove_irrelevant_tags)
    validation_final['tag'] = validation_final['tag'].apply(convert_tag_to_numerical)
    return validation_final


def get_validation_labels():
    df_labels_validate_final = df_labels_validate.iloc[:, 1]
    df_labels_validate_final = df_labels_validate_final.apply(convert_genre_to_numerical)
    return df_labels_validate_final


def get_test_features():
    final_test = df_features_test.iloc[:, 3:]
    final_test['year'] = final_test['year'].apply(remove_errors).apply(int)
    final_test['tag'] = final_test['tag'].apply(reduce_tags, remove_irrelevant_tags)
    final_test['tag'] = final_test['tag'].apply(convert_tag_to_numerical)
    return final_test


