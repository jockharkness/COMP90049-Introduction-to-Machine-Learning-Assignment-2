from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import get_data

pd.set_option('display.max_rows', None)

df_features = get_data.get_features()
df_labels = get_data.get_labels()
df_features_validate = get_data.get_validation_features()
df_labels_validate = get_data.get_validation_labels()
df_features_test = get_data.get_test_features()

# Concatenate datasets
dataset_whole = pd.concat([df_features, df_features_validate, df_features_test])

# Fix errors
dataset_whole['title'] = dataset_whole['title'].fillna('N/A')


def remove_errors(self):
    if type(self) == str and self.isdigit() == False:
        return 0
    else:
        return self


dataset_whole['year'] = dataset_whole['year'].apply(remove_errors).apply(int)

# Lemmatize and remove stop words from titles feature
word_lem = WordNetLemmatizer()


def reduce_titles(self):
    # Process titles
    split = self.split(' ')
    split_new = ""
    for each in split:
        if each in stopwords.words('english'):
            break
        else:
            new = word_lem.lemmatize(each, pos="v")
            split_new = split_new + " " + new
    return split_new


dataset_whole['title'] = dataset_whole['title'].apply(reduce_titles)

# Get audio and visual data
visual_data = dataset_whole.iloc[:, 5:112]
audio_data = dataset_whole.iloc[:, 112:]

visual_and_audio_data = pd.concat([visual_data, audio_data], axis=1)

# Vectorize the tags and titles columns
vectorizer = CountVectorizer()
tags = vectorizer.fit_transform(dataset_whole['tag'])
tags_vectorized = pd.DataFrame(tags.toarray())

titles = vectorizer.fit_transform(dataset_whole['title'])
titles_vectorized = pd.DataFrame(titles.toarray())

years_standardized = dataset_whole['year']
years_standardized = years_standardized.values.reshape(-1, 1)
years_standardized = StandardScaler().fit_transform(years_standardized)
years_standardized = pd.DataFrame(years_standardized)


# Getter files allow the processed data to be parsed to other files

def get_titles():
    return titles_vectorized


def get_visual_and_audio_data():
    return visual_and_audio_data


def get_tags():
    return tags_vectorized


def get_years():
    return years_standardized


def get_visual_data():
    return visual_data


def get_audio_data():
    return audio_data

