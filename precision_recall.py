from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import feature_extraction, get_data
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
# setup plot details
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

X = feature_extraction.get_tags()[:5240]
y = get_data.get_labels()

classifier = MLPClassifier()
classifier.fit(X, y)
y_score = classifier.predict(X)


# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(get_data.get_validation_labels(), y_score)
#
# print('Average precision-recall score: {0:0.2f}'.format(
#       average_precision))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

disp = plot_precision_recall_curve(classifier, get_data.get_validation_features(), get_data.get_validation_labels)
# disp.ax_.set_title('2-class Precision-Recall curve: '
#                    'AP={0:0.2f}'.format(average_precision))


plt.show()