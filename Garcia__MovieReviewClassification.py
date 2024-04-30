import pickle
import re
import os
from vectorizer import vect
clf = pickle.load(open(os.path.join('/Users/josegarcia/Desktop/SentimentClassification/movieclassifier/pkl_objects', '/Users/josegarcia/Desktop/SentimentClassification/movieclassifier/pkl_objects/classifier.pkl'), 'rb'))

import numpy as np
label = {0:'negative', 1:'positive'}
example = ["I love this movie. It's amazing."]
X = vect.transform(example)
print('Prediction: %s\nProbability: %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_proba(X))*100))


