import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print('Target Names:')
print(twenty_train.target_names)

print('Length of train data and file names:')
len(twenty_train.data)
len(twenty_train.filenames)

print('Print first lines of the first loaded file:')
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])

"""
Supervised learning algorithms will require a category label for each document in the training set. 
In this case the category is the name of the newsgroup which also happens to be the name of the 
folder holding the individual documents.
"""

print('Integers correspond to category for space efficiency:')
print(twenty_train.target[:10])

print('Getting category names from ints:')
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


"""
Extracting features from text files using the concept of "Bag of Words"
In order to perform machine learning on text documents, 
we first need to turn the text content into numerical feature vectors.
The most intuitive way to do so is the bags of words representation:

    1. Assign a fixed integer id to each word occurring in any document of the training set.
    (for instance by building a dictionary from words to integer indices).
    
    2. For each document #i, count the number of occurrences of each word w 
    and store it in X[i, j] as the value of feature #j where j is the index of word w in the dictionary
"""

"""
Tokenizing text with scikit-learn

Text preprocessing, tokenizing and filtering of stopwords are included in a high level component 
that is able to build a dictionary of features and transform documents to feature vectors:

"""

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print('(n_samples, n_features)')
print(X_train_counts.shape)

"""
CountVectorizer supports counts of N-grams of words or consecutive characters. 
Once fitted, the vectorizer has built a dictionary of feature indices:
"""
print('Index of Algorithm: ', count_vect.vocabulary_.get(u'algorithm'))


"""
Fit: Fit our estimator to the data.
Transform: transform our count-matrix to a tf-idf representation
"""

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('(n_samples, n_features)')
print(X_train_tfidf.shape)


"""
Now that we have our features, we can train a classifier to try to predict the category of a post. 
Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task. 
scikit-learn includes several variants of this classifier; 
the one most suitable for word counts is the multinomial variant:
"""
print('Building a classifier:')
print('X_train_tfidf:')
print(X_train_tfidf)

print('twenty_train.data')
print(str(twenty_train.data))

print('twenty_train.target:')
print(twenty_train.target)


"""
Now that we have our features, we can train a classifier to try to predict the category of a post. 
Let’s start with a naïve Bayes classifier, which provides a nice baseline for this task. 
scikit-learn includes several variants of this classifier; 
the one most suitable for word counts is the multinomial variant:
"""

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)


docs_new = ['God is love', 'OpenGL on the GPU is fast']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


"""
In order to make the vectorizer => transformer => classifier easier to work with, 
scikit-learn provides a Pipeline class that behaves like a compound classifier:
"""
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

text_clf.fit(twenty_train.data, twenty_train.target)

"""
Evaluation of the performance on the test set:
Evaluating the predictive accuracy of the model is equally easy.
"""

twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
print('Mean of prediction: ', np.mean(predicted == twenty_test.target))
print('target names: ', twenty_test.target_names)
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))


