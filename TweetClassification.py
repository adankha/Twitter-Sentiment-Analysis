import dis
import re
import csv
import string
from collections import Set
from random import randint

import numpy as np
import nltk
import time
from itertools import groupby

import sklearn
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor, BernoulliRBM
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
stop_words = set()

# TODO: Check to see if nltk has an easy way to check to see if a test is in english. (PyEnchant is an alternative)
# TODO: Look more into vectorizations and how they play a role in creating features.
# TODO: Create an excel spreadsheet for Obama: Positive/Negative, Romney: Positive/Negative, and a Neutral file.
# TODO: Reasoning for a neutral file: Since the data is neutral, we can use it in both training sets. Currently
# TODO: we are only looking at the Obama file and classifying pos,neg,neu, but we can extend our train set.
# TODO: Create functions to tune/optimize the parameters for different classifers/algorithms
# TODO: Include other algorithms besides (NNeighbor, etc).
# TODO: Once we reach here, then it's time to look at Deep-Learning if applicable.


class StemmedCountVectorizer(CountVectorizer):
    """
    Taken from a tutorial. TODO: Provide info on what this does.
    Essentially does the same thing as StemmerPorter
    """
    def build_analyzer(self):
        stemmer = SnowballStemmer("english")
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


def optimize_stop_words():
    """
    Since nltk.corpus.stopwords creates a list to hold their stopwords, I created a set for quicker lookup.

    :return: returns the new set of stop words.
    """
    global stop_words
    list_stop_words = nltk.corpus.stopwords.words('english')
    for word in list_stop_words:
        stop_words.add(word)
    stop_words.add('rt')
    stop_words.add('retweet')
    stop_words.add('e')



def remove_stop_words(tweet):
    """
    Iterates through the tweet and removes any stop_words.

    :param tweet: Holds the tweet in a list form
    :return: returns the new "clean_tweet"
    """
    global stop_words
    clean_tweet = []
    for word in tweet:
        if word not in stop_words:
            clean_tweet.append(word)
    return clean_tweet


def porter_stemmer(tweet):
    """
    The following function is a built in algorithm in the nltk library.
    Essentially the PorterStemmer attempts to remove suffixes and prefixes of words
    to get to the "root" word.
    Functionality: Traverse through each word and stem the word.

    :param tweet: Holds the tweet in the form of a list.
    :return: Returns the tweet after stemming
    """
    ps = nltk.stem.PorterStemmer()
    stemmed_tweet = [ps.stem(word) for word in tweet]
    stemmed_tweet = ' '.join(stemmed_tweet)
    return str(stemmed_tweet)


def regex_tweet(regex, tweet):
    """
    The following function utilizes regular expressions to handle the raw tweets
    Uses the regex.compile from func read_tweets and replaces any findings with a space.
    Removes unnecessary trailing white spaces and ignores anything that can't be parsed in utf-8.

    :param regex: Holds the initial regex.compile from read_tweets
    :param tweet: Holds the current tweet being evaluated
    :return: Returns the tweet in list form
    """
    tweet = regex.sub(' ', tweet)
    tweet = re.sub(' +', ' ', tweet).strip()
    tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    tweet = ''.join(''.join(s)[:2] for _, s in groupby(tweet))
    tweet = tweet.encode('ascii', errors='ignore').decode('utf-8').lower()
    tweet = tweet.split()
    return tweet


def pre_processing(regex, curr_tweet):
    """
    Functionality: This function uses multiple regular expressions to remove the noise of the raw tweet.
    regex removes: links, tags to people (i.e. @Obama), any non-alphabetical character. It calls a function to remove
    all stop words and to apply the porter_stemmer algorithm (to get roots of words)

    :param regex: Holds the initial regex
    :param curr_tweet: Holds the current tweet
    :return: Returns the current tweet (in string form)
    """
    curr_tweet = regex_tweet(regex, curr_tweet)
    curr_tweet = remove_stop_words(curr_tweet)
    curr_tweet = porter_stemmer(curr_tweet)
    return curr_tweet


def valid_classification(classification):
    """
    Checks to see if valid classification according to specifications.

    :param classification: Holds the current row's classification (-1, 0, 1, 2, or 'irrelevant' in our csv files)
    :return: returns boolean if we have a valid (for our project) class
    """
    if classification == '-1' or classification == '0' or classification == '1':
        return True
    return False


def read_tweets(file_name, neutral_tweets):
    """
    The following function reads the raw tweets from the file passed in, cleans the raw tweets, then adds to a list

    :param file_name: Current file name of tweets in csv format
    :return: Returns a list that holds the cleaned tweets and classifications for each tweet for the specific file
    """

    tweet_list = []
    class_list = []

    with open(file_name, 'r', encoding='utf8') as csvfile:

        reader = csv.DictReader(csvfile)
        regex = re.compile(r'<.*?>|https?[^ ]+|([@])[^ ]+|[^a-zA-Z\' ]+|\d+/?')
        for row in reader:

            classification = row['classification']
            if valid_classification(classification):

                clean_tweet = pre_processing(regex, row['Annotated tweet'])
                tweet_list.append(clean_tweet)
                class_list.append(row['classification'])

                #TODO: Temporary for our obama set. Fix this [create a neutral tweets file]
                # if 'romney.csv' == file_name:
                #     neutral_tweets.append(clean_tweet)

    return [tweet_list, class_list]


def tfidf_transform_tweets(counts):
    """
    Revisit doc
    :param counts:
    :return:
    """
    tfid_transformer = TfidfTransformer()
    X_train_counts = tfid_transformer.fit_transform(counts)
    return X_train_counts


def count_vectorize_tweets(corpus):
    """
    revisit doc
    :param corpus:
    :return:
    """
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(corpus)
    print('count:',X_train_counts.shape)
    #print('shape:',X_train_counts.data)

    # for row in X_train_counts.data:
    #     print('row: ', row)
    #     print(str(type(row)))
    print((len(X_train_counts.data)))
    X_train_counts = tfidf_transform_tweets(X_train_counts)
    return X_train_counts


def hash_vectorize_tweets(corpus):
    """
    Revisit doc
    :param corpus:
    :return:
    """
    vectorizer = HashingVectorizer(n_features=10)
    vectors = vectorizer.transform(corpus)
    return vectors


def vectorize_tweets(corpus):
    """
    Takes every tweet and essentially converts it to TF_IDF features.
    Since we are using a large dataset of tweets, there will be many words
    :param corpus:
    :return:
    """
    vectorizer = TfidfVectorizer(max_features=3200, binary=True)
    vectors = vectorizer.fit_transform(corpus)
    print(vectors)
    idf = vectorizer._tfidf.idf_

    # Prints the (key,vals) of the features generated from TfidfVectorizer()
    features = dict(zip(vectorizer.get_feature_names(), idf))
    # for word in features:
    #     print('key: ', word, ' value: ', features[word])
    print('features: ', len(features))
    return vectors


def get_average_result(actual, prediction):
    """
    Gets the average of the precision, recall, f1_score, and accuracy score of negative, neutral, and positive classes
    :param actual: List of actual test classifications
    :param prediction: Predictions of each tweet
    :return: info on predictions
    """

    labels = ['-1', '0', '1']
    avg = 'macro'

    precision = sklearn.metrics.precision_score(actual, prediction, labels=labels, average=avg)
    recall = sklearn.metrics.recall_score(actual, prediction, labels=labels, average=avg)
    f_score = sklearn.metrics.f1_score(actual, prediction, labels=labels, average=avg)
    acc = sklearn.metrics.accuracy_score(actual, prediction)

    print('Avg Precision: ', precision)
    print('Avg Recall: ', recall)
    print('Avg F1-Score: ', f_score)
    print('Avg Accuracy: ', acc)

    return [precision, recall, f_score, acc]


def get_individual_results(actual, prediction):
    """
    Similar to above function, but focuses on break down per class and category
    :param actual: List of actual test classifications
    :param prediction: Predictions of each tweet
    :return: Array that holds info per class (recall, precision, fscore, support)
    """
    labels = ['-1', '0', '1']
    avg = None
    info_for_classes = sklearn.metrics.precision_recall_fscore_support(actual, prediction, labels=labels, average=avg)

    # Prints Row: precision, recall, f_score, support | Columns: Negative, Neutral, Positive
    for c in info_for_classes:
        print(c)

    return info_for_classes


def multinomial_nb_classifer(train_data, test_data):
    clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    clf = clf.fit(train_data[0], train_data[1])
    predicted = clf.predict(test_data)
    return predicted


def svm_classifier(train_data, test_data):
    clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier())])
    clf = clf.fit(train_data[0], train_data[1])
    predicted = clf.predict(test_data)
    return predicted


def best_classifer(train_data, test_data):
    """
    This function will always hold the best classifer found so far.
    Current Best: SVC with certain parameters (see below)
    This function also has (commented out) the technique for parameter tuning using GridSearchCV

    :param train_data: train_data[0] holds all the tweets, train_data[1] holds all classifications
    :param test_data: holds all the test tweets
    :return: return the results of the predictions
    """

    # TODO: Look up StemmedCountVectorizer. How does this stemmer give different results than porterstemmer????!!!
    stemmed_count_vect = StemmedCountVectorizer()
    other_vector = TfidfVectorizer()

    # Copy paste each of these in the 'clf' section. Current best: SVC with a Fscore of 66%
    # SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, random_state=4193)
    # LogisticRegression(penalty='l2',class_weight='balanced', random_state=41)
    # Perceptron(alpha=0.001, penalty=None, class_weight='balanced', random_state=42)
    # SVC(kernel="rbf", gamma=1, C=1, degree=2, class_weight='balanced', random_state=42)

    clf_stemmed = Pipeline([('vect', stemmed_count_vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf',  SVC(kernel="rbf", gamma=1, C=1, degree=2, class_weight='balanced', random_state=42))])

    clf_stemmed = clf_stemmed.fit(train_data[0], train_data[1])
    predicted = clf_stemmed.predict(test_data)

    # Parameters used to "tune" a classifier. Visit documentation online to learn about what the parameters are.
    # parameters = {
    #                 'clf__penalty': ('l2', None)
    # }
    #
    #
    # gs_clf = GridSearchCV(text_mnb_stemmed, parameters, n_jobs=-1, cv=10)
    # gs_clf = gs_clf.fit(obama_tweets[0], obama_tweets[1])
    #
    #
    # print('best score: ', gs_clf.best_score_)
    # print('best param:', gs_clf.best_params_)

    return predicted


def multiple_classifier(train_data, test_data):
    """
    The following function uses multiple classifiers and prints their results.

    :param train_data: train_data[0] holds all the tweets, train_data[1] holds all classifications
    :param test_data: test_data[0] holds all the test tweets, test_data[1] holds all classifications of test set
    :return: No return value
    """

    classifiers = {
        'NearestCentroid': NearestCentroid(),
        'Logistic Regression': LogisticRegression(penalty='l2',class_weight='balanced', random_state=41),
        'SVC1': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                    decision_function_shape='ovr', degree=3, gamma=3, kernel='rbf',
                    max_iter=-1, probability=False, random_state=None, shrinking=True,
                    tol=0.001, verbose=False),
        'SVC2': SVC(kernel="linear", gamma=3, C=1, class_weight='balanced'),
        'KNN': KNeighborsClassifier(n_neighbors=150, algorithm='auto', p=2),
        'LinearSVC': LinearSVC(C=.5, class_weight='balanced'),
        'DecisionTree': DecisionTreeClassifier(max_depth=5, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, class_weight='balanced'),
        'MLP': MLPClassifier(alpha=1),
        'AdaBoost': AdaBoostClassifier()
    }

    # TODO: Look up StemmedCountVectorizer. How does this stemmer give different results to porterstemmer????!!!
    vect = StemmedCountVectorizer()
    print('\nPrinting Results of Multiple Classifiers\n')
    for classifier in classifiers.keys():
        start = time.time()
        text_stemmed = Pipeline([('vect', vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', classifiers[classifier])])

        text_stemmed = text_stemmed.fit(train_data[0], train_data[1])
        predicted_mnb_stemmed = text_stemmed.predict(test_data[0])

        print('Classifier: ', classifier)
        get_individual_results(test_data[1], predicted_mnb_stemmed)
        get_average_result(test_data[1], predicted_mnb_stemmed)
        end = time.time()

        print('Total Time Elapsed: ', (end - start) * 1000, '\n\n')


def main():

    start = time.time()
    optimize_stop_words()
    neutral_tweets = []

    # obama_tweets[0][x] <- holds all the tweets
    # obama_tweets[1][x] <- holds all the classifications
    obama_tweets = read_tweets('obama.csv', neutral_tweets)
    romney_tweets = read_tweets('romney.csv', neutral_tweets)
    obama_test_tweets = read_tweets('obama_test.csv', neutral_tweets)

    # # MultinomialNB Predictions
    # print('MultinomialNB Predictions:')
    # predicted_multi_nb = multinomial_nb_classifer(obama_tweets, obama_test_tweets[0])
    # get_individual_results(obama_test_tweets[1], predicted_multi_nb)
    # get_average_result(obama_test_tweets[1], predicted_multi_nb)
    #
    #
    # # SVM Predictions
    # print('SVM Predictions:')
    # predicted_svm = svm_classifier(obama_tweets, obama_test_tweets[0])
    # get_individual_results(obama_test_tweets[1], predicted_svm)
    # get_average_result(obama_test_tweets[1], predicted_svm)

    # Multiple classifier predictions
    print('Printing best classifier [for now]')
    predicted = best_classifer(obama_tweets, obama_test_tweets[0])
    get_individual_results(obama_test_tweets[1], predicted)
    get_average_result(obama_test_tweets[1], predicted)

    print('Printing multiple Predictions:')
    multiple_classifier(obama_tweets, obama_test_tweets)

    end = time.time()
    print('Total Executed Time: ', end - start)


if __name__ == '__main__':
    main()
