import re
import csv
import string
import numpy as np
import nltk
import time
from itertools import groupby

import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer


stop_words = set()

#TODO: Check to see if nltk has an easy way to check to see if a test is in english. (PyEnchant is an alternative)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
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
    clean_tweet = []
    for word in tweet:
        if word not in stop_words:
            clean_tweet.append(word)
    return clean_tweet


def stemmer_algorithm(tweet):
    """
    The following function is a built in algorithm in the nltk algorithm.
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


def remove_noise(regex, curr_tweet):
    """
    Functionality: This function uses multiple regular expressions to remove the noise of the raw tweet.
    regex removes: links, tags to people (i.e. @Obama), any non-alphabetical character.

    :param regex: Holds the initial regex
    :param curr_tweet: Returns the tweet [current: in string format]
    :return: Returns the current tweet (in string form)
    """
    curr_tweet = regex_tweet(regex, curr_tweet)
    curr_tweet = remove_stop_words(curr_tweet)
    curr_tweet = stemmer_algorithm(curr_tweet)
    return curr_tweet


def valid_classification(classification):
    """
    Checks to see if valid classification according to specifications.

    :param classification: Holds the current row's classification (-1, 0, 1, 2, or 'irrelevant'
    :return: returns boolean if we have a valid (for our project) class
    """
    if classification == '-1' or classification == '0' or classification == '1':
        return True
    return False


def read_tweets(file_name):
    """
    The following function reads the tweets from the file passed in, cleans the raw tweets, adds to a list

    :param file_name: Current fild name to be evaluated
    :return: Returns a list that holds the tweet list and classifications for each tweet for the specific file
    """

    tweet_list = []
    class_list = []

    with open(file_name, 'r', encoding='utf8') as csvfile:

        reader = csv.DictReader(csvfile)
        regex = re.compile(r'<.*?>|https?[^ ]+|([@])[^ ]+|[^a-zA-Z\' ]+|\d+/?')
        for row in reader:
            if valid_classification(row['classification']):
                clean_tweet = remove_noise(regex, row['Annotated tweet'])
                tweet_list.append(clean_tweet)
                class_list.append(row['classification'])

    return [tweet_list, class_list]


def tfidf_transform_tweets(counts):
    tfid_transformer = TfidfTransformer()
    X_train_counts = tfid_transformer.fit_transform(counts)
    return X_train_counts


def count_vectorize_tweets(corpus):
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
    Look this up later
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

def main():

    start = time.time()
    optimize_stop_words()

    # obama_tweets[0][x] <- holds all the tweets
    # obama_tweets[1][x] <- holds all the classifications
    obama_tweets = read_tweets('obama.csv')
    romney_tweets = read_tweets('romney.csv')

    obama_test_tweets = read_tweets('obama_test.csv')

    # Quick tester to check tweets out
    # for tweet in range(250):
    #     print(obama_tweets[0][tweet])

    #obama_vectors = vectorize_tweets(obama_tweets[0])
    obama_vectors = count_vectorize_tweets(obama_tweets[0])
    clf = MultinomialNB().fit(obama_vectors, obama_tweets[1])

    text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

    text_clf = text_clf.fit(obama_tweets[0], obama_tweets[1])
    predicted = text_clf.predict(obama_test_tweets[0])
    print(np.mean(predicted == obama_test_tweets[1]))

    text_clf_svm = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', SGDClassifier())])

    text_clf_svm = text_clf_svm.fit(obama_tweets[0], obama_tweets[1])
    predicted_svm = text_clf_svm.predict(obama_test_tweets[0])
    print('svm mean: ', np.mean(predicted_svm == obama_test_tweets[1]))

    parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3)}
    gs_clf = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(obama_tweets[0], obama_tweets[1])

    print('best: ', gs_clf.best_score_)
    print('best:param:', gs_clf.best_params_)

    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

    text_mnb_stemmed = Pipeline([('vect', TfidfVectorizer(max_features=3200, binary=True, ngram_range=(1, 1))),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf-svm', SGDClassifier(loss='hinge',
                                                           penalty='l2',
                                                           alpha=0.01,
                                                           max_iter=5,
                                                           tol=1,
                                                           random_state=42))])

    text_mnb_stemmed = text_mnb_stemmed.fit(obama_tweets[0], obama_tweets[1])
    predicted_mnb_stemmed = text_mnb_stemmed.predict(obama_test_tweets[0])
    print(np.mean(predicted_mnb_stemmed == obama_test_tweets[1]))

    precision = sklearn.metrics.precision_score(obama_test_tweets[1],
                                                predicted_mnb_stemmed,
                                                labels=['-1', '0', '1'],
                                                average='macro')

    recall = sklearn.metrics.recall_score(obama_test_tweets[1],
                                          predicted_mnb_stemmed,
                                          labels=['-1', '0', '1'],
                                          average='macro')

    fscore = sklearn.metrics.f1_score(obama_test_tweets[1],
                                      predicted_mnb_stemmed,
                                      labels=['-1', '0', '1'],
                                      average='macro')
    acc = sklearn.metrics.accuracy_score(obama_test_tweets[1], predicted_mnb_stemmed)

    info_for_classes = sklearn.metrics.precision_recall_fscore_support(obama_test_tweets[1],
                                                                       predicted_mnb_stemmed,
                                                                       labels=['-1', '0', '1'],
                                                                       average=None)
    for c in info_for_classes:
        print(c)


    print('Avg Precision: ', precision)
    print('Avg Recall: ', recall)
    print('Avg F1-Score: ', fscore)
    print('Avg Accuracy: ', acc)

    end = time.time()
    print('Total Executed Time: ', end - start)


if __name__ == '__main__':
    main()
