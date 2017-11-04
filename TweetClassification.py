import re
import csv
import string

import goslate
import nltk
import json
import time
import inflect
from itertools import groupby

stop_words = set()

#TODO: Check to see if nltk has an easy way to check to see if a test is in english. (PyEnchant is an alternative)


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
    stemmed_tweet = " ".join(stemmed_tweet)
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


def main():

    start = time.time()
    optimize_stop_words()

    # obama_tweets[0][x] <- holds all the tweets
    # obama_tweets[1][x] <- holds all the classifications
    obama_tweets = read_tweets('obama.csv')
    romney_tweets = read_tweets('romney.csv')

    # Quick tester to check tweets out
    for tweet in range(250):
        print(obama_tweets[0][tweet])

    end = time.time()
    print('Total Executed Time: ', end - start)


if __name__ == '__main__':
    main()
