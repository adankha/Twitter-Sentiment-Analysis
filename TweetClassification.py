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
    global stop_words
    list_stop_words = nltk.corpus.stopwords.words('english')
    for word in list_stop_words:
        stop_words.add(word)
    stop_words.add('rt')
    stop_words.add('retweet')
    stop_words.add('e')


def remove_stop_words(tweet):
    clean_tweet = []
    for word in tweet:
        if word not in stop_words:
            clean_tweet.append(word)
    return clean_tweet


def remove_noise(regex, curr_tweet):
    curr_tweet = regex.sub(' ', curr_tweet)
    curr_tweet = re.sub(' +', ' ', curr_tweet).strip()
    curr_tweet = re.sub('[%s]' % re.escape(string.punctuation), '', curr_tweet)
    curr_tweet = ''.join(''.join(s)[:2] for _, s in groupby(curr_tweet))
    curr_tweet = curr_tweet.encode('ascii', errors='ignore').decode('utf-8').lower()
    curr_tweet = curr_tweet.split(' ')
    curr_tweet = remove_stop_words(curr_tweet)
    return curr_tweet


def read_tweets(file_name):

    tweet_list = []

    with open(file_name, 'r', encoding='utf8') as csvfile:
        reader = csv.DictReader(csvfile)
        regex = re.compile(r'<.*?>|https?[^ ]+|([@])[^ ]+|[^a-zA-Z\' ]+|\d+/?')

        for row in reader:
            if row['classification'] == '-1' or row['classification'] == '0' or row['classification'] == '1':
                clean_tweet = remove_noise(regex, row['Annotated tweet'])
                tweet_list.append(clean_tweet)

    return tweet_list


def main():
    start = time.time()
    optimize_stop_words()
    obama_tweets = read_tweets('obama.csv')
    romney_tweets = read_tweets('romney.csv')
    for x in range(200):
        print(obama_tweets[x])


if __name__ == '__main__':
    main()