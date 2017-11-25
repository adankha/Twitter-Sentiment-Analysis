"""
NOTE: The following code was my first attempt on this research project.

Please visit the TwitterSentimentAnalysis.py file for the actual program.
"""

import re
import csv
import goslate
import nltk
import json
import time
import inflect
from itertools import groupby

p = inflect.engine()

# TODO: Find a better way to figure out if a Tweet is in English or not [CURRENT: Using goslate, but too many calls to
# TODO: Google api]
# TODO: Create FREQ count for tweets  [DONE using map]
# TODO: Get rid of / use "stop" words [DONE using nltk]
# TODO: Try to "combine" words i.e. weirdoooooooooo -> weirdo [Potentially create a regex to remove multiple letters?]


# TODO: Also, if you have apples or apple, create just one word -> apple [No clue. Try to find a library?]
# TODO: ABOVE IS POTENTIALLY DONE. Used Library inflect to have singular_nouns translated if possible.
# TODO: i.e. romneys --> romney, turns --> turn

# TODO: Is Levenshtein (sp?) useful here to find commonalities between words?

"""
NAIVE APPROACH:
    Step 1: Create a list for Romney and Obama where the map holds every Tweet and their classification.
        Note: Map will be word -> list[frequency_classification] 
        i.e. "monkey" -> [freq_positive, freq_negative, freq_neutral, freq_mixed]
        
    Step 2: Parse each tweet in the following way.
        For each word in the tweet
        Check to see if word exists in map
        If it does, +1 to the corresponding classification
        
    After step 2, we will have a map that has every single word shown in every single tweet as the key.
    The value, as described above, will have a frequency of how many times that word was used per classification
    
    Why is that useful?
    
    My approach [for now] will be to train my machine such that 
    
    # of tweets * number of words
    15k * 14.2k = 213 mil...
    
    # TODO: Look more into creating Features for our Dataset? Each word fits in some feature, then use the feature
    # TODO: for train/test data. This will significantly reduce column count 
        
    # Time taken to execute so far: 38 seconds
    # Bottlenecks: Stopwords. FIXED: Created a map 38k ms --> 400 ms, GG.
    
    # Tradeoff: Translated plurals if singular_nouns to singular
    # Increase time to 3 seconds, but reduce words down to 2.6k
"""


def add_to_dict(stop_words, row, word, words):

    curr_class = row['classification']

    if curr_class == '-1' or curr_class == '0' or curr_class == '1' and word not in stop_words:
        singular_word = p.singular_noun(word)

        if singular_word is not False:
            word = singular_word

        if word not in words:
            if row['classification'] == '-1':
                words[word] = [1, 0, 0]
            elif row['classification'] == '0':
                words[word] = [0, 1, 0]
            elif row['classification'] == '1':
                words[word] = [0, 0, 1]
            elif row['classification'] == '2':
                words[word] = [0, 0, 0]
        else:
            if row['classification'] == '-1':
                words[word][0] += 1
            elif row['classification'] == '0':
                words[word][1] += 1
            elif row['classification'] == '1':
                words[word][2] += 1
            elif row['classification'] == '2':
                words[word][3] += 1


def remove_noise(regex, line):
    line = regex.sub(' ', line)
    line = re.sub(' +', ' ', line).strip()
    line = ''.join(''.join(s)[:2] for _, s in groupby(line))
    line = line.encode('ascii', errors='ignore').decode('utf-8').lower()
    line = line.split(' ')
    return line


def train_read_lines(reader, words, stop_words):
    # Current Regex Removes:
    # Links
    # Tags to other people (@Obama)
    # Hashtag words
    # Anything not an alphabet
    # All digits
    regex = re.compile(r'<.*?>|https?[^ ]+|([@#])[^ ]+|[^a-zA-Z ]+|\d+/?')
    count = 0
    for row in reader:
        if count >= 6420:
            break

        tweet = row['Annotated tweet']
        tweet = remove_noise(regex, tweet)

        for word in tweet:
            add_to_dict(stop_words, row, word, words)
        count += 1


def get_word_classification_idx(cur_word, words):
    if cur_word in words:
        return words[cur_word].index(max(words[cur_word]))
    else:
        return -999


def test_read_line(tweets, obama_words, romney_words, stop_words):
    regex = re.compile(r'<.*?>|https?[^ ]+|([@#])[^ ]+|[^a-zA-Z ]+|\d+/?')

    classifications = []

    for tweet in tweets:
        actual_class = tweet[1]
        tweet_file = tweet[2]
        tweet = remove_noise(regex, tweet[0])
        class_freq = [0, 0, 0]

        dicts = [obama_words, romney_words]

        if tweet_file == 'obama_file':
            dict_to_use = dicts[0]
        else:
            dict_to_use = dicts[1]

        for word in tweet:

            if word not in stop_words:
                word_idx = get_word_classification_idx(word, dict_to_use)
                if word_idx != -999:
                    class_freq[word_idx] += 1
                else:
                    if dict_to_use == dicts[0]:
                        dict_to_use = dicts[1]
                    else:
                        dict_to_use = dicts[0]
                    word_idx = get_word_classification_idx(word, dict_to_use)
                    if word_idx != -999:
                        if word_idx == 0:
                            class_freq[1] += 1
                        else:
                            class_freq[0] += 1

        # idx_of_max holds the index of the maximum value
        idx_of_max = class_freq.index(max(class_freq))
        result_class = -999
        if idx_of_max == 0:
            result_class = -1
        elif idx_of_max == 1:
            result_class = 0
        elif idx_of_max == 2:
            result_class = 1

        classifications.append((tweet, result_class, actual_class, tweet_file))

    return classifications


def main():
    start = time.time()

    # For some reason, these geniuses made nltk.corpus.stopwords.words('english') as a list
    # Here I create a set for "constant" look-up.
    # Reduced time by 37 seconds.
    stop_words = set()
    obama_freq_words = {}
    romney_freq_words = {}

    test_data = []
    list_stop_words = nltk.corpus.stopwords.words('english')
    for word in list_stop_words:
        stop_words.add(word)

    with open('obama.csv', 'r', encoding='utf8') as csvfile, open('romney.csv', 'r', encoding='utf8') as csvfile2:

        reader = csv.DictReader(csvfile)
        reader2 = csv.DictReader(csvfile2)
        train_read_lines(reader, obama_freq_words, stop_words)
        train_read_lines(reader2, obama_freq_words, stop_words)

        print('END:')

        # At this point we've reached 80% of file read
        # Now we take the 20% of files and append to our test_data
        for line in reader:
            if line['classification'] != '2':
                test_data.append((line['Annotated tweet'], line['classification'], 'obama_file'))
        for line in reader2:
            if line['classification'] != '2':
                test_data.append((line['Annotated tweet'], line['classification'], 'romney_file'))

    for key in obama_freq_words:
        total = sum(obama_freq_words[key])
        obama_freq_words[key] = [(float(i) / total) for i in obama_freq_words[key]]
        print('obama_key: ', key, ' ', obama_freq_words[key],
              'index of max: ', obama_freq_words[key].index(max(obama_freq_words[key])))

    for key in romney_freq_words:
        total = sum(romney_freq_words[key])
        romney_freq_words[key] = [(float(i) / total) for i in romney_freq_words[key]]
        print('romney_key: ', key, ' ', romney_freq_words[key],
              'index of max: ', romney_freq_words[key].index(max(romney_freq_words[key])))

    result = test_read_line(test_data, obama_freq_words, romney_freq_words, stop_words)
    l_res = len(result)

    for i in range(-1, 2):
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for x in range(l_res):
            if result[x][2] != result[x][3]:
                # if i == 1:
                #     print(result[x])
                try:
                    if result[x][1] == i and result[x][1] == int(result[x][2].strip()):
                        tp += 1.0
                    elif result[x][1] == i and int(result[x][2].strip()) != i:
                        fp += 1.0
                    elif result[x][1] != i and int(result[x][2].strip()) == i:
                        fn += 1.0
                    elif result[x][1] != i and int(result[x][2].strip()) != i:
                        tn += 1.0
                except:
                    pass

        #accuracy = tp + tn / (tp + tn + fp + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f_score = 2 * ((precision * recall) / (precision + recall))
        #print('total: ', (tp + tn + fp + fn))
        print('class: ', i)
        #print('accuracy: ', accuracy)
        print('f-score: ', f_score, '\n')

    end = time.time()
    print('Total time: ', (end - start) * 1000)


if __name__ == '__main__':
    main()
