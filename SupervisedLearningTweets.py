import re
import csv
import goslate
import nltk
import json
import time
import inflect
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
    # TODO: for train/test data. This will signficantly reduce column count 
        
    # Time taken to execute so far: 38 seconds
    # Bottlenecks: Stopwords. FIXED: Created a map 38k ms --> 400 ms, GG.
    
    # Tradeoff: Translated plurals if singular_nouns to singular
    # Increase time to 3 seconds, but reduce words down to 2.6k
"""

def add_to_dict(stop_words, row, word, words):

    if word not in stop_words:
        singular_word = p.singular_noun(word)
        if singular_word is not False:
            word = singular_word

        if word not in words:

            if row['classification'] == '-1':
                words[word] = [1, 0, 0, 0]
            elif row['classification'] == '0':
                words[word] = [0, 1, 0, 0]
            elif row['classification'] == '1':
                words[word] = [0, 0, 1, 0]
            elif row['classification'] == '2':
                words[word] = [0, 0, 0, 1]
        else:

            if row['classification'] == '-1':
                words[word][0] += 1
            elif row['classification'] == '0':
                words[word][1] += 1
            elif row['classification'] == '1':
                words[word][2] += 1
            elif row['classification'] == '2':
                words[word][3] += 1


def read_lines(reader, words, stop_words):

    regex = re.compile(r'<.*?>|https?[^ ]+|([@])[^ ]+|[^a-zA-Z ]+|\d+/?')


    for row in reader:
        line = row['Anootated tweet']
        line = regex.sub(' ', line)
        line = re.sub(' +', ' ', line).strip()
        line = line.encode('ascii', errors='ignore').decode('utf-8').lower()
        line = line.split(' ')
        for word in line:
            add_to_dict(stop_words, row, word, words)


def main():

    start = time.time()

    # For some reason, these geniuses made nltk.corpus.stopwords.words('english') as a list
    # Here I create a set for "constant" look-up.
    # Reduced time by 37 seconds.
    stop_words = set()
    words = {}
    list_stop_words = nltk.corpus.stopwords.words('english')
    for word in list_stop_words:
        stop_words.add(word)

    with open('obama.csv', 'r', encoding='utf8') as csvfile, open('romney.csv', 'r', encoding='utf8') as csvfile2:

        reader = csv.DictReader(csvfile)
        reader2 = csv.DictReader(csvfile2)
        read_lines(reader, words, stop_words)
        read_lines(reader2, words, stop_words)

    for key in words:
        print('key: ', key, ' ', words[key])

    # json_words = json.dumps(words)
    # loaded_words = json.loads(json_words)
    # print(loaded_words)

    print('Total number of words: ', len(words))
    end = time.time()
    print('Total time: ', (end - start) * 1000)


if __name__ == '__main__':
    main()
