import re
import csv
import goslate

# TODO: Find a better way to figure out if a Tweet is in English or not
# TODO: Create FREQ count for tweets
"""
NAIVE APPROACH:
    Step 1: Create a list for Romney and Obama where the map holds every Tweet and their classification.
        Note: Map will be word -> list[frequency_classification] 
        i.e. "monkey" -> [freq_positive, freq_negative, freq_neutral, freq_mixed]
        
    Step 2: Parse each tweet in the following way.
        For each word in the tweet
        Check to see if word exists in map
        If it does, +1 to the corresponding classification
        


"""


with open('obama.csv', 'r', encoding='utf8') as csvfile, open('romney.csv', 'r', encoding='utf8') as csvfile2:

    reader = csv.DictReader(csvfile)
    reg = re.compile(r'<.*?>|https?[^ ]+|([@#])[^ ]+|@?#?\'?\"?')
    gs = goslate.Goslate()
    for row in reader:

        line = row['Anootated tweet']
        line = reg.sub('', line)
        line = re.sub(' +|\d+/?', ' ', line)
        line = line.encode('ascii', errors='ignore').decode('utf-8')
        #line = gs.translate(line, 'en')

        print(line)

    reader = csv.DictReader(csvfile2)

    for row in reader:
        line = row['Anootated tweet']
        line = reg.sub('', line)
        line = re.sub(' +|\d+/?', ' ', line)
        line = line.encode('ascii', errors='ignore').decode('utf-8')
        #line = gs.translate(line, 'en')

        print(line)

