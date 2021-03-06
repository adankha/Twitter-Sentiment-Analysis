# Twitter Sentiment Analysis (Supervised Learning for Obama v Romney for the 2012 Presidential Elections)

###### Created By:
Ashour Dankha

## Introduction:

Sentiment Analysis (AKA Opinion Mining or Emotion AI) is a popular text analysis that typically focuses on determining whether a piece of writing is negative, neutral, or positive. SA essentially determines the opinion or attitude of what the message is.

The following program was done for a Research Project for a Graduate Level Data Mining Course. <br />
The following program uses a dataset of raw tweets for the Obama vs Romney 2012 Presidential Elections. <br />
The following program uses the SciKit-Learn Machine Learning Library for Python. <br />

## Goal:
The goal is to the achieve the highest F-Score for the Negative, Neutral, and Positive classifications (Note: the dataset contains other classifications). We do this by using many of the models provided in the Scikit-Learn ML Python Library and compare results to find the best F-Scores.

## Process:
Below is the 3-stage process that I took for this project. Each stage consists of multiple sub-stages/processes that is thoroughly explained in the pdf project write-up.

###### Pre-Processing Stage:
The first step is reading in the dataset(s) and preprocessing the data. Since they are raw tweets, there are multiple variables that are not necesscary when training your models. Some variables include the removal of, html tags, website links, accents, stop-words, and non-UTF-8 symbols.

*Note: Stop-words are words that are commonly used in sentences. (is, a, the, etc.)*

###### Model Comparison Stage:
The second step (once the noise has been cleaned up from the csv files) is to Fit the data (after some form of vectorization and parameter tuning using GridSearchCV) to multiple models and then compared to see which produces the highest result (based on F0-Score).

###### Graphical Representation Stage:
Once the models have been trained and tested, their statistics on Accuracy, Precision, Recall, and F-Score are recorded to create graphical representations.

Example:

![alt text](https://i.imgur.com/fHZ4lvR.png)



