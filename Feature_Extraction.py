import EmoticonDetector
import numpy as np
import emoji
import re
from nltk.tokenize import word_tokenize
import flair
import os.path
import pandas as pd

#import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class TwitterData_ExtraFeatures:
    processed_data = []

    def __init__(self):
        pass

    def build_features(self, train_A):
        self.processed_data = []

        raw_text = [] # UNPROCESSED TEXT OF TWEETS
        tokenized_text = [] # TOKENIZED TEXT OF TWEETS
        for sentence in train_A['tweet']:
            raw_text.append(sentence)
            # Tokenize tweets
            tokenized_text.append(word_tokenize(sentence))

        print(raw_text)
        print(tokenized_text)

        #1 number of uppercase words
        uppercase = []
        upper_count = 0 # The count of uppercase words in a tweet
        for tweets in tokenized_text:
            for word in tweets:
                if word.isupper():
                    upper_count += 1
            uppercase.append(upper_count)
            upper_count = 0
        self.processed_data.append(uppercase)

        count = 0
        for numb in uppercase:
            count += numb
        print("Total count of uppercase words from all tweets: ", count)
        print("Average count of uppercase words per tweet: ", count / len(uppercase))



        #2 number of !
        exclamations = []
        for tweets in tokenized_text:
            exclamations.append(tweets.count("!"))
        self.processed_data.append(exclamations)

        count = 0
        for numb in exclamations:
            count += numb
        print("Total count of exclamations from all tweets: ", count)
        print("Average count of exclamations per tweet: ", count / len(exclamations))



        #3 number of mentions
        mentions = []
        for tweets in tokenized_text:
            mentions.append(tweets.count("@"))
        self.processed_data.append(mentions)

        count = 0
        for numb in mentions:
            count += numb
        print("Total count of mentions from all tweets: ", count)
        print("Average count of mentions per tweet: ", count / len(mentions))



        #4 number of sad emoticons
        emot_detect = EmoticonDetector.EmoticonDetector()
        raw_text = train_A['tweet'].str.replace('http\S+|www.\S+', '', case=False) # Delete URLs to fix error of detecting the emoji :/ in a http://... link
        emots = [] # Saves the number of emojis per tweet
        sad_list_emojis = [emoji.emojize(":crying_cat_face:"), emoji.emojize(":angry:"), emoji.emojize(":disappointed:"),
                           emoji.emojize(":cry:"), emoji.emojize(":rage:"), emoji.emojize(":worried:"), emoji.emojize(":fearful:")]
        for tweet in raw_text:
            # emoj_counter: Counts the number of emojis in a tweet
            # Adding whitespace at the end of each tweet to match the whitespace at the end of each emoji saved at the emoticons.txt file
            emoj_counter = emot_detect.count_emoticons(tweet + ' ') # Count emoticons that are created using text, like :D
            for token in tweet:
                if token in sad_list_emojis:
                    emoj_counter += 1
            emots.append(emoj_counter)
        self.processed_data.append(emots)

        count = 0
        for numb in emots:
            count += numb
        print("Total count of emoticons from all tweets: ", count)
        print("Average count of emoticons per tweet: ", count / len(emots))
        print("extra features shape: ", np.array(self.processed_data).shape)



        # 5 percentange of pronouns
        personal_pronouns = {"I", "me", "my", "mine", "myself"}

        rest_pronouns = {"you", "he", "she", "it", "we", "they", "her", "him", "us", "them", "our", "your", "his",
                         "its", "their", "ours", "yours", "hers", "theirs", "himself", "herself", "themselves",
                         "itself", "yourself", "yourselves", "ourselves"}

        pronoun_percentage = []
        # calculate stats for total count of personal and rest pronouns
        personal = 0
        rest = 0
        for tweets in tokenized_text:
            count_rest = 0
            count_personal = 0
            for x_rest in rest_pronouns:
                count_rest += tweets.count(x_rest)
            for x_personal in personal_pronouns:
                count_personal += tweets.count(x_personal)
            rest += count_rest
            personal += count_personal
            if count_rest:
                pronoun_percentage.append(count_personal / count_rest)
            elif count_personal: # If there are no rest pronouns in this tweet but personal pronouns exist
                pronoun_percentage.append(1)
            else: # If there are no pronouns in this tweet
                pronoun_percentage.append(0)
        self.processed_data.append(pronoun_percentage)

        count = 0
        for numb in pronoun_percentage:
            count += numb
        print("Total count of PERSONAL PRONOUNS from all tweets: ", personal,
              "\nTotal count of REST PRONOUNS from all tweets: ", rest)
        print("Average count of PERSONAL PRONOUNS from all tweets: ", personal / len(pronoun_percentage),
              "\nAverage count of REST PRONOUNS from all tweets: ", rest / len(pronoun_percentage))
        print("Average percentage of PERSONAL/REST PRONOUNS per tweet: ", count / len(pronoun_percentage))




        # 6 sentiment analysis of a tweet (4 FEATURES IN 1 PROCESS)
     #   flair_sentiment = flair.models.TextClassifier.load('en-sentiment')

        vader_sentiment = SentimentIntensityAnalyzer()
        print(self.processed_data)

        negative_emot = []
        neutral_emot = []
        positive_emot = []
        compound_emot = []
        for tweet in raw_text:
            #    s = flair.data.Sentence(tweet)
            #     flair_sentiment.predict(s)
            #      flair_total_sentiment = s.labels
            #       print("flair: ", flair_total_sentiment)

            vader_total_sentiment = vader_sentiment.polarity_scores(tweet)

            #print("vader: ", vader_total_sentiment)

            negative_emot.append(vader_total_sentiment["neg"])
            neutral_emot.append(vader_total_sentiment["neu"])
            positive_emot.append(vader_total_sentiment["pos"])
            compound_emot.append(vader_total_sentiment["compound"])

        self.processed_data.append(negative_emot)
        self.processed_data.append(neutral_emot)
        self.processed_data.append(positive_emot)
        self.processed_data.append(compound_emot)

        for x in self.processed_data:
            print("features: ", x)



        # 7 Count of absolute words
        absolute_words = ["no one", "all", "nobody", "everybody", "every", "never", "each", "none", "everyone", "always",
                         "nothing", "completely", "only", "complete", "absolute", "end",  "empty", "entirely", "fatal",
                         "dead", "final", "full", "fully", "impossible"]

        save_count_absolute = []
        total_absolute = 0
        for tweets in tokenized_text:
            count_absolute = 0
            for x_absolute in absolute_words:
                count_absolute += tweets.count(x_absolute)
            total_absolute += count_absolute
            save_count_absolute.append(count_absolute)

        self.processed_data.append(save_count_absolute)


        mean_absolute_per_tweet = total_absolute / len(tokenized_text)
        absolute_percentage = []
        for count in save_count_absolute:
            if mean_absolute_per_tweet:
                absolute_percentage.append(count / mean_absolute_per_tweet)
            else:
                absolute_percentage.append(0)


        print("total absolute: ", total_absolute)
        print("percentage absolute: ", absolute_percentage)
        print("mean absolute words per tweet: ", mean_absolute_per_tweet)



        # 8 Count of words conveing sad feelings and symptoms of depression (percenage of number of depressive words divided by the average number of depressive words in all twees)
        dir = os.getcwd()  # Gets the current working directory
        depr_lex = dir + '\\dataset\\train\\depression_lexicon.txt'

        with open(depr_lex, "r", encoding="utf8") as fd:
            depr_lexicon = fd.read().splitlines()

        save_count_lex = []
        total_lex = 0
        for tweets in tokenized_text:
            count_lex = 0
            for x_lex in depr_lexicon:
                count_lex += tweets.count(x_lex)
            total_lex += count_lex
            save_count_lex.append(count_lex)

        self.processed_data.append(save_count_lex)


        mean_lex_per_tweet = total_lex / len(tokenized_text)
        absolute_percentage = []
        for count in save_count_lex:
            if mean_lex_per_tweet:
                absolute_percentage.append(count / mean_lex_per_tweet)
            else:
                absolute_percentage.append(0)


        print("total lexicon words: ", total_lex)
        print("percentage lexicon: ", absolute_percentage)
        print("mean lexicon words per tweet: ", mean_lex_per_tweet)


        print("depr_lexicon: ", depr_lexicon)