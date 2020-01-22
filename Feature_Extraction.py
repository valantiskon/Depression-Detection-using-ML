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

        # Used to count the number of quotes in a tweet
        def count_occurences(character, word_array):
            counter = 0
            for j, word in enumerate(word_array):
                for char in word:
                    if char in character:
                        counter += 1
            if counter % 2 != 0: # Used for estimation in case we have count that is not divided by two exactly. In that case we subtrack 1 from counter to estimate the number of queotes
                counter -= 1
            return counter


        self.processed_data = []

        raw_text = [] # UNPROCESSED TEXT OF TWEETS
        tokenized_text = [] # TOKENIZED TEXT OF TWEETS
        for i in range(0, len(train_A)):
            raw_text.append(train_A.iloc[i][0])
            # Tokenize tweets
            tokenized_text.append(word_tokenize(train_A.iloc[i][0]))

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


        #3 number of ?
        questions = []
        for tweets in tokenized_text:
            questions.append(tweets.count("?"))
        self.processed_data.append(questions)

        count = 0
        for numb in questions:
            count += numb
        print("Total count of question marks from all tweets: ", count)
        print("Average count of question marks per tweet: ", count / len(questions))



        #4 number of ...
        ellipsis = []
        for tweets in tokenized_text:
            ellipsis.append(tweets.count("..."))
        self.processed_data.append(ellipsis)

        count = 0
        for numb in ellipsis:
            count += numb
        print("Total count of ellipsis from all tweets: ", count)
        print("Average count of ellipsis per tweet: ", count / len(ellipsis))



        #5 number of hashtags
        hashtags = []
        for tweets in tokenized_text:
            hashtags.append(tweets.count("#"))
        self.processed_data.append(hashtags)

        count = 0
        for numb in hashtags:
            count += numb
        print("Total count of hashtags from all tweets: ", count)
        print("Average count of hashtags per tweet: ", count / len(hashtags))



        #6 number of mentions
        mentions = []
        for tweets in tokenized_text:
            mentions.append(tweets.count("@"))
        self.processed_data.append(mentions)

        count = 0
        for numb in mentions:
            count += numb
        print("Total count of mentions from all tweets: ", count)
        print("Average count of mentions per tweet: ", count / len(mentions))



        #7 number of quotes
        quote_list = ["'", '"'] # List of quotes
        quotes = list(map(lambda plain_text: int(count_occurences(quote_list, [plain_text]) / 2), raw_text))
        self.processed_data.append(quotes)

        count = 0
        for numb in quotes:
            count += numb
        print("Total count of quotes from all tweets: ", count)
        print("Average count of quotes per tweet: ", count / len(quotes))



        #8 number of urls
        urls = [] # Keep the number of URLs per tweet
        for tweets in raw_text:
            count = len(re.findall('http\S+|www.\S+', tweets))
            urls.append(count)
        self.processed_data.append(urls)

        count = 0
        for numb in urls:
            count += numb
        print("Total count of urls from all tweets: ", count)
        print("Average count of urls per tweet: ", count / len(urls))



        #9 number of emoticons
        emot_detect = EmoticonDetector.EmoticonDetector()
        raw_text = train_A['tweet'].str.replace('http\S+|www.\S+', '', case=False) # Delete URLs to fix error of detecting the emoji :/ in a http://... link
        emots = [] # Saves the number of emojis per tweet
        for tweet in raw_text:
            # emoj_counter: Counts the number of emojis in a tweet
            # Adding whitespace at the end of each tweet to match the whitespace at the end of each emoji saved at the emoticons.txt file
            emoj_counter = emot_detect.count_emoticons(tweet + ' ') # Count emoticons that are created using text, like :D
            for token in tweet:
                if token in emoji.UNICODE_EMOJI:
                    emoj_counter += 1
            emots.append(emoj_counter)
        self.processed_data.append(emots)

        count = 0
        for numb in emots:
            count += numb
        print("Total count of emoticons from all tweets: ", count)
        print("Average count of emoticons per tweet: ", count / len(emots))
        print("extra features shape: ", np.array(self.processed_data).shape)



        # 10 percentange of pronouns
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
                count_rest = tweets.count(x_rest)
            for x_personal in personal_pronouns:
                count_personal = tweets.count(x_personal)
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




        # 11 sentiment analysis of a tweet (4 FEATURES IN 1 PROCESS)
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

            print("vader: ", vader_total_sentiment)

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



        # 12 Count of absolute words
        absolute_words = ["no one", "all", "nobody", "everybody", "every", "never", "each", "none", "everyone", "always",
                         "nothing", "completely", "only", "complete", "absolute", "end",  "empty", "entirely", "fatal",
                         "dead", "final", "full", "fully", "impossible"]

        save_count_absolute = []
        total_absolute = 0
        for tweets in tokenized_text:
            count_absolute = 0
            for x_absolute in absolute_words:
                count_absolute = tweets.count(x_absolute)
            total_absolute += count_absolute
            save_count_absolute.append(count_absolute)

        mean_absolute_per_tweet = total_absolute / len(tokenized_text)

        absolute_percentage = []
        for count in save_count_absolute:
            if mean_absolute_per_tweet:
                absolute_percentage.append(count / mean_absolute_per_tweet)
            else:
                absolute_percentage.append(0)

        self.processed_data.append(absolute_percentage)

        print("percentage absolute: ", absolute_percentage)
        print("absolute per tweet: ", save_count_absolute)
        print("mean absolute words per tweet: ", mean_absolute_per_tweet)



        # 13 Count of words conveing sad feelings and symptoms of depression (percenage of number of depressive words divided by the average number of depressive words in all twees)
        dir = os.getcwd()  # Gets the current working directory
        depr_lex = dir + '\\dataset\\train\\depression_lexicon.txt'

        with open(depr_lex, "r", encoding="utf8") as fd:
            depr_lexicon = fd.read().splitlines()

        save_count_lex = []
        total_lex = 0
        for tweets in tokenized_text:
            count_lex = 0
            for x_lex in depr_lexicon:
                count_lex = tweets.count(x_lex)
            total_lex += count_lex
            save_count_lex.append(count_lex)

        mean_lex_per_tweet = total_lex / len(tokenized_text)

        absolute_percentage = []
        for count in save_count_lex:
            if mean_lex_per_tweet:
                absolute_percentage.append(count / mean_lex_per_tweet)
            else:
                absolute_percentage.append(0)

        self.processed_data.append(absolute_percentage)

        print("percentage lexicon: ", absolute_percentage)
        print("lexicon words per tweet: ", save_count_lex)
        print("mean lexicon words per tweet: ", mean_lex_per_tweet)


        print("depr_lexicon: ", depr_lexicon, depr_lexicon[3])

        # 14 Percentage computed from the the total number of verbs in past time divided by current/future time

