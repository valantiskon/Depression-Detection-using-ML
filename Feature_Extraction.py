import EmoticonDetector
import numpy as np
import emoji
import re
from nltk.tokenize import word_tokenize

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