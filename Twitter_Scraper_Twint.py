import nest_asyncio
nest_asyncio.apply()
import twint
import pandas as pd
import re
import os.path

# add some tweets with depressed and depression tags, for a particular year

#depress_tags = ["#depressed", "#depression", "#loneliness", "#hopelessness"]
depress_tags = ["I lost interest in doing things", "I feel down", "depressed", "hopeless",
                "I have trouble falling asleep", "I have trouble staying asleep", "I am sleeping too much",
                "I feel tired", "I have little energy", "I become easily fatigued",
                "I have poor appetite", "I observed changes in my appetite", "I am overeating",
                "I feel bad about myself", "I am a failure", "I have let myself down", "I have let my family down",
                "I am struggling with feelings of worthlessness", "I am struggling with feelings of worthlessness",
                "I am struggling with feelings of guilt", "I have trouble concentrating",
                "I struggle to keep my attention", "I would be better off dead", "I struggle with thoughts of suicide",
                "I struggle with thoughts of hurting myself", "I become irritated easily", "I become angry for no reason"]

dir = os.getcwd()

content = {}
for i in range(len(depress_tags)):
    print(depress_tags[i])
    c = twint.Config()

    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 250
    c.Year = 2019
    c.Store_csv = True
    c.Store_Object = True
    c.Output = dir + '\\dataset\\train\\depress\\UNPROCESSED_dataset_en_all19.csv'
    c.Hide_output = True
    c.Stats = True
    c.Lowercase = True
    c.Filter_retweets = True
    twint.run.Search(c)

# add more examples of depressed and depression tags, but with another year so it doesnt overlap

#depress_tags = ["#depressed", "#depression"]
depress_tags = ["I was diagnosed with PTSD", "I have depression"]

content = {}
for i in range(len(depress_tags)):
    c = twint.Config()

    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 250
    c.Year = 2019
    c.Store_csv = True
    c.Store_Object = True
    c.Output = dir + '\\dataset\\train\\depress\\UNPROCESSED_dataset_en_all18.csv'
    c.Hide_output = True
    c.Stats = True
    c.Lowercase = True
    twint.run.Search(c)

df1 = pd.read_csv(dir + '\\dataset\\train\\depress\\UNPROCESSED_dataset_en_all19.csv')
df2 = pd.read_csv(dir + '\\dataset\\train\\depress\\UNPROCESSED_dataset_en_all18.csv')
#df_all = pd.concat([df1, df2])
df_all = df1

# Check for the size of each dataset
print("len 2019: ", len(df1), "len 2018: ", len(df2), "len both: ", len(df_all))

print("hashtags 2019: ", df1.hashtags.value_counts())


print("number of hashtags 2019: ", len(df_all.id.value_counts()))

# 1. Combine dataset and remove duplicates based on id and tweet content
df_all = df_all.drop_duplicates(subset=["id"])
print(df_all.shape)

pd.set_option('display.max_colwidth', -1)
print("all 2018-2019 print: ", df_all.head())
print("all 2018-2019 hashtags: ", df_all.hashtags.value_counts().head(20))

# Let's have a look at an example where there are the same long stream of tags reoccurring many times. That looks suspiciously like a marketing message
print(df_all[df_all["hashtags"] =="['#depression', '#hopelessness', '#invisibleillness', '#robinwilliams', '#socialmedia', '#suicide']"])

# 2. Filtering out the relevant rows
#
# Ideas for cleaning / filtering
#
# remove entries that contain positive, or medical sounding tags
# remove entries with more than three hashtags, as it may be promotional messages
# remove entries with at mentions, as it may be promotional messages
# remove entries with less than x chars / words
# remove entries containing urls - again as they are likely to be promotional messages

selection_to_remove = ["#mentalhealth", "#health", "#happiness", "#mentalillness", "#happy", "#joy", "#wellbeing"]

# 1. remove entries that contain positive, or medical sounding tags
mask1 = df_all.hashtags.apply(lambda x: any(item for item in selection_to_remove if item in x))
df_all[mask1].tweet.tail()

# review the result of removing certain tags
print("removing hashtags: ", df_all[mask1==False].tweet.head(10))

# above results look good, let's apply the mask1
df_all = df_all[mask1==False]
print("len of all: ", len(df_all))

# 2. remove entries with more than three hashtags, as it may be promotional messages
mask2 = df_all.hashtags.apply(lambda x: x.count("#") < 4)

# applying the mask2
df_all = df_all[mask2]

#Check dataset size
print("removing tweets with more than three hashtags, len: ", len(df_all))
print("some tweets : ", df_all.head())

# 3. remove tweets with at mentions as they are sometimes retweets
mask3 = df_all.mentions.apply(lambda x: len(x) < 5)
# applying mask3
df_all = df_all[mask3]
print("removing mentions,len: ", len(df_all))

# let's check the hashtags value counts again
print("remove mentions head: ", df_all.hashtags.value_counts().head(20))
print("remove mentions tail: ", df_all.tweet.tail(10))

# 4. remove entries with less than x chars / words
mask4a = df_all.tweet.apply(lambda x: len(x) > 25)
df_all = df_all[mask4a]
print("remove little tweets: ", len(df_all))
mask4b = df_all.tweet.apply(lambda x: x.count(" ") > 5)
df_all = df_all[mask4b]
print("len all: ", len(df_all))
print("all tweets: ", df_all.tweet)

# 5. remove entries containing urls - as they are likely to be promotional messages
mask5 = df_all.urls.apply(lambda x: len(x) < 5)

# let's have a look at what we will be removing from the dataset
print("remove head: ", df_all[mask5==False].tweet.head(10), "remove tail: ", df_all[mask5==False].tweet.tail(10))
df_all = df_all[mask5]
print(len(df_all))

# 3. Finally, let's create a column containing the tweet text, but with all hashtags removed
# This column can be used as input to the model, or can be sent to another software for further emotion and
# linguistic analysis. The idea is, if the hashtags are removed, the model and the software will examine the
# text and clairy if the actual emotion is negative and indicative of depression
df_all["mod_text"] = df_all["tweet"].apply(lambda x: re.sub(r'#\w+', '', x))
print("remove head: ", df_all.mod_text.head(15), "remove tail: ", df_all.mod_text.tail(15))

# let's check the hashtags value counts again
df_all.hashtags.value_counts().head(20)
print("all columns", df_all.columns)
col_list = ["id", "conversation_id", "date", "username", "mod_text", "hashtags", "tweet"]
df_final1 = df_all[col_list]
df_final1 = df_final1.rename(columns={"mod_text": "tweet_processed", "tweet": "tweet_original"})
df_final1["target"] = 1
print("remove head: ", df_final1.head())

print("len(df_final1): ", len(df_final1))
len_df_final1 = len(df_final1)

df_final1_1 = df_final1[:int(len_df_final1/4)]
df_final1_2 = df_final1[int(len_df_final1/4):int(len_df_final1/2)]
df_final1_3 = df_final1[int(len_df_final1/2):int(3*len_df_final1/4)]
df_final1_4 = df_final1[int(3*len_df_final1/4):]
print("len(df_final1_1)", len(df_final1_1), "len(df_final1_2)", len(df_final1_2), "len(df_final1_3)", len(df_final1_3), "len(df_final1_4)", len(df_final1_4))

df_final1.to_csv(dir + "\\dataset\\train\\depress\\ALL_tweets_final.csv")

df_final1_1.to_csv(dir + "\\dataset\\train\\depress\\SPLIT1_tweets_final_1.csv")
df_final1_2.to_csv(dir + "\\dataset\\train\\depress\\SPLIT2_tweets_final_2.csv")
df_final1_3.to_csv(dir + "\\dataset\\train\\depress\\SPLIT3_tweets_final_3.csv")
df_final1_4.to_csv(dir + "\\dataset\\train\\depress\\SPLIT4_tweets_final_4.csv")


df_all.to_csv(dir + "\\dataset\\train\\depress\\USELESS_ALL_INFO_tweets_v3.csv")

users = df2.username


# For each user collected from above methods, scrap 30 tweets containing #depressed
content = {}
for i in users:  # users1['Names']:

    c = twint.Config()
    c.Search = "#depressed"
    c.Username = "noneprivacy"
    c.Username = i
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Limit = 30
    c.Store_csv = True
    c.Store_Object = True
    c.Output = dir + "\\dataset\\train\\depress\\GET_TWEETS_FROM_SCRAPED_USERS_dataset_v3.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase = True
    twint.run.Search(c)

    #     tweets = twint.output.tweets_list()
    #     print(tweets)
    #     for tweet in tweets:
    #     # then iterate over the hashtags of that single tweet
    #         for t in tweet.tweet:
    #         # increment the count if the hashtag already exists, otherwise initialize it to 1
    #             if tweet.username in content:
    #                 content[tweet.username].append(t)
    #             else:
    #                 content[tweet.username] = []
    #                 content[tweet.username].append(t)

    print(i)
#     print(content)
#     with open('dataset.csv', 'w') as output:
#         output.write('username, tweet\n')
#         for user in content:
#             for h in content[user]:
#                 output.write('{},{}\n'.format(user, content[user][h]))
