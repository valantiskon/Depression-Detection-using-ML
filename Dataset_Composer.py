import numpy as np
import pandas as pd
import os.path
from sklearn.model_selection import train_test_split


# Create GROUND TRUTH dataset
def ground_truth():
    dir = os.getcwd()  # Gets the current working directory
    df = pd.read_csv(dir + '\\dataset\\train\\imbalanced_tweets.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.10, random_state = 42)

    # Clear and combine datasets
    train = pd.DataFrame(list(zip(y_train, X_train)), columns=['label', 'tweet'])
    test = pd.DataFrame(list(zip(y_test, X_test)), columns=['label', 'tweet'])

    train = train.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    count_0, count_1 = train['label'].value_counts()
    print(count_1, count_0)

    count_0, count_1 = test['label'].value_counts()
    print(count_1, count_0)

    train.head(20)
    test.head(20)

    train.to_csv(dir + '\\dataset\\train\\training_imbalanced_temp.csv')
    test.to_csv(dir + '\\dataset\\train\\ground_truth.csv')

    print("END SCRIPT")



# CREATE BALANCED DATASET
def balance_dataset():
    dir = os.getcwd()  # Gets the current working directory

    train_file_A = dir + '\\dataset\\train\\training_imbalanced_temp.csv'

    train_A = pd.read_csv(train_file_A)
    # Drop the first column of reading file
    train_A.drop(['numb'], axis=1, inplace=True)


    label_0 = train_A.loc[train_A['label'] == 0]
    label_1 = train_A.loc[train_A['label'] == 1]
    print("label 0: ", label_0)
    print("label 1: ", label_1)

    getIndex= list()

    while len(getIndex) < len(label_1):
        for i in range(label_0.shape[0]):
            if np.random.uniform(0, 1) < 0.54 and i not in getIndex:
                getIndex.append(i)
    print(len(getIndex), len(label_1))
    getData = label_0.iloc[getIndex]
    print(getData)

    # Clear and combine datasets
    df_final = pd.concat([label_1, getData])
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    df_final.head(20)

    count_0, count_1 = df_final['label'].value_counts()
    print(count_1, count_0)

    df_final.to_csv(dir + '\\dataset\\train\\imbalanced_training.csv')

    print("END SCRIPT")




# Get as many non-depressive tweets as depressive tweets from SCRAPING
def Combine_Scraped_and_Positive_tweets():
    dir = os.getcwd()  # Gets the current working directory

    # Positive tweets
    train_file_A = dir + '\\dataset\\train\\general_tweets.csv'

    train_A = pd.read_csv(train_file_A)
    # Drop the first column of reading file
    train_A.drop(['numb'], axis=1, inplace=True)


    # Scraped tweets
    train_file_B = dir + '\\dataset\\train\\depress\\ALL_tweets_final.csv'

    label_1 = pd.read_csv(train_file_B)
    # Drop the first column of reading file
    label_1.drop(['Unnamed: 0'], axis=1, inplace=True)
    label_1.drop(['id'], axis=1, inplace=True)
    label_1.drop(['conversation_id'], axis=1, inplace=True)
    label_1.drop(['date'], axis=1, inplace=True)
    label_1.drop(['username'], axis=1, inplace=True)
    label_1.drop(['hashtags'], axis=1, inplace=True)
    label_1.drop(['tweet_original'], axis=1, inplace=True)


    label_0 = train_A.loc[train_A['label'] == 0]
    print("label 0: ", label_0)
    print("label 1: ", label_1)

    getIndex= list()

    while len(getIndex) < len(label_1):
        for i in range(label_0.shape[0]):
            if np.random.uniform(0, 1) < 0.32 and i not in getIndex:
                getIndex.append(i)
    print(len(getIndex), len(label_1))
    getData = label_0.iloc[getIndex]
    print(getData)

    # Clear and combine datasets
    df_final = pd.concat([label_1, getData])
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    df_final.head(20)

    print(df_final['label'].value_counts())

    df_final.to_csv(dir + '\\dataset\\train\\POSITIVE_DEPRESSED_SCRAPED.csv')

    print("END SCRIPT")




# COMBINE DATASETS
def combine_datasets():
    dir = os.getcwd()  # Gets the current working directory

    #train_file_A = dir + '\\dataset\\train\\depression_tweets.txt'
    train_file_A = dir + '\\dataset\\train\\TEMP_ALL_SPLIT_tweets_final.csv'


    train_A = pd.read_csv(train_file_A)
    # Drop the first column of reading file
    #train_A.drop(['numb'], axis=1, inplace=True)
    train_A.drop(['Unnamed: 0'], axis=1, inplace=True)
    train_A.drop(['id'], axis=1, inplace=True)
    train_A.drop(['conversation_id'], axis=1, inplace=True)
    train_A.drop(['date'], axis=1, inplace=True)
    train_A.drop(['username'], axis=1, inplace=True)
    train_A.drop(['hashtags'], axis=1, inplace=True)
    train_A.drop(['tweet_original'], axis=1, inplace=True)

    # Convert label from float to int
    import numpy as np
    train_A['label'] = np.where(train_A['label'] == 0.0, 0, 1)

    print(train_A)
    print(train_A.shape)

    #train_file_B = dir + '\\dataset\\train\\tweets_combined.csv'
    train_file_B = dir + '\\dataset\\train\\tweets_combined.csv'


    train_B = pd.read_csv(train_file_B)
    # Drop the first column of reading file
    train_B.drop(['numb'], axis=1, inplace=True)

    print(train_B)
    print(train_B.shape)

    # Clear and combine datasets
    df_final = pd.concat([train_A, train_B])
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    df_final.head(20)

    print(df_final)
    print(df_final.shape)

    df_final.to_csv(dir + '\\dataset\\train\\imbalanced_tweets.csv')

    print("END SCRIPT")


#ground_truth()
#Combine_Scraped_and_Positive_tweets()
balance_dataset()
#combine_datasets()