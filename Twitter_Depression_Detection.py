import os.path
import numpy as np
import pandas as pd
from nltk.util import ngrams
import gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import Feature_Extraction
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import scale
from sklearn import utils
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer

import nltk
nltk.download('stopwords')


##############################################################################################################################################################

class Reader:
    dir = os.getcwd()  # Gets the current working directory

    train_A = None # dataframe of the dataset

    words_of_tweets = []  # Saves all the tweet cleared from stop-words, stemmed and tokenized

    called_once = False  # Indicates if the GloVe model has been trained (read) or not

    onehot_encoder = CountVectorizer()

    scaler = MinMaxScaler(feature_range=(0, 1))

    tester = MinMaxScaler(feature_range=(0, 1))

    def dummy_fun(self, doc):
        return doc

    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun)

    # min_df : float in range [0.0, 1.0] or int, default=1
    # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
    # This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents,
    # integer absolute counts. This parameter is ignored if vocabulary is not None.
    vectorizer1 = TfidfVectorizer(analyzer=lambda x: x, min_df=7)

    # sg: CBOW if 0, skip-gram if 1
    # ‘min_count’ is for neglecting infrequent words.
    # negative (int) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
    # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
    model = Word2Vec()

    # dm: DBOW if 0, distributed-memory if 1
    # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
    modeldoc = Doc2Vec()

    # GloVe model
    glove_model = {}

    # Feature Selection

    # Univariate_Selection
    test = SelectKBest(score_func=chi2, k=100)

    # Feature Extraction with RFE# Feature Extraction with Recursive Feature Elimination
    rfe = RFE(model, 100)

    # Feature Extraction with PCA
    pca = PCA(n_components=100)

    # Feature Extraction with TruncatedSVD
    svd = TruncatedSVD(n_components=100)

    # Feature Importance with Extra Trees Classifier
    sfm = RandomForestClassifier()
    models = SelectFromModel(sfm)


    ##############################################################################################################################################################

    # Pre-processing and convert the input using one hot encoding, TF-IDF and other encoders

    ##############################################################################################################################################################

    def tokenize(self, text):
        # Tokenize tweets
        words = word_tokenize(text)

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        words = [w.translate(table) for w in words]

        # remove all tokens that are not alphabetic
        words = [word for word in words if word.isalpha()]

        # Delete Stop-Words
        whitelist = ["n't", "not"]  # Keep the words "n't" and "not"
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if w not in stop_words or w in whitelist]
        stopwords_wordcloud = set(STOPWORDS)
        words = [w for w in words if w not in stopwords_wordcloud or w in whitelist]

        return words

    # Print the counts of the top 85 most used words and print a graph with the words of the data set
    def wordcloud(self):
        stopwords_wordcloud = set(STOPWORDS)

        # Print the counts of the top 85 most used words in tweets

        vectorizer = CountVectorizer(
            analyzer='word',
            tokenizer=self.tokenize,
            lowercase=True,
            stop_words=stopwords_wordcloud,
            max_features=85
        )

        corpus_words = vectorizer.fit_transform(self.train_A['tweet'])
        corpus_words = corpus_words.toarray()
        vocab = vectorizer.get_feature_names()

        # Sum up the counts of each vocabulary word
        dist = np.sum(corpus_words, axis=0)

        # For each, print the vocabulary word and the number of times it
        # appears in the data set
        for tag, count in zip(vocab, dist):
            print(count, ' ', tag)

        # Print a scheme with most used words that are not stopwords
        wordcloud = WordCloud(background_color="black", stopwords=stopwords_wordcloud, random_state=500, relative_scaling=1.0,
                              colormap='summer').generate(" ".join([i for i in self.train_A['tweet']]))
        plt.figure(facecolor='k')
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.title("Most used words in tweets")
        plt.show()

        # Print a scheme with most used POSITIVE words that are not stopwords
        wordcloud_positive = WordCloud(background_color="black", stopwords=stopwords_wordcloud, random_state=500,
                              relative_scaling=1.0,
                              colormap='summer').generate(" ".join([i for i in self.train_A['tweet'][self.train_A['label'] == 0]]))
        plt.figure(facecolor='k')
        plt.imshow(wordcloud_positive)
        plt.axis("off")
        plt.title("Most used words in POSITIVE tweets")
        plt.show()

        # Print a scheme with most used DEPRESSIVE words that are not stopwords
        wordcloud_depressive = WordCloud(background_color="black", stopwords=stopwords_wordcloud, random_state=500,
                              relative_scaling=1.0,
                              colormap='summer').generate(" ".join([i for i in self.train_A['tweet'][self.train_A['label'] == 1]]))
        plt.figure(facecolor='k')
        plt.imshow(wordcloud_depressive)
        plt.axis("off")
        plt.title("Most used words in DEPRESSIVE tweets")
        plt.show()

    ##############################################################################################################################################################

    # Pre-processing of the tweets
    def pre_processing(self):
        # Feature Extraction
        data = Feature_Extraction.TwitterData_ExtraFeatures()
        data.build_features(self.train_A)
        self.extra_features = data.processed_data

        # Clearing training dataset and Integer Encoding

        # Delete URLs
        self.train_A['tweet'] = self.train_A['tweet'].str.replace('http\S+|www.\S+', '', case=False)
        # Delete Usernames
        self.train_A['tweet'] = self.train_A['tweet'].str.replace(r'@\S+', '', case=False)
        # Replace hashtags with space to deal with the case where the tweet appears to be one word but is consisted by more seperated from hashtags
        self.train_A['tweet'] = self.train_A['tweet'].str.replace(r'#', ' ', case=False)

#        print('Average number of words per sentence: ', np.mean([len(s.split(" ")) for s in self.train_A.tweet]))

        for sentence in self.train_A['tweet']:
            # substitute contractions with full words
            words = self.replace_contractions(sentence)

            # Tokenize tweets
            words = word_tokenize(words)

            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            words = [w.translate(table) for w in words]

            # remove all tokens that are not alphabetic
            words = [word for word in words if word.isalpha()]

            # stemming of words
            porter = PorterStemmer()
            words = [porter.stem(word) for word in words]

            # Delete Stop-Words
            whitelist = ["n't", "not", 'nor', "nt"]  # Keep the words "n't" and "not", 'nor' and "nt"
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words or w in whitelist]

            # Keep the tokenized tweets
            self.words_of_tweets.append(words)

        # self.wordcloud() # Print number of 85 most used words and a scheme with most used words that are not stopwords

    def get_contractions(self):
        contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                            "could've": "could have", "couldn't": "could not", "didn't": "did not",
                            "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
                            "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
                            "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                            "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                            "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have",
                            "i'll": "i will", "i'll've": "i will have", "i'm": "i am", "i've": "i have",
                            "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                            "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                            "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not",
                            "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
                            "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have",
                            "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                            "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                            "she'll've": "she will have", "she's": "she is", "should've": "should have",
                            "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",
                            "so's": "so as", "this's": "this is", "that'd": "that would",
                            "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                            "there'd've": "there would have", "there's": "there is", "here's": "here is",
                            "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                            "they'll've": "they will have", "they're": "they are", "they've": "they have",
                            "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
                            "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have",
                            "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                            "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                            "when've": "when have", "where'd": "where did", "where's": "where is",
                            "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                            "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have",
                            "will've": "will have", "won't": "will not", "won't've": "will not have",
                            "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                            "y'all": "you all", "y'all'd": "you all would", "y'all'd've": "you all would have",
                            "y'all're": "you all are", "y'all've": "you all have", "you'd": "you would",
                            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                            "you're": "you are", "you've": "you have"}

        contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
        return contraction_dict, contraction_re


    def replace_contractions(self, text):
        contractions, contractions_re = self.get_contractions()
        def replace(match):
            return contractions[match.group(0)]

        return contractions_re.sub(replace, text)

###############################################################################################################################################
    ###############################################################################################################################################

    # Select the proper encoding and Feature Selection
    # x_enc: training data set or test data set
    # train_test: whether x_enc is training set or test set
    # y: the irony labels of either the training set or the test set
    # dataset_index: the indexes of train set or test set
    # extra_features: Added features from feature extraction
    # feature_selection: number that indicates what feature selection algorithm will be used
    # encoding: number that indicates what encoding algorithm will be used
    # print_file: the file name that the print will be written
    def get_enc(self, x_enc, train_test, y, dataset_index, extra_features, feature_selection, encoding, print_file):
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Encodings
        encoded_tweets = []

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # TF-IDF
        if encoding == 1:
            encoded_tweets = self.tf_idf(x_enc, train_test).toarray()  # Used to convert sparse matrix (produced from TF-IDF) to dense matrix (needed for concatenate)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # One hot encoding
        if encoding == 2:
            encoded_tweets = self.one_hot_enc(x_enc, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Bi-grams
        if encoding == 3:
            encoded_tweets = self.bigrams_enc(x_enc, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Word2Vec
        if encoding == 4:
            encoded_tweets = self.Word2Vec_enc(x_enc, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Doc2Vec
        if encoding == 5:
            encoded_tweets = self.Doc2Vec_enc(x_enc, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # GloVe
        if encoding == 6:
            encoded_tweets = self.GloVe_enc(x_enc, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Feature Selection

        # Format the features from Feature Extraction
        extra_features = zip(*extra_features)  # * in used to unzip the list, result is transposed rows with columns. Rows changed to number of tweets and columns changed to number of features
        extra_features = list(extra_features)
        extra_features = np.array(extra_features)
        extra_features = extra_features[dataset_index]
        print("features chosen shape: ", extra_features.shape)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("features chosen shape: " + str(extra_features.shape) + '\n')

        # Normalize each of the columns of the added features form Feature Selection

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("features before normalization: " + str(extra_features) + '\n')

        if train_test == 1:  # Train set
            # train the normalization
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler = self.scaler.fit(extra_features)
            # normalize the train dataset
            extra_features = self.scaler.transform(extra_features)

        if train_test == 0:  # Test set
            # normalize the test dataset
            extra_features = self.scaler.transform(extra_features)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("features after normalization: " + str(extra_features) + '\n')

        # Adding features to encoded_tweets
        print("encoded_tweets before tweets shape: ", encoded_tweets.shape)
        print("before tweets extra_features shape: ", extra_features.shape)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("encoded_tweets before tweets shape: " + str(encoded_tweets.shape) + '\n' + "before tweets extra_features shape: " + str(extra_features.shape) + '\n' + "before encoded_tweets: " + str(encoded_tweets) + '\n')

        encoded_tweets = numpy.concatenate((encoded_tweets, extra_features), axis=1)
        encoded_tweets = np.array(encoded_tweets)
        print("final encoded_tweets shape: ", encoded_tweets.shape)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("final encoded_tweets shape: " + str(encoded_tweets.shape) + '\n' + "final encoded_tweets: " + str(encoded_tweets) + '\n')

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Univariate Selection

        #  One-hot-encoding, TF-IDF, Bigrams
        if feature_selection == 7:
            encoded_tweets = self.Univariate_Selection(encoded_tweets, y, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Recursive Feature Elimination

        #  One-hot-encoding, TF-IDF, Bigrams
        if feature_selection == 8:
            encoded_tweets = self.Recursive_Feature_Elimination(encoded_tweets, y, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Principal Component Analysis

        #  One-hot-encoding, TF-IDF, Bigrams
        if feature_selection == 9:
            encoded_tweets = self.Principal_Component_Analysis(encoded_tweets, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Truncated SVD (alternative of PCA for TF-IDF)

        #  One-hot-encoding, TF-IDF, Bigrams
        if feature_selection == 10:
            encoded_tweets = self.TruncatedSVD(encoded_tweets, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Feature Importance

        #  One-hot-encoding, TF-IDF, Bigrams
        if feature_selection == 11:
            encoded_tweets = self.Feature_Importance(encoded_tweets, y, train_test)

        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print("Final encoded_tweets, after feature selection, shape: ", encoded_tweets.shape)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("Final encoded_tweets, after feature selection, shape: " + str(encoded_tweets.shape) + '\n')

        return encoded_tweets

    ###############################################################################################################################################
    ###############################################################################################################################################

    # Create a dictionary for one hot encoding and encode with one hot encoding
    def one_hot_enc(self, x_enc, train_test):
        encoded_tweets = []
        x_enc = list(x_enc)

        if train_test == 1:  # Train set
            self.onehot_encoder = CountVectorizer(
                analyzer='word',
                tokenizer=self.dummy_fun,
                lowercase=False,
                binary=True
            )

            xenc = []
            for x in x_enc:
                xenc.append(x)

            encoded_tweets = self.onehot_encoder.fit_transform(xenc)
            encoded_tweets = encoded_tweets.toarray()
            vocab = self.onehot_encoder.get_feature_names()
            print(np.array(vocab).shape)

            for i in range(0, len(encoded_tweets[0])):
                if encoded_tweets[0][i] == 1:
                    print("i: ", i, " ", encoded_tweets[0][i], ' = ', vocab[i])

        if train_test == 0:  # Test set
            xenc = []
            for x in x_enc:
                xenc.append(x)
            encoded_tweets = self.onehot_encoder.transform(xenc)
            encoded_tweets = encoded_tweets.toarray()
            vocab = self.onehot_encoder.get_feature_names()

            for i in range(0, len(encoded_tweets[0])):
                if encoded_tweets[0][i] == 1:
                    print("i: ", i, " ", encoded_tweets[0][i], ' = ', vocab[i])

        return encoded_tweets

    ###############################################################################################################################################
    ###############################################################################################################################################

    # TF-IDF
    def tf_idf(self, x_enc, train_test):
        encoded_tweets = []
        if (train_test == 1):  # train
            self.vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', tokenizer=self.dummy_fun,
                                              preprocessor=self.dummy_fun)
            encoded_tweets = self.vectorizer.fit_transform(x_enc)
        if (train_test == 0):  # test
            encoded_tweets = self.vectorizer.transform(x_enc)

        return encoded_tweets

    ###############################################################################################################################################
    ###############################################################################################################################################

    def bigrams_enc(self, x_enc, train_test):
        bigrams = []  # Bi-grams of all tweets

        # Use the pre-processing done above
        for y in range(0, len(x_enc)):
            bigrams.append(list(ngrams(x_enc[y], 2)))

        encoded_tweets = []

        if train_test == 1:  # Train set
            self.onehot_encoder = CountVectorizer(
                analyzer='word',
                tokenizer=self.dummy_fun,
                lowercase=False,
                binary=True
            )

            xenc = []
            for x in bigrams:
                xenc.append(x)

            encoded_tweets = self.onehot_encoder.fit_transform(xenc)
            encoded_tweets = encoded_tweets.toarray()
            vocab = self.onehot_encoder.get_feature_names()

            for i in range(0, len(encoded_tweets[0])):
                if encoded_tweets[0][i] == 1:
                    print("i: ", i, " ", encoded_tweets[0][i], ' = ', vocab[i])

        if train_test == 0:  # Test set
            xenc = []
            for x in bigrams:
                xenc.append(x)
            encoded_tweets = self.onehot_encoder.transform(xenc)
            encoded_tweets = encoded_tweets.toarray()
            vocab = self.onehot_encoder.get_feature_names()

            for i in range(0, len(encoded_tweets[0])):
                if encoded_tweets[0][i] == 1:
                    print("i: ", i, " ", encoded_tweets[0][i], ' = ', vocab[i])

        return encoded_tweets

    ###############################################################################################################################################
    ###############################################################################################################################################

    def Word2Vec_enc(self, x_enc, train_test):
        encoded_tweets = self.labelizeTweets(x_enc, 'TRAIN')

        vector_size = 100

        if train_test == 1:  # Train set
            # sg: CBOW if 0, skip-gram if 1
            # ‘min_count’ is for neglecting infrequent words.
            # negative (int) – If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
            # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
            self.model = Word2Vec(size=vector_size, min_count=0, sg=1)
            self.model.build_vocab([x.words for x in encoded_tweets])
            self.model.train([x.words for x in encoded_tweets], total_examples=len(encoded_tweets), epochs=10)

            self.vectorizer1 = TfidfVectorizer(analyzer=lambda x: x, min_df=7)
            self.vectorizer1.fit_transform([x.words for x in encoded_tweets])

        if train_test == 0:  # Data set
            self.vectorizer1.transform([x.words for x in encoded_tweets])

        tfidf = dict(zip(self.vectorizer1.get_feature_names(), self.vectorizer1.idf_))
        train_vecs_w2v = np.concatenate([self.buildWordVector(self.model, tweet, vector_size, tfidf) for tweet in
                                         map(lambda x: x.words, encoded_tweets)])
        encoded_tweets = scale(train_vecs_w2v)
        print(encoded_tweets)

        return encoded_tweets

    # Used for computing the mean of word2vec and implementing the transform function
    def buildWordVector(self, model, tweet, size, tfidf):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tweet:
            try:
                vec += model[word].reshape((1, size)) * tfidf[word]
                count += 1.
            except KeyError:  # handling the case where the token is not
                # in the corpus. useful for testing.
                continue
        if count != 0:
            vec /= count
        return vec

    def labelizeTweets(self, tweets, label_type):
        LabeledSentence = gensim.models.doc2vec.LabeledSentence

        labelized = []
        for i, v in enumerate(tweets):
            label = '%s_%s' % (label_type, i)
            labelized.append(LabeledSentence(v, [label]))
        return labelized

    ###############################################################################################################################################
    ###############################################################################################################################################

    def Doc2Vec_enc(self, x_enc, train_test):
        encoded_tweets = self.labelizeTweets(x_enc, 'TRAIN')

        vector_size = 100

        if train_test == 1:  # Train set
            # dm: DBOW if 0, distributed-memory if 1
            # window: number of words accounted for each context( if the window size is 3, 3 word in the left neighorhood and 3 word in the right neighborhood are considered)
            self.modeldoc = Doc2Vec(vector_size=vector_size, min_count=0, dm=0)

            self.modeldoc.build_vocab([x for x in encoded_tweets])
            self.modeldoc.train(utils.shuffle([x for x in encoded_tweets]), total_examples=len(encoded_tweets), epochs=10)

            # Get the vectors created for each tweet
            encoded_tweets = np.zeros((len(x_enc), vector_size))
            for i in range(0, len(x_enc)):
                prefix_train_pos = 'TRAIN_' + str(i)
                encoded_tweets[i] = self.modeldoc.docvecs[prefix_train_pos]

        if train_test == 0:  # Test set
            encoded_tweets = np.zeros((len(x_enc), vector_size))
            for i in range(0, len(x_enc)):
                encoded_tweets[i] = self.modeldoc.infer_vector(x_enc[i])

        return encoded_tweets


    ###############################################################################################################################################
    ###############################################################################################################################################


    def GloVe_enc(self, x_enc, train_test):
        encoded_tweets = self.labelizeTweets(x_enc,'TRAIN')  # Different encoding of tweets (One Hot Encoding, TF-IDF, One hot encoding of ngrams)

        if train_test == 1:  # Train set
            if not self.called_once:  # Used to ensure that training-reading the GloVe model is done just once
                self.called_once = True
                gloveFile = self.dir + '\\GloVe_train\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
                print("Loading Glove Model")
                f = open(gloveFile, 'r', encoding="utf8")
                self.glove_model = {}
                for line in f:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    self.glove_model[word] = embedding

            self.vectorizer1 = TfidfVectorizer(analyzer=lambda x: x, min_df=7)
            self.vectorizer1.fit_transform([x.words for x in encoded_tweets])

        if train_test == 0:  # Data set
            self.vectorizer1.transform([x.words for x in encoded_tweets])

        tfidf = dict(zip(self.vectorizer1.get_feature_names(), self.vectorizer1.idf_))
        vector_size = 200  # Dimensions of vectors are stated at the name of the GloVe txt files
        train_vecs_w2v = np.concatenate([self.buildWordVector(self.glove_model, tweet, vector_size, tfidf) for tweet in
                                         map(lambda x: x.words, encoded_tweets)])
        encoded_tweets = scale(train_vecs_w2v)

        return encoded_tweets


    ###############################################################################################################################################
    ###############################################################################################################################################

    # Feature Selection

    ###############################################################################################################################################
    ###############################################################################################################################################


    def Univariate_Selection(self, x, y, train_test):
        # Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
        features = []

        if train_test == 1:  # Train set
            # feature extraction
            self.test = SelectKBest(score_func=chi2, k=100)
            features = self.test.fit_transform(x, y)
            # summarize scores
            numpy.set_printoptions(precision=3)  # Format print to show only 3 decimals of floats

        if train_test == 0:  # Test set
            features = self.test.transform(x)
            # summarize scores
            numpy.set_printoptions(precision=3)  # Format print to show only 3 decimals of floats

        return features


    def Recursive_Feature_Elimination(self, x, y, train_test):
        # Feature Extraction with RFE
        features = []

        if train_test == 1:  # Train set
            # feature extraction
            model = RandomForestClassifier(n_estimators=250,

                                           max_features=7,

                                           max_depth=30,

                                           min_samples_split=2, random_state=0,

                                           n_jobs=-1)
            self.rfe = RFE(model, 100)
            features = self.rfe.fit_transform(x, y)

        if train_test == 0:  # Test set
            features = self.rfe.transform(x)

        return features


    def Principal_Component_Analysis(self, x, train_test):
        # Feature Extraction with PCA
        features = []

        if train_test == 1:  # Train set
            # feature extraction
            self.pca = PCA(n_components=100)
            features = self.pca.fit_transform(x)

        if train_test == 0:  # Test set
            features = self.pca.transform(x)

        return features


    def TruncatedSVD(self, x, train_test):
        # Feature Extraction with TruncatedSVD
        features = []

        if train_test == 1:  # Train set
            # feature extraction
            self.svd = TruncatedSVD(n_components=100)
            features = self.svd.fit_transform(x)

        if train_test == 0:  # Test set
            features = self.svd.transform(x)

        return features


    def Feature_Importance(self, x, y, train_test):
        # Feature Importance with Extra Trees Classifier
        features = []

        if train_test == 1:  # Train set
            # feature extraction

            # Create a random forest classifier with the following Parameters
            self.sfm = RandomForestClassifier(n_estimators=250, max_features=7, max_depth=30)

            self.sfm.fit(x, y)

            # Select features which have higher contribution in the final prediction
            self.models = SelectFromModel(self.sfm, threshold="9*mean")
            self.models.fit(x, y)
            features = self.models.transform(x)

        if train_test == 0:  # Test set
            features = self.models.transform(x)

        return features


    ###############################################################################################################################################
    ###############################################################################################################################################


    ##############################################################################################################################################################

    # Read the training files for task (with emojis)

    # train_A

    ##############################################################################################################################################################

    def readTrain(self):
        # Read the training file
        train_file_A = self.dir + '\\dataset\\train\\imbalanced_training.csv'

        self.train_A = pd.read_csv(train_file_A)
        # Drop the first column of reading file
        self.train_A.drop(['numb'], axis=1, inplace=True)


        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Pre-processing
        self.pre_processing()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    ##############################################################################################################################################################

    # Check if the dataset is imbalanced

    ##############################################################################################################################################################


    def checkImbalance(self):
        # Count the percentage of depressive and non-depressive tweets
        print(self.train_A['label'].value_counts())
        count_0, count_1 = self.train_A['label'].value_counts()
        print(count_1, count_0)
        counter_all = count_0 + count_1
        print('File A without emojis -> Percentage of tweets classified as 0: ' + str((count_0 / counter_all) * 100))
        print('File A without emojis -> Percentage of tweets classified as 1: ' + str(
            (count_1 / counter_all) * 100) + '\n ----------------------------------------')

        # Plot the imbalance with two bars indicating each label
#        color = ['blue', 'orange']
#        self.train_A['label'].value_counts().plot(kind='bar', title='Count (label)', color=color)
#        plt.show()