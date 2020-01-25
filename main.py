print('imp1')
import Twitter_Depression_Detection # Reads the input and the training sets
from Twitter_Depression_Detection import Reader
print('imp2')
import SVM # Implements SVM classification
print('imp2.1')
import nltk
nltk.download('punkt')
'''
import NaiveBayes # Implements Naive Bayes Classification
'''
print('imp3')
import KNeighbors # Implements KNeighbors classification
print('imp4')
import VotingEnsembles # Implements VotingEnsembles classification
print('imp5')
#import LSTM # Implements  LSTM classification
print('imp6')
#import Conv1D # Implements Conv1D classification
print('imp7')
import os.path


##############################################################################################################################################################
##############################################################################################################################################################

                                                                    # Main

##############################################################################################################################################################
##############################################################################################################################################################

print('start')
reading = Reader() # Import the Twitter_Depression_Detection.py file, that reads the input and the training sets # Import the Twitter_Depression_Detection.py file, that reads the input and the training sets
print('set cwd')
dir = os.getcwd() # Gets the current working directory


##############################################################################################################################################################

# Read input and training file, check if the dataset is imbalanced

##############################################################################################################################################################

print("Read train")
#reading.switch()
reading.readTrain()
print('check for imbalance')
#reading.checkImbalance()


##############################################################################################################################################################

# Call all algorithms with different combinations of feature selection and encoding

##############################################################################################################################################################







##############################################################################################################################################################

# Implementation of STACKING method

##############################################################################################################################################################

print("Start bayes")
model = SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '/SVM/PCA + TF-IDF.txt')
reading2 = Reader()
print('@@@'+str(len(reading2.words_of_tweets)))
reading2.words_of_tweets = []
reading2.trainA = None
reading2.readTrain2()
SVM.svm_func2(model, reading2.train_A, reading2.words_of_tweets, reading2.extra_features, 9, 1, dir + '/SVM/PCA + TF-IDF.txt')
print('DONE FILE 10')

'''

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\SVM\\PCA + TF-IDF.txt')
print('DONE FILE 10')

'''
##############################################################################################################################################################

# Call SVM classification for Irony Detection

##############################################################################################################################################################


'''
SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\SVM\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\SVM\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\SVM\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\SVM\\SVD + TF-IDF.txt')
print('DONE FILE 4')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\SVM\\SVD + One-Hot.txt')
print('DONE FILE 5')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\SVM\\SVD + Bigrams.txt')
print('DONE FILE 6')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\SVM\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\SVM\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\SVM\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\SVM\\PCA + TF-IDF.txt')
print('DONE FILE 10')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\SVM\\PCA + One-Hot.txt')
print('DONE FILE 11')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\SVM\\PCA + Bigrams.txt')
print('DONE FILE 12')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\SVM\\word2vec.txt')
print('DONE FILE 13')


#SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\SVM\\GloVe.txt')
#print('DONE FILE 14')


SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\SVM\\doc2vec.txt')
print('DONE FILE 15')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\SVM\\TF-IDF.txt')
print('DONE FILE 16')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\SVM\\One-Hot.txt')
print('DONE FILE 17')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\SVM\\Bigrams.txt')
print('DONE FILE 18')
'''


'''
SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\SVM\\RFE + TF-IDF.txt')
print('DONE FILE 19')
'''

'''

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\SVM\\RFE + One-Hot.txt')
print('DONE FILE 20')

SVM.svm_func(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\SVM\\RFE + Bigrams.txt')
print('DONE FILE 21')
'''



##############################################################################################################################################################

# Call Naive Bayes classification for Irony Detection

##############################################################################################################################################################


'''
NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\Bayes\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')
'''
NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\SVM\PCA + TF-IDF.txt')
print('DONE FILE 1')
'''
'''
NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\Bayes\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\Bayes\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\Bayes\\SVD + TF-IDF.txt')
print('DONE FILE 4')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\Bayes\\SVD + One-Hot.txt')
print('DONE FILE 5')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\Bayes\\SVD + Bigrams.txt')
print('DONE FILE 6')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\Bayes\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\Bayes\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\Bayes\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\Bayes\\PCA + TF-IDF.txt')
print('DONE FILE 10')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\Bayes\\PCA + One-Hot.txt')
print('DONE FILE 11')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\Bayes\\PCA + Bigrams.txt')
print('DONE FILE 12')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\Bayes\\word2vec.txt')
print('DONE FILE 13')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\Bayes\\GloVe.txt')
print('DONE FILE 14')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\Bayes\\doc2vec.txt')
print('DONE FILE 15')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\Bayes\\TF-IDF.txt')
print('DONE FILE 16')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\Bayes\\One-Hot.txt')
print('DONE FILE 17')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\Bayes\\Bigrams.txt')
print('DONE FILE 18')
'''

'''

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\Bayes\\RFE + TF-IDF.txt')
print('DONE FILE 19')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\Bayes\\RFE + One-Hot.txt')
print('DONE FILE 20')

NaiveBayes.Bayes(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\Bayes\\RFE + Bigrams.txt')
print('DONE FILE 21')
'''





##############################################################################################################################################################

# Call LSTM classification for Irony Detection

##############################################################################################################################################################

'''
LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\LSTM\\word2vec.txt')
LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\LSTM\\doc2vec.txt')

LSTM.lstm(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\LSTM\\GloVe.txt')

'''



##############################################################################################################################################################

# Call Conv1D classification for Irony Detection

##############################################################################################################################################################

'''
Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\Conv1D\\word2vec.txt')
Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\Conv1D\\doc2vec.txt')

Conv1D.conv1d_class(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\Conv1D\\GloVe.txt')
'''


##############################################################################################################################################################

# Call K-Neighbors to predict irony and evaluate the outcome

##############################################################################################################################################################


KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\KNeighbors\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\KNeighbors\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\KNeighbors\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\KNeighbors\\SVD + TF-IDF.txt')
print('DONE FILE 4')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\KNeighbors\\SVD + One-Hot.txt')
print('DONE FILE 5')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\KNeighbors\\SVD + Bigrams.txt')
print('DONE FILE 6')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\KNeighbors\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\KNeighbors\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\KNeighbors\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\KNeighbors\\PCA + TF-IDF.txt')
print('DONE FILE 10')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\KNeighbors\\PCA + One-Hot.txt')
print('DONE FILE 11')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\KNeighbors\\PCA + Bigrams.txt')
print('DONE FILE 12')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\KNeighbors\\word2vec.txt')
print('DONE FILE 13')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\KNeighbors\\GloVe.txt')
print('DONE FILE 14')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\KNeighbors\\doc2vec.txt')
print('DONE FILE 15')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\KNeighbors\\TF-IDF.txt')
print('DONE FILE 16')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\KNeighbors\\One-Hot.txt')
print('DONE FILE 17')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\KNeighbors\\Bigrams.txt')
print('DONE FILE 18')
'''

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\KNeighbors\\RFE + TF-IDF.txt')
print('DONE FILE 19')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\KNeighbors\\RFE + One-Hot.txt')
print('DONE FILE 20')

KNeighbors.K_Neighbors(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\KNeighbors\\RFE + Bigrams.txt')
print('DONE FILE 21')

'''



##############################################################################################################################################################

# Call Voting Ensembles, using various algorithms, to predict irony and evaluate the outcome

##############################################################################################################################################################


VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 1, dir + '\\VotingEnsembles\\Univariate Selection + TF-IDF.txt')
print('DONE FILE 1')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 2, dir + '\\VotingEnsembles\\Univariate Selection + One-Hot.txt')
print('DONE FILE 2')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 7, 3, dir + '\\VotingEnsembles\\Univariate Selection + Bigrams.txt')
print('DONE FILE 3')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 1, dir + '\\VotingEnsembles\\SVD + TF-IDF.txt')
print('DONE FILE 4')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 2, dir + '\\VotingEnsembles\\SVD + One-Hot.txt')
print('DONE FILE 5')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 10, 3, dir + '\\VotingEnsembles\\SVD + Bigrams.txt')
print('DONE FILE 6')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 1, dir + '\\VotingEnsembles\\Feature Improtance + TF-IDF.txt')
print('DONE FILE 7')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 2, dir + '\\VotingEnsembles\\Feature Improtance + One-Hot.txt')
print('DONE FILE 8')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 11, 3, dir + '\\VotingEnsembles\\Feature Improtance + Bigrams.txt')
print('DONE FILE 9')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 1, dir + '\\VotingEnsembles\\PCA + TF-IDF.txt')
print('DONE FILE 10')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 2, dir + '\\VotingEnsembles\\PCA + One-Hot.txt')
print('DONE FILE 11')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 9, 3, dir + '\\VotingEnsembles\\PCA + Bigrams.txt')
print('DONE FILE 12')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 4, dir + '\\VotingEnsembles\\word2vec.txt')
print('DONE FILE 13')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 6, dir + '\\VotingEnsembles\\GloVe.txt')
print('DONE FILE 14')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 5, dir + '\\VotingEnsembles\\doc2vec.txt')
print('DONE FILE 15')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 1, dir + '\\VotingEnsembles\\TF-IDF.txt')
print('DONE FILE 16')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 2, dir + '\\VotingEnsembles\\One-Hot.txt')
print('DONE FILE 17')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 0, 3, dir + '\\VotingEnsembles\\Bigrams.txt')
print('DONE FILE 18')
'''

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 1, dir + '\\VotingEnsembles\\RFE + TF-IDF.txt')
print('DONE FILE 19')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 2, dir + '\\VotingEnsembles\\RFE + One-Hot.txt')
print('DONE FILE 20')

VotingEnsembles.Voting_Ensembles(reading.train_A, reading.words_of_tweets, reading.extra_features, 8, 3, dir + '\\VotingEnsembles\\RFE + Bigrams.txt')
print('DONE FILE 21')
'''


'''
