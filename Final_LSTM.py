import Final_ClassRead # Reads the input and the training sets
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.constraints import maxnorm
from keras.optimizers import RMSprop
from keras import callbacks

def lstm(train_A, train_words_of_tweets, train_extra_features, test_words_of_tweets, test_A, test_extra_features):
    reading = Final_ClassRead.Reader()  # Import the Final_ClassRead.py file, to get the encoding

    x_train = np.array(train_words_of_tweets)
    y_train = train_A['label']

    x_test = np.array(test_words_of_tweets)
    y_test = test_A['label']

    # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
    x_train, x_test = reading.get_enc(x_train, 1, y_train, train_extra_features), reading.get_enc(x_test, 0, y_test, test_extra_features)


#######################################################################################################################

    # Initializing Neural Network
    classifier = Sequential()


    print(x_train.shape[0], ' ', x_train.shape[1])
    print(x_test.shape[0], ' ', x_test.shape[1])
    x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

    print("second dimension (feature dimension): ", x_train.shape)

    classifier.add(LSTM(10, return_sequences=True, activation='softplus', input_shape=(1, x_train.shape[2])))
    classifier.add(Dropout(0.2))
    classifier.add(LSTM(20, activation='softplus'))
    classifier.add(Dropout(0.2))

    classifier.add(Dense(500, kernel_initializer='glorot_uniform', activation='softsign', kernel_constraint=maxnorm(2)))

    # Adding the output layer with 1 output
    classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

    optimizer = RMSprop(lr=0.001)

    # Compiling Neural Network
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

    # Fitting our model
    classifier.fit(x_train, y_train, batch_size=20, epochs=50)

#######################################################################################################################

    # Your model is fit. Time to predict our output and test our training data
    print("Evaluating model...")
    test_preds = classifier.predict_proba(x_test, verbose=0)

    roc = roc_auc_score(y_test, test_preds)
    scores = classifier.evaluate(x_test, y_test)
    print(scores)

    # Print your model summary
    print(classifier.summary())

    # Print your ROC-AUC score for your kfold, and the running score average
    print('ROC: ', roc)

#######################################################################################################################

    # Predicting the Test set results
    y_pred = classifier.predict(x_test)
    y_pred = (y_pred > 0.5)

    # Creating the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Print average of metrics
    print("Precision: ", precision)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("F1-score: ", f1score)