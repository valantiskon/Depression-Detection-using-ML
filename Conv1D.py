import Twitter_Depression_Detection  # Reads the input and the training sets
import numpy as np
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv1D
from keras.layers import GlobalAveragePooling1D
from keras.optimizers import RMSprop
from keras import callbacks



def conv1d_class(train_A, words_of_tweets, extra_features, feature_selection, encoding, print_file):
    reading = Twitter_Depression_Detection.Reader()  # Import the ClassRead.py file, to get the encoding


    # fix random seed for reproducibility
    numpy.random.seed(7)


    x = np.array(words_of_tweets)
    y = train_A['label']

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize the roc-auc score running average list
    # Initialize a count to print the number of folds
    # Initialize metrics to print their average
    av_roc = 0.
    count = 0
    precision = 0
    accuracy = 0
    recall = 0
    f1score = 0

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Initialize your 10 - cross vailidation
    # Set shuffle equals True to randomize your splits on your training data
    kf = KFold(n_splits=10, random_state=41, shuffle=True)

    # Set up for loop to run for the number of cross vals you defined in your parameter
    for train_index, test_index in kf.split(x):
        count += 1
        print('Fold #: ', count)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write('Fold #: ' + str(count) + '\n')

        # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
        x_train, x_test = reading.get_enc(x[train_index], 1, y[train_index], train_index, extra_features,
                                          feature_selection, encoding, print_file), reading.get_enc(x[test_index], 0,
                                                                                                    y[test_index],
                                                                                                    test_index,
                                                                                                    extra_features,
                                                                                                    feature_selection,
                                                                                                    encoding,
                                                                                                    print_file)
        y_train, y_test = y[train_index], y[test_index]

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Initializing Neural Network
        classifier = Sequential()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 'minority': resample only the minority class;
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=10, random_state=0)
        x_train, y_train = oversample.fit_resample(x_train, y_train)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        print(x_train.shape[0], ' ', x_train.shape[1])
        print(x_test.shape[0], ' ', x_test.shape[1])
        x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

        classifier.add(Dense(20, kernel_initializer='glorot_uniform', activation='softsign', kernel_constraint=maxnorm(2), input_shape=(1, x_train.shape[2])))

        classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        classifier.add(Dropout(0.2))

        classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

        classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

        classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))

        classifier.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        classifier.add(Dropout(0.2))

        classifier.add(GlobalAveragePooling1D())

        classifier.add(Dense(500, kernel_initializer='glorot_uniform', activation='softsign', kernel_constraint=maxnorm(2)))

        # Adding the output layer with 1 output
        classifier.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))

        optimizer = RMSprop(lr=0.001)

        # Compiling Neural Network
        classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')


        # Fitting our model
        classifier.fit(x_train, y_train, batch_size=20, epochs=50)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write("Evaluating model..." + '\n')

        test_preds = classifier.predict_proba(x_test, verbose=0)

        roc = roc_auc_score(y_test, test_preds)
        scores = classifier.evaluate(x_test, y_test)
        print(scores)

        # Print your model summary
        print(classifier.summary())

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write('Scores: ' + str(scores) + '\n' + 'Classifier summary: ' + str(
                classifier.summary()) + '\n' + 'ROC: ' + str(roc) + '\n' + 'Continued Avg: ' + str(
                av_roc / count) + '\n')

        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Predicting the Test set results
        y_pred = classifier.predict(x_test)
        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        with open(print_file, "a") as myfile:  # Write above print into output file
            myfile.write(str(cm) + '\n')

        temp_accuracy = accuracy_score(y_test, y_pred)
        temp_precision, temp_recall, temp_f1_score, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                        average='binary')

        accuracy += temp_accuracy
        precision += temp_precision
        recall += temp_recall
        f1score += temp_f1_score

        print("Accuracy: ", temp_accuracy)
        print("Precision: ", temp_precision)
        print("Recall: ", temp_recall)
        print("F1 score: ", temp_f1_score)

    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    print('Average ROC:', av_roc / 10)

    with open(print_file, "a") as myfile:  # Write above print into output file
        myfile.write("Average Precision: " + str(precision / 10) + '\n' + "Average Accuracy: " + str(
            accuracy / 10) + '\n' + "Average Recall: " + str(recall / 10) + '\n' + "Average F1-score: " + str(
            f1score / 10) + '\n' + 'Average ROC:' + str(av_roc / 10) + '\n')