import Final_ClassRead # Reads the input and the training sets
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# Import packages to visualize the Learning Curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

    print("STEP 1")

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    print("STEP 2")

    # Create means and standard deviations of training set scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    print("STEP 3")

    # Create means and standard deviations of test set scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    print("STEP 4")

    # Draw bands
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    print("STEP 5")

    # Draw lines
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")

    return plt



def Bayes(train_A, train_words_of_tweets, train_extra_features, test_words_of_tweets, test_A, test_extra_features):
    reading = Final_ClassRead.Reader()  # Import the Final_ClassRead.py file, to get the encoding

    x_train = np.array(train_words_of_tweets)
    y_train = train_A['label']

    x_test = np.array(test_words_of_tweets)
    y_test = test_A['label']

    # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
    x_train, x_test = reading.get_enc(x_train, 1, y_train, train_extra_features), reading.get_enc(x_test, 0, y_test, test_extra_features)


#######################################################################################################################

    model = GaussianNB()

    # Fit Gaussian Naive Bayes according to x, y
    # Make a prediction using the Naive Bayes Model
    model.fit(x_train,
              y_train)  # x : array-like, shape (n_samples, n_features)   Training vectors, where n_samples is the number of samples and n_features is the number of features.
                        # y : array-like, shape (n_samples,)   Target values.

#######################################################################################################################
    '''
    # Learning Curve

    title = "Learning Curves (Gaussian Naive Bayes)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # Use only the training data set (cannot use whole data set cause it is not encoded)
    plot_learning_curve(model, title, x_train, y_train, ylim=(0.5, 1.01), cv=cv, n_jobs=1)

    plt.show()
    '''
#######################################################################################################################

    y_pred = model.predict(x_test)

#######################################################################################################################

    # Your model is fit. Time to predict our output and test our training data
    print("Evaluating model...")
    roc = roc_auc_score(y_test, y_pred)

    # Print your ROC-AUC score for your kfold, and the running score average
    print('ROC: ', roc)

#######################################################################################################################

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