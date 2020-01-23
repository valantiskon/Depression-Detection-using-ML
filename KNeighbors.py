import Twitter_Depression_Detection # Reads the input and the training sets
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from imblearn.over_sampling import SMOTE

# Import libraries to compute ROC-AUC Curve
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc



def compute_ROC_Curve(tprs, mean_fpr, aucs):
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic - Gaussian Naive Bayes')
    plt.legend(loc="lower right")
    plt.show()







def K_Neighbors(train_A, words_of_tweets, extra_features, feature_selection, encoding, print_file):
    reading = Twitter_Depression_Detection.Reader()  # Import the ClassRead.py file, to get the encoding

    x = np.array(words_of_tweets)
    y = train_A['label']

    # Initialize the roc-auc score running average list
    # Initialize a count to print the number of folds
    # Initialize metrics to print their average
    av_roc = 0.
    count = 0
    precision = 0
    accuracy = 0
    recall = 0
    f1score = 0
    # Above 3 variables are used for ROC-AUC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)


    # Initialize your 10 - cross vailidation
    # Set shuffle equals True to randomize your splits on your training data
    kf = KFold(n_splits=10, random_state=41, shuffle=True)

    # Set up for loop to run for the number of cross vals you defined in your parameter
    for train_index, test_index in kf.split(x):
        count += 1
        print('Fold #: ', count)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write('Fold #: ' + str(count) + '\n')

        # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
        x_train, x_test = reading.get_enc(x[train_index], 1, y[train_index], train_index, extra_features, feature_selection, encoding, print_file), reading.get_enc(x[test_index], 0, y[test_index], test_index, extra_features, feature_selection, encoding, print_file)
        y_train, y_test = y[train_index], y[test_index]

#######################################################################################################################

        # leaf_size: int, optional(default=30)

        # p : integer, optional (default = 2)
        # When p = 1, this is equivalent to using manhattan_distance (l1),
        # and euclidean_distance (l2) for p = 2.
        # For arbitrary p, minkowski_distance (l_p) is used.

        # algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional Algorithm used to compute the nearest neighbors:
        # ‘ball_tree’ will use BallTree
        # ‘kd_tree’ will use KDTree
        # ‘brute’ will use a brute-force search.
        # ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

        # weights : str or callable, optional (default = ‘uniform’) weight function used in prediction. Possible values:
        # ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
        # ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

        scaler = Normalizer()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)


        classifier = KNeighborsClassifier(n_neighbors=40)

        # 'minority': resample only the minority class;
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=10, random_state=0)
        x_train, y_train = oversample.fit_resample(x_train, y_train)

        classifier.fit(x_train, y_train)
        y_pred = classifier.predict(x_test)

#######################################################################################################################

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write("Evaluating model..." + '\n')

        roc = roc_auc_score(y_test, y_pred)

        # Print your ROC-AUC score for your kfold, and the running score average
        print('ROC: ', roc)
        av_roc += roc
        print('Continued Avg: ', av_roc / count)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write('ROC: ' + str(roc) + '\n' + 'Continued Avg: ' + str(av_roc / count) + '\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        '''
        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (count - 1, roc_auc))
        '''
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        y_pred = (y_pred > 0.5)

        # Creating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write(str(cm) + '\n')

        report = classification_report(y_test, y_pred)
        print(report)

        temp_accuracy = accuracy_score(y_test, y_pred)
        temp_precision, temp_recall, temp_f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

        accuracy += temp_accuracy
        precision += temp_precision
        recall += temp_recall
        f1score += temp_f1_score

        print("Accuracy: ", temp_accuracy)
        print("Precision: ", temp_precision)
        print("Recall: ", temp_recall)
        print("F1 score: ", temp_f1_score)

    # Create ROC-AUC curve
#    compute_ROC_Curve(tprs, mean_fpr, aucs)


    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    print('Average ROC:', av_roc / 10)

    with open(print_file, "a") as myfile:  # Write above print into output file
        myfile.write("Average Precision: " + str(precision / 10) + '\n' + "Average Accuracy: " + str(accuracy / 10) + '\n' + "Average Recall: " + str(recall / 10) + '\n' + "Average F1-score: " + str(f1score / 10) + '\n' + 'Average ROC:' + str(av_roc / 10) + '\n')