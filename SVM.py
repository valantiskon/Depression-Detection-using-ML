import Twitter_Depression_Detection # Reads the input and the training sets
import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import naive_bayes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
print('2.2')
import seaborn as sns
print('2.3')
#import tensorflow
from imblearn.over_sampling import SMOTE
print('2.4')
# Import packages to visualize the ROC-AUC Curve
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn import metrics




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
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()



# Visualize the dataset siplified to 2 dimension (with dimension reduction)
def visualize_data(X, y, label):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )

    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()



def svm_func(train_A, words_of_tweets, extra_features, feature_selection, encoding, print_file):
    reading = Twitter_Depression_Detection.Reader()  # Import the Twitter_Depression_Detection.py file, to get the encoding
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(words_of_tweets)
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
    # Below 3 variables are used for ROC-AUC curve
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

        # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object
        # For very large C, the margin is hard, and points cannot lie in it. For smaller C, the margin is softer, and can grow to encompass some points.
        # gamma: Higher the value of gamma, will try to exact fit the training data set i.e.generalization error and cause over-fitting problem.
        model = naive_bayes.GaussianNB()


#######################################################################################################################
        # Feature Scaling
        minMaxScaler = MinMaxScaler(feature_range=(0, 1))
        # Get points and discard classification labels
        #x_train = minMaxScaler.fit_transform(x_train)
        #x_test = minMaxScaler.transform(x_test)
#######################################################################################################################
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=10, random_state=0)
        model.fit(x_train, y_train)
        return model
#######################################################################################################################
        # Visualization of normal and oversampled data

        '''visualize_data(x_train, y_train, "Normal Dataset")'''

        # 'minority': resample only the minority class;
        x_train, y_train = oversample.fit_resample(x_train, y_train)
        '''visualize_data(x_train, y_train, "Oversampled Dataset")'''

#######################################################################################################################

        model.score(x_train, y_train)
        # Predict Output
        y_pred = model.predict(x_test)
        #return model
#######################################################################################################################

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write("Evaluating model..." + '\n')

        #roc = roc_auc_score(y_test, y_pred)

        # Print your ROC-AUC score for your kfold, and the running score average
        #print('ROC: ', roc)
        #av_roc += roc
        #print('Continued Avg: ', av_roc / count)

        #with open(print_file, "a") as myfile: # Write above print into output file
            #myfile.write('ROC: ' + str(Continued Avg: ' + str(av_roc / count) + '\n')

        #y_pred = (y_pred > 0.5)

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

        '''
        # Creating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)


        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write(str(cm) + '\n')
        '''
        print(y_pred)
        temp_accuracy = accuracy_score(y_test, y_pred)
        temp_precision, temp_recall, temp_f1_score, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                        average='macro')

        accuracy += temp_accuracy
        precision += temp_precision
        recall += temp_recall
        f1score += temp_f1_score

        print("Accuracy: ", temp_accuracy)
        print("Precision: ", temp_precision)
        print("Recall: ", temp_recall)
        print("F1 score: ", temp_f1_score)

        print(metrics.classification_report(y_test,y_pred))


    # Create ROC-AUC curve
#    compute_ROC_Curve(tprs, mean_fpr, aucs)


##########################################################################################################################


    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    #print('Average ROC:', av_roc / 10)

    with open(print_file, "a") as myfile:  # Write above print into output file
        myfile.write("Average Precision: " + str(precision / 10) + '\n' + "Average Accuracy: " + str(accuracy / 10) + '\n' + "Average Recall: " + str(recall / 10) + '\n' + "Average F1-score: " + str(f1score / 10) + '\n' + 'Average ROC:' + str(av_roc / 10) + '\n')





def svm_func2(model2, train_A, words_of_tweets, extra_features, feature_selection, encoding, print_file):
    reading = Twitter_Depression_Detection.Reader()  # Import the Twitter_Depression_Detection.py file, to get the encoding

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
    # Below 3 variables are used for ROC-AUC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    # Initialize your 10 - cross vailidation
    # Set shuffle equals True to randomize your splits on your training data
    kf = KFold(n_splits=10, random_state=7, shuffle=True)
    print(x.size)
    # Set up for loop to run for the number of cross vals you defined in your parameter
    for train_index, test_index in kf.split(x):
        count += 1
        print('Fold #: ', count)
        print(train_index)
        print(test_index)


        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write('Fold #: ' + str(count) + '\n')

        # This indexs your train and test data for your cross validation and sorts them in random order, since we used shuffle equals True
        x_train, x_test = reading.get_enc(x[train_index], 1, y[train_index], train_index, extra_features, feature_selection, encoding, print_file), reading.get_enc(x[test_index], 0, y[test_index], test_index, extra_features, feature_selection, encoding, print_file)
        y_train, y_test = y[train_index], y[test_index]
        x_train = model2.predict_proba(x_train)
        x_test = model2.predict_proba(x_test)
        # Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
        # Create SVM classification object
        # For very large C, the margin is hard, and points cannot lie in it. For smaller C, the margin is softer, and can grow to encompass some points.
        # gamma: Higher the value of gamma, will try to exact fit the training data set i.e.generalization error and cause over-fitting problem.
        model = svm.SVC(kernel='rbf', C=100, gamma=0.1)


#######################################################################################################################
        # Feature Scaling
        minMaxScaler = MinMaxScaler(feature_range=(0, 1))
        # Get points and discard classification labels
        x_train = minMaxScaler.fit_transform(x_train)
        x_test = minMaxScaler.transform(x_test)
#######################################################################################################################

        model.fit(x_train, y_train)


        model.score(x_train, y_train)
        # Predict Output
        y_pred = model.predict(x_test)
        #return model
#######################################################################################################################

        # Your model is fit. Time to predict our output and test our training data
        print("Evaluating model...")

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write("Evaluating model..." + '\n')

        #roc = roc_auc_score(y_test, y_pred)

        # Print your ROC-AUC score for your kfold, and the running score average
        #print('ROC: ', roc)
        #av_roc += roc
        #print('Continued Avg: ', av_roc / count)

        #with open(print_file, "a") as myfile: # Write above print into output file
            #myfile.write('ROC: ' + str(Continued Avg: ' + str(av_roc / count) + '\n')

        #y_pred = (y_pred > 0.5)

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

        '''
        # Creating the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        with open(print_file, "a") as myfile: # Write above print into output file
            myfile.write(str(cm) + '\n')
        '''
        print(y_pred)
        temp_accuracy = accuracy_score(y_test, y_pred)
        temp_precision, temp_recall, temp_f1_score, _ = precision_recall_fscore_support(y_test, y_pred,
                                                                                        average='macro')

        accuracy += temp_accuracy
        precision += temp_precision
        recall += temp_recall
        f1score += temp_f1_score

        print("Accuracy: ", temp_accuracy)
        print("Precision: ", temp_precision)
        print("Recall: ", temp_recall)
        print("F1 score: ", temp_f1_score)


# =============================================================================

    # Plot HEATMAP

# =============================================================================

        '''plt.title('SVM - Confusion Matrix '

                  '\n[Accuracy = %0.2f, Recall = %0.2f, Precision = %0.2f, F1-Score = %0.2f] '
                  '\nTrue Positive = %d, False Positive = %d '
                  '\nFalse Negative = %d, True Negative = %d]' % (
            temp_accuracy * 100, temp_recall * 100, temp_precision * 100, temp_f1_score * 100, cm[0][0], cm[0][1], cm[1][0], cm[1][1]))

        sns.heatmap(cm, cmap='Oranges',  # Color of heatmap
                    annot=True, fmt="d",
                    # Enables values inside the heatmap boxes and sets that are integer values with fmt="d"
                    cbar=False,  # Delete the heat bar (shows the numbers corresponding to colors)
                    xticklabels=["depression", "no depression"], yticklabels=["depression", "no depression"]
                    # Name the x and y value labels
                    ).tick_params(left=False, bottom=False)  # Used to delete dash from name values of axis x and y

        # Fix a bug where heatmap top and bottom boxes are cut off
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values

        plt.xlabel('True output')
        plt.ylabel('Predicted output')
        plt.show()
        '''
# =============================================================================


    # Create ROC-AUC curve
#    compute_ROC_Curve(tprs, mean_fpr, aucs)


##########################################################################################################################


    # Print average of metrics
    print("Average Precision: ", precision / 10)
    print("Average Accuracy: ", accuracy / 10)
    print("Average Recall: ", recall / 10)
    print("Average F1-score: ", f1score / 10)

    # Print your final average ROC-AUC score and organize your models predictions in a dataframe
    #print('Average ROC:', av_roc / 10)

    with open(print_file, "a") as myfile:  # Write above print into output file
        myfile.write("Average Precision: " + str(precision / 10) + '\n' + "Average Accuracy: " + str(accuracy / 10) + '\n' + "Average Recall: " + str(recall / 10) + '\n' + "Average F1-score: " + str(f1score / 10) + '\n' + 'Average ROC:' + str(av_roc / 10) + '\n')
