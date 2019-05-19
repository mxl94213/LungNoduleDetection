from sklearn.metrics import roc_curve, auc, confusion_matrix
import csv
import matplotlib.pyplot as plt

def Performance_evaluate():
    f = open('../submission_files/predicted_regions_labels.csv','r')
    csvReader = list(csv.reader(f))
    del csvReader[0]
    actual = []
    predicted = []
    for item in csvReader:
        actual.append(int(eval(item[3])))
        predicted.append(int(eval(item[4])))
    fpr, tpr, thresholds = roc_curve(actual, predicted, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("The false positive rate is: ",fpr)
    print("The true positive rate is:", tpr)
    print(thresholds)
    print("The auc is:", roc_auc)
    count = 0
    for i in range(len(predicted)):
        if actual[i] == predicted[i]:
            count += 1
    accuracy = count / len(predicted)
    print('The accuracy is: ', accuracy)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.axis('equal')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    confs_matrix = confusion_matrix(actual, predicted)

    TN = confs_matrix[0][0]
    FP = confs_matrix[0][1]
    FN = confs_matrix[1][0]
    TP = confs_matrix[1][1]

    precision = TP * 1.0 / (TP + FP)
    recall = TP * 1.0 / (TP + FN)
    specificity = TN * 1.0 / (TN + FP)

    print("The sensitivity is:", recall)
    print("The precision is:", precision)
    print("The specificity is:", specificity)
    plt.show()

    f.close()

