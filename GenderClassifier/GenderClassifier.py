import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Correlation Between Gender, Height
# And Shoe Size In Australian University Students
# y-height, x-shoe size (american)
# females: y=2.9x + 143.7
# males:  y=3.6x + 142.9


def _replace_item(x):
    if x == "male":
        return 0
    else:
        return 1


# Order: [height, weight, shoe size]
samples = [ [181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
            [190, 90, 47], [175, 64, 39],
            [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

# Gender
labels = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
          'female', 'male', 'male']

validation_set = [[185, 80, 44,], [210, 150, 53], [173, 63, 40], [150, 50, 39]]
reference_labels = ['male', 'male', 'female', 'female']
reference_labels_binary = np.asarray([0, 0, 1, 1])


# Classifiers
clfs = [DecisionTreeClassifier(), GaussianProcessClassifier(), RandomForestClassifier(), AdaBoostClassifier(), GaussianNB(), SVC()]
for clf in clfs:
    clf = clf.fit(samples, labels)
    prediction = clf.predict(validation_set)
    # Prediction_binary = np.asarray(map(lambda x: 0 if x == ['male'] else 1, prediction))
    print(list(prediction))
    prediction_binary = np.asarray(list(map(int, list(map(lambda x: '0' if x == 'male' else '1', list(prediction))))))
        
    print(clf)
    print(prediction)
    
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(prediction_binary == 1, reference_labels_binary == 1))
 
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(prediction_binary == 0, reference_labels_binary == 0))
 
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(prediction_binary == 1, reference_labels_binary == 0))
 
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(prediction_binary == 0, reference_labels_binary == 1))
 
    SENS = TP / (TP + FN)
    SPEC = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    print ('TP: %i, FP: %i, TN: %i, FN: %i \n SENS: %f SPEC: %f  ACC: %f \n' % (TP, FP, TN, FN, SENS, SPEC, ACC))