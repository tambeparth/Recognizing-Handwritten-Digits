Digits Recognition with scikit-learn
This project demonstrates Optical Recognition of Handwritten Digits using various machine learning classifiers.

Dataset Information
The dataset is from sklearn's digits dataset:

Number of Instances: 1797

Number of Attributes: 64 (8x8 image of integer pixels in range 0..16)

No missing attribute values

Source: E. Alpaydin (July 1998)

Link: UCI Digits Dataset

The dataset contains images of handwritten digits (0-9), preprocessed into 8x8 pixel bitmaps with integer values representing pixel intensity.

Dataset Preview
Example of a digit image data (array form):

python
Copy
Edit
digits.images[0]
array([[ 0.,  0.,  5., 13.,  9.,  1.,  0.,  0.],
       [ 0.,  0., 13., 15., 10., 15.,  5.,  0.],
       [ 0.,  3., 15.,  2.,  0., 11.,  8.,  0.],
       [ 0.,  4., 12.,  0.,  0.,  8.,  8.,  0.],
       [ 0.,  5.,  8.,  0.,  0.,  9.,  8.,  0.],
       [ 0.,  4., 11.,  0.,  1., 12.,  7.,  0.],
       [ 0.,  2., 14.,  5., 10., 12.,  0.,  0.],
       [ 0.,  0.,  6., 13., 10.,  0.,  0.,  0.]])
Visualization of Some Digits
Example code to visualize digits using matplotlib:

python
Copy
Edit
import matplotlib.pyplot as plt

def show_digit(index):
    plt.imshow(digits.images[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('The digit is: ' + str(digits.target[index]))
    plt.show()

show_digit(7)
Splitting Dataset
Training set: first 1791 samples

Validation set: last 6 samples

Example visualization of validation digits:

python
Copy
Edit
plt.subplot(321)
plt.imshow(digits.images[1791], cmap=plt.cm.gray_r, interpolation='nearest')
# similarly for digits 1792 to 1796 with plt.subplot(322) to plt.subplot(326)
Classifiers Used
Support Vector Classifier (SVC)
python
Copy
Edit
from sklearn import svm

svc = svm.SVC(gamma=0.001, C=100.)
svc.fit(main_data[:1790], targets[:1790])

predictions = svc.predict(main_data[1791:])
print(predictions, targets[1791:])
Accuracy: 100% on validation data (6 samples)

Decision Tree Classifier
python
Copy
Edit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dt = DecisionTreeClassifier(criterion='gini')
dt.fit(main_data[:1600], targets[:1600])

prediction2 = dt.predict(main_data[1601:])
print(confusion_matrix(targets[1601:], prediction2))
print(accuracy_score(targets[1601:], prediction2))
Accuracy: ~78% on test data (197 samples)

Random Forest Classifier
python
Copy
Edit
from sklearn.ensemble import RandomForestClassifier

rc = RandomForestClassifier(n_estimators=150)
rc.fit(main_data[:1500], targets[:1500])

prediction3 = rc.predict(main_data[1501:])
print(accuracy_score(targets[1501:], prediction3))
Accuracy: ~92.5% on test data (297 samples)

Conclusion
Data quality and quantity are crucial for model performance.

Support Vector Classifier achieved the highest accuracy (~100%) with most training data.

Random Forest also performed well (~92.5%) even with fewer training samples.

Decision Tree performed relatively lower (~78%), suggesting the need for better tuning or more data.

