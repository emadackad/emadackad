import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import io
from scipy.sparse import coo_matrix, csr_matrix

    #Read Data
X = io.mmread('matrix.mtx')
counts = torch.tensor(X.transpose().tocsr(), dtype=torch.float)
metadata = pd.read_csv('metadata.csv')

le = LabelEncoder().fit(metadata['Annotations'])
labels = le.transform(metadata['Annotations'])

    #Log Transform and Scale
features_log = torch.log1p(counts)    
scaler = StandardScaler(with_mean=False)
features_log_std = scaler.fit_transform(features_log.detach().numpy())
  #Train Test Split
x_train, x_test, y_train, y_test = train_test_split(features_log_std, labels, test_size = 0.2)
  #Initialize Model
model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.001, max_iter=20, verbose=0)
model.fit(x_train, y_train)
#print("Slopes: ", model.coef_)
#print("intercept: ", model.intercept_)
    
    #Calculate predicted output probabilities
predicted_soft = model.predict_proba(x_train)
pred = model.predict(x_train)
print("Training Score: ", model.score(x_train, y_train))
total_accuracy = accuracy_score(y_train, pred)*100
print("Training Accuracy: ",total_accuracy)

    #Test Predictions Accuracy and Model Score
test_pred_soft = model.predict_proba(x_test)
test_pred = model.predict(x_test)
print("Test Score: ", model.score(x_test, y_test))
test_accuracy = accuracy_score(y_test,test_pred)*100
print("Test Accuracy: ",test_accuracy)

    #Confusion Matrix
confusion_mat = confusion_matrix(le.inverse_transform(y_test), le.inverse_transform(test_pred))
cm_display = ConfusionMatrixDisplay(confusion_matrix = confusion_mat, display_labels = metadata['Annotations'].unique())

  #Pot Confusion Matrix
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(metadata['Annotations'].unique()))
plt.xticks(tick_marks, metadata['Annotations'].unique(), rotation=45, ha="right")  # Adjust rotation angle and horizontal alignment
plt.yticks(tick_marks, metadata['Annotations'].unique())
plt.tight_layout()
plt.xlabel('Model Label')
plt.ylabel('Original Label')
