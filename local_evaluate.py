from csv import DictReader
import numpy as np
from sklearn import metrics


predict_file = 'submission_April_23_01:16:55.csv'
label_file = 'WE_data/test_local_label.csv'

# label
print ("load label ...")
labels = []
for row in DictReader(open(label_file)):
    labels.append(float(row['Click']))
print ("load label finish.")

print ("load prediction ...")
# predict
preds = []
for row in DictReader(open(predict_file)):
    preds.append(float(row['Prediction']))
print ("load prediction finish.")

y = np.array(labels)
pred = np.array(preds)
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
print("AUC==>" + str(metrics.auc(fpr, tpr)))

