# simple dy (for testing)
# usage: python3 scripts/simple_dy_training.py DY
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import auc,roc_auc_score
from common.data_preparation import data_preparation


if len(sys.argv) != 2:
    sys.exit("Provide data sample name. Try again!")

sample_input = sys.argv[1]
data = data_preparation(drop_class=False)
X_train, Y_train, X_test, Y_test = \
    data.dataset(sample_name=sample_input,
                 sampling=False,split_sample=0.3)


species = "class"

model = SVC(C=50, gamma=0.01, kernel='linear', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
auc = roc_auc_score(Y_test, Y_pred_dec)
print("Testing accuracy linear kernel", auc)

model = SVC(C=50, gamma=0.01, kernel='rbf', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
auc = roc_auc_score(Y_test, Y_pred_dec)
print("Testing accuracy radial kernel", auc)

