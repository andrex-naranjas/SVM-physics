# simple dy (for testing)
# usage: python3 scripts/simple_dy_training.py DY
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import accuracy_score,precision_score
from common.data_preparation import data_preparation
from common.svm_methods import RBFPrecomputed
import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    sys.exit("Provide data sample name. Try again!")

sample_input = sys.argv[1]
data = data_preparation(drop_class=False)
X_train, Y_train, X_test, Y_test = \
    data.dataset(sample_name=sample_input,
                 sampling=False,split_sample=0.3)

species = "class"

X_train = X_train.drop("z_eta_og", axis=1)
X_train = X_train.drop("z_mass_og", axis=1)
X_train = X_train.drop("z_pt_og", axis=1)
X_train = X_train.drop("reco_Z_masses", axis=1)

X_test = X_test.drop("z_eta_og", axis=1)
X_test = X_test.drop("z_mass_og", axis=1)
X_test = X_test.drop("z_pt_og", axis=1)
X_test = X_test.drop("reco_Z_masses", axis=1)

# print(Y_train.tail)
# print(Y_test.tail)
# input()

X0_train = X_train[X_train['class'] == -1]
X1_train = X_train[X_train['class'] == 1]

size = 1400
random_X0_train = X0_train.sample(n=int(size/2), replace = False)
random_X1_train = X1_train.sample(n=int(size/2), replace = False)
X_balanced = pd.concat([random_X0_train, random_X1_train], ignore_index=True)

print(X_balanced)
print(X_balanced[X_balanced['class'] == -1].shape,  X_balanced[X_balanced['class'] == 1].shape,  "alto ahi")

from sklearn.preprocessing import MinMaxScaler
X_balanced = pd.DataFrame(MinMaxScaler().fit_transform(X_balanced), # not doing this for gaussian distributions!!                                  
                          columns = list(X_balanced.columns))

X_test = pd.DataFrame(MinMaxScaler().fit_transform(X_test), # not doing this for gaussian distributions!!                                  
                          columns = list(X_test.columns))

model = SVC(C=1, gamma=10, kernel='poly', degree=3, shrinking = True, probability = False, tol = 0.001)
model.fit(X_balanced.drop(species, axis=1), X_balanced["class"])
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
Y_pred = model.predict(X_train.drop(species, axis=1))
# auc = roc_auc_score(Y_test, Y_pred_dec)
Y_test_p = X_train["class"]
print(Y_test_p)
prc = precision_score(Y_test_p, Y_pred, pos_label=1)
acc = accuracy_score(Y_test_p, Y_pred)
auc = roc_auc_score(Y_test_p, Y_pred_dec)
print("Testing accuracy linear kernel BALANCED", acc, auc)
print("Testing precisio linear kernel BALANCED", prc)

# input()

X_train = X_balanced
Y_train = X_balanced["class"]
Y_test  = X_test["class"]

model = SVC(C=50, gamma=0.01, kernel='rbf', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
Y_pred = model.predict(X_test.drop(species, axis=1))
# auc = roc_auc_score(Y_test, Y_pred_dec)
prc = precision_score(Y_test, Y_pred, pos_label=1)
acc = accuracy_score(Y_test, Y_pred)
print("Testing accuracy linear kernel", acc)

model = SVC(C=50, gamma=0.01, kernel='rbf', degree=3, shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
Y_pred = model.predict(X_test.drop(species, axis=1))
auc = roc_auc_score(Y_test, Y_pred_dec)
prc = precision_score(Y_test, Y_pred, pos_label=1)
acc = accuracy_score(Y_test, Y_pred)
print("Testing auc radial kernel", auc)
print("Testing precision radial kernel", prc)
print("Testing accuracy radial kernel", acc)

print("before precomp")
input()

# precomputed
test_rbf = RBFPrecomputed([X_train.drop(species, axis=1)])
matrix_test = RBFPrecomputed([X_test.drop(species, axis=1), X_train.drop(species, axis=1)])
# precomputed kernel, explicit calculation
matrix_ep = test_rbf.compute()
model_ep = SVC(C=50, gamma=0.01, kernel="precomputed")
model_ep.fit(matrix_ep, Y_train)
matrix_test_ep = matrix_test.compute()
Y_pred_ep = model_ep.predict(matrix_test_ep)
acc_ep = accuracy_score(Y_test, Y_pred_ep)
prc_ep = precision_score(Y_test, Y_pred_ep)
print("Testing accuracy precom kernel", acc_ep)


from common.svm_methods import PolyPrecomputed
kernel_comp = PolyPrecomputed([X_train.drop(species, axis=1)], gamma=1, deg=2, coef=0)
kernel_test_comp = PolyPrecomputed([X_test.drop(species, axis=1), X_train.drop(species, axis=1)], gamma=1, deg=2, coef=0)
kernel_comp = kernel_comp.compute()
kernel_test_comp = kernel_test_comp.compute()
model_ep = SVC(C=50, gamma=0.1, kernel="precomputed")
model_ep.fit(kernel_comp, Y_train)
Y_pred_ep = model_ep.predict(kernel_test_comp)
acc_ep = accuracy_score(Y_test, Y_pred_ep)
prc_ep = precision_score(Y_test, Y_pred_ep)
print("Testing accuracy precom kernel", acc_ep)
print("Testing accuracy precom kernel", prc_ep)


from common.boosted_svm import BoostedSVM
# from common.common_methods import roc_curve_adaboost,plot_roc_curve

svm_boost = BoostedSVM(C=100, gammaEnd=100, myKernel='poly', myDegree=3, myCoef0=+1,
                       Diversity=False, early_stop=True, debug=False)
svm_boost.fit(X_train.drop(species, axis=1), Y_train)
y_preda = svm_boost.predict(X_test.drop(species, axis=1))
y_thresholds = svm_boost.decision_thresholds(X_test.drop(species, axis=1), glob_dec=True)
# TPR, FPR = roc_curve_adaboost(y_thresholds, Y_test)
prec = precision_score(Y_test, y_preda)
acc = accuracy_score(Y_test, y_preda)
# area = auc(FPR,TPR)
nWeaks = len(svm_boost.alphas)
print(len(svm_boost.alphas), len(svm_boost.weak_svm), "how many alphas we have BOOSTING")
print(acc, prec, "acc and prc BOOSTED")


# def make_unitary(matrix):
#     # Perform singular value decomposition (SVD)
#     U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
#     # Construct a unitary matrix from the SVD
#     unitary_matrix = np.dot(U, Vh)
#     return unitary_matrix

# kernel_comp = make_unitary(kernel_comp_1.compute())
# kernel_comp_og = kernel_comp_1.compute()
# kernel_test_comp = make_unitary(kernel_test_comp.compute())
# print("testing...")
# print(kernel_comp.shape, kernel_comp_og.shape)
# print(type(kernel_comp), type(kernel_comp_og))
# y0_index = X_train[X_train['class'] == -1].index
# y1_index = X_train[X_train['class'] == 1].index
# random_y0 = np.random.choice(y0_index, int(size/2), replace = False)
# random_y1 = np.random.choice(y1_index, int(size/2), replace = False)
# input()
# print(X_balanced.duplicated().tolist())
# input()
