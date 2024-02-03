# simple dy (for testing)
# usage: python3 scripts/simple_dy_training.py DY
import sys
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import auc,roc_auc_score
from sklearn.metrics import accuracy_score,precision_score
from common.data_preparation import data_preparation
from common.svm_methods import RBFPrecomputed
import numpy as np

def make_unitary(matrix):
    # Perform singular value decomposition (SVD)
    U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Construct a unitary matrix from the SVD
    unitary_matrix = np.dot(U, Vh)

    return unitary_matrix




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


model = SVC(C=50, gamma=0.01, kernel='linear', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
Y_pred = model.predict(X_test.drop(species, axis=1))
# auc = roc_auc_score(Y_test, Y_pred_dec)
acc = accuracy_score(Y_test, Y_pred)
print("Testing accuracy linear kernel", acc)

model = SVC(C=50, gamma=0.01, kernel='rbf', shrinking = True, probability = False, tol = 0.001)
model.fit(X_train.drop(species, axis=1), Y_train)
Y_pred_dec = model.decision_function(X_test.drop(species, axis=1))
Y_pred = model.predict(X_test.drop(species, axis=1))
#auc = roc_auc_score(Y_test, Y_pred_dec)
acc = accuracy_score(Y_test, Y_pred)
print("Testing accuracy radial kernel", acc)

# # precomputed
# test_rbf = RBFPrecomputed([X_train.drop(species, axis=1)])
# matrix_test = RBFPrecomputed([X_test.drop(species, axis=1), X_train.drop(species, axis=1)])
# # precomputed kernel, explicit calculation
# matrix_ep = test_rbf.compute()
# model_ep = SVC(C=50, gamma=0.01, kernel="precomputed")
# model_ep.fit(matrix_ep, Y_train)
# matrix_test_ep = matrix_test.compute()
# Y_pred_ep = model_ep.predict(matrix_test_ep)
# acc_ep = accuracy_score(Y_test, Y_pred_ep)
# prc_ep = precision_score(Y_test, Y_pred_ep)
# print("Testing accuracy precom kernel", acc_ep)


from common.svm_methods import PolyPrecomputed
kernel_comp_1 = PolyPrecomputed([X_train.drop(species, axis=1)], gamma=1, deg=3, coef=0)
kernel_test_comp = PolyPrecomputed([X_test.drop(species, axis=1), X_train.drop(species, axis=1)], gamma=1, deg=3, coef=0)
kernel_comp = make_unitary(kernel_comp_1.compute())
kernel_comp_og = kernel_comp_1.compute()
kernel_test_comp = make_unitary(kernel_test_comp.compute())
print("testing...")
print(kernel_comp.shape, kernel_comp_og.shape)
print(type(kernel_comp), type(kernel_comp_og))

input()
model_ep = SVC(C=50, gamma=0.01, kernel="precomputed")
print("model..")
model_ep.fit(kernel_comp, Y_train)
print("trained")
Y_pred_ep = model_ep.predict(kernel_test_comp)
acc_ep = accuracy_score(Y_test, Y_pred_ep)
prc_ep = precision_score(Y_test, Y_pred_ep)
print("Testing accuracy precom kernel", acc_ep)
