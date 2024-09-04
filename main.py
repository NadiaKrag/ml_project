##################
#  Introduction  #
##################
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from boosting.adaboost import AdaBoost
from mda.mda import MixtureDiscriminant
from linear.logreg import LogisticRegression

from model_selection.splitter import KFold
from model_selection.hyper_opt import GridSearch
from metrics.metrics import accuracy,recall,precision,f1_score

import pickle as pk

X = np.loadtxt(fname='data/olive_train.csv',skiprows=1,delimiter=',',usecols=(1,2))
y = np.loadtxt(fname='data/olive_train.csv',skiprows=1,delimiter=',',usecols=(0))
X_test = np.loadtxt(fname='data/olive_test.csv',skiprows=1,delimiter=',',usecols=(1,2))
y_test = np.loadtxt(fname='data/olive_test.csv',skiprows=1,delimiter=',',usecols=(0))

def get_number_of_combos(params):
    n_combos = 1
    for val in params.values():
        n_combos *= len(val)
    return n_combos

#########################
#  Grid-search example  #
#########################
# MDA
f_name = 'MDA_5repeats'
path = 'results/' + f_name

components = list()
for i in range(1,12):
    for j in range(1,12):
        components.append(np.array([i,j]))

params = {"Ks":components}

print(f_name, get_number_of_combos(params))

gs = GridSearch(MixtureDiscriminant(),params,k_splits=5,n_repeats=5,verbose=1)
gs.fit(X,y)

info = gs.give_info()
combos = gs.give_combos()

topten = reversed(np.argsort(info["total_oof_acc"])[-10:])
with open(path + '.txt','w') as f:
    f.write("Top ten highest out-of-fold accuracies (with std):\n")
    for i in topten:
        f.write('{:2d}, {:.7f}, {:.7f}, {}\n'.format(i,info["total_oof_acc"][i],info["oof_std"][i],combos[i]))

pk.dump(file=open(path + '.pk','wb'),obj=gs)

# AdaBoost
f_name = 'adaboost'
path = 'results/' + f_name

components = list(range(1,200))

params = {"m_estimators":components}

print(f_name, get_number_of_combos(params))

gs = GridSearch(AdaBoost(),params,k_splits=5,n_repeats=1,verbose=1)
gs.fit(X,y)

info = gs.give_info()
combos = gs.give_combos()

topten = reversed(np.argsort(info["total_oof_acc"])[-10:])
with open(path + '.txt','w') as f:
    f.write("Top ten highest out-of-fold accuracies (with std):\n")
    for i in topten:
        f.write('{:2d}, {:.7f}, {:.7f}, {}\n'.format(i,info["total_oof_acc"][i],info["oof_std"][i],combos[i]))

pk.dump(file=open(path + '.pk','wb'),obj=gs)

# Logistic regression
f_name = 'logreg_3'
path = 'results/' + f_name

Ms = []
for i in range(2,12):
    for j in range(2,12):
        Ms.append(np.array([i,j]))
ss = [i / i**2 for i in range(20,0,-1)]
ss += list(np.linspace(0.05,1,25))
params = {"s":ss, "M_array":Ms, "basis_name":['rbf','sbf']}
#params = { "s":ss , "basis_name":['rbf','sbf'], 'M_array': np.array(([9,7],[4,8],[7, 6],[7, 4],[4, 6]))}

print(f_name, get_number_of_combos(params))

gs = GridSearch(LogisticRegression(),params,k_splits=5,n_repeats=1,verbose=1)
gs.fit(X,y)

info = gs.give_info()
combos = gs.give_combos()

pk.dump(file=open(path + '.pk','wb'),obj=gs)
topten = reversed(np.argsort(info["total_oof_acc"])[-10:])
print(topten)
with open(path + '.txt','w') as f:
    f.write("Top ten highest out-of-fold accuracies (with std):\n")
    for i in topten:
        f.write('{:2d}, {:.7f}, {:.7f}, {}\n'.format(i,info["total_oof_acc"][i],info["oof_std"][i],combos[i]))

pk.dump(file=open(path + '.pk','wb'),obj=gs)

######################
#  Cross-validation  #
######################

# MDA
print('MDA')
folds = KFold(k_splits=500)
n_repeats = 5

oof_pred = np.zeros(X.shape[0])
val_accs = list()

total_oof_acc = 0
total_oof_recall = np.zeros(2)
total_oof_precision = np.zeros(2)
total_oof_f1 = np.zeros(2)
for i in range(n_repeats):
    for n_fold,(idx_train,idx_val) in enumerate(folds.split(X,y)):
        scaler = Standardizer()
        X_train = scaler.fit_transform(X[idx_train])
        X_val = scaler.transform(X[idx_val])


        clf = MixtureDiscriminant(max_iterations=1000,Ks=np.array([6,8]))
        clf.fit(X_train,y[idx_train])
        oof_pred[idx_val] = clf.predict(X_val)

        train_acc = accuracy(clf.predict(X_train),y[idx_train])
        val_acc = accuracy(oof_pred[idx_val],y[idx_val])
        val_accs.append(val_acc)

    total_oof_acc += accuracy(oof_pred,y)
    total_oof_recall += np.array(recall(oof_pred,y))
    total_oof_precision += np.array(precision(oof_pred,y))
    total_oof_f1 += np.array(f1_score(oof_pred,y))

pk.dump(clf,open('results/MDA_CV_5repeats.pk','wb'))
print("Validation accuracy: {}".format(total_oof_acc/n_repeats))
print("95 percent fall within: {:.5f}".format(2*np.std(val_accs)))
print("Variance: {:.5f}".format(np.var(val_accs)))
print()
print("Validation recall: {:.3f}, {:.3f}".format(*(total_oof_recall/n_repeats)))
print("Validation precision: {:.3f}, {:.3f}".format(*(total_oof_precision/n_repeats)))
print("Validation F1-score: {:.3f}, {:.3f}".format(*(total_oof_f1 / n_repeats)))
print("-"*20)

# AdaBoost
folds = KFold(k_splits=500)

oof_pred = np.zeros(X.shape[0])
val_accs = list()

for n_fold,(idx_train,idx_val) in enumerate(folds.split(X,y)):
    scaler = Standardizer()
    X_train = scaler.fit_transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])


    clf = AdaBoost(m_estimators=80)
    clf.fit(X_train,y[idx_train])
    oof_pred[idx_val] = clf.predict(X_val)

    train_acc = accuracy(clf.predict(X_train),y[idx_train])
    val_acc = accuracy(oof_pred[idx_val],y[idx_val])
    val_accs.append(val_acc)

    # print("train acc: {:.3f} \t val acc: {:.3f}".format(train_acc,val_acc))

pk.dump(clf,open('results/adaboost_CV.pk','wb'))
print('AdaBoost')
print("Validation accuracy: {}".format(accuracy(oof_pred,y)))
print("95 percent fall within: {:.5f}".format(2*np.std(val_accs)))
print("Variance: {:.5f}".format(np.var(val_accs)))
print()
print("Validation recall: {:.3f}, {:.3f}".format(recall(oof_pred,y)[0],recall(oof_pred,y)[1]))
print("Validation precision: {:.3f}, {:.3f}".format(precision(oof_pred,y)[0],precision(oof_pred,y)[1]))
print("Validation F1-score: {:.3f}, {:.3f}".format(f1_score(oof_pred,y)[0],f1_score(oof_pred,y)[1]))
print("-"*20)

# Logistic Regression
folds = KFold(k_splits=500)

oof_pred = np.zeros(X.shape[0])
val_accs = list()

for n_fold,(idx_train,idx_val) in enumerate(folds.split(X,y)):
    scaler = Standardizer()
    X_train = scaler.fit_transform(X[idx_train])
    X_val = scaler.transform(X[idx_val])


    clf = LogisticRegression(basis_name="sbf",s=0.247917,M_array=[9,7])
    clf.fit(X_train,y[idx_train])
    oof_pred[idx_val] = clf.predict(X_val)

    train_acc = accuracy(clf.predict(X_train),y[idx_train])
    val_acc = accuracy(oof_pred[idx_val],y[idx_val])
    val_accs.append(val_acc)

    # print("train acc: {:.3f} \t val acc: {:.3f}".format(train_acc,val_acc))

pk.dump(clf,open('results/logreg_CV.pk','wb'))
print('Logistic Regression')
print("Validation accuracy: {}".format(accuracy(oof_pred,y)))
print("95 percent fall within: {:.5f}".format(2*np.std(val_accs)))
print("Variance: {:.5f}".format(np.var(val_accs)))
print()
print("Validation recall: {:.3f}, {:.3f}".format(recall(oof_pred,y)[0],recall(oof_pred,y)[1]))
print("Validation precision: {:.3f}, {:.3f}".format(precision(oof_pred,y)[0],precision(oof_pred,y)[1]))
print("Validation F1-score: {:.3f}, {:.3f}".format(f1_score(oof_pred,y)[0],f1_score(oof_pred,y)[1]))
print("-"*20)

######################
#  Final prediction  #
######################

# MDA
scaler = Standardizer()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

n_repeats = 5
test_accuracy = 0
for i in range(n_repeats):
    clf = MixtureDiscriminant(max_iterations=1000,Ks=np.array([6,8]))
    clf.fit(X_train,y)

    test_pred = clf.predict(X_test)
    test_accuracy += accuracy(test_pred,y_test)
pk.dump(clf,open('results/MDA_test.pk','wb'))
print("MDA")
print("Test accuracy: {}".format(test_accuracy / n_repeats))
print("-----------------------")

clf = LogisticRegression(basis_name="sbf",s=0.247917,M_array=[9,7])
clf.fit(X_train,y)
pk.dump(clf,open('results/logreg_test.pk','wb'))
test_pred = clf.predict(X_test)
test_accuracy = accuracy(test_pred,y_test)
print("Logistic Regression")
print("Test accuracy: {}".format(test_accuracy))
print("-----------------------")

clf = AdaBoost(m_estimators=80)
clf.fit(X_train,y)
pk.dump(clf,open('results/adaboost_test.pk','wb'))
test_pred = clf.predict(X_test)
test_accuracy = accuracy(test_pred,y_test)
print("AdaBoost")
print("Test accuracy: {}".format(test_accuracy))
print("-----------------------")
