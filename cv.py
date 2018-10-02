del le_test, le_train, ohe_test, ohe_train, response, test_cats_encoded
del train_cats_encoded, train_ids, test_ids, train_maj, train_min, train_min_sampled
del test, train

from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

kfold = KFold(n_splits=10, shuffle=True, random_state=1986)

array = np.array(train_pca_50, copy=True)
#array = array[:,:50]
#array = train.copy()
results_pca = list()

for tr, te in kfold.split(array):
    train_cv = array[tr]
    test_cv = array[te]
    
    response_train = train_cv[:,-1]
    response_test = test_cv[:,-1]
    
    train_cv = np.delete(train_cv, -1, axis=1)
    test_cv = np.delete(test_cv, -1, axis=1)
    
    iter_results = dict()
    
    log_reg = LogisticRegression(C=0.001, class_weight='balanced')

    log_reg.fit(train_cv, response_train)
    
    log_reg_pred = log_reg.predict(test_cv)
    
    accuracy = accuracy_score(response_test, log_reg_pred)
    iter_results['accuracy'] = accuracy
    
    roc = roc_auc_score(response_test, log_reg_pred)
    iter_results['auc_roc'] = roc
    
    kappa = cohen_kappa_score(response_test, log_reg_pred)
    iter_results['kappa'] = kappa
    
    results_pca.append(iter_results)
    
    
###############################
    
from sklearn.model_selection import KFold    
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

kfold = KFold(n_splits=10, shuffle=True, random_state=1986)

array = np.array(train, copy=True)
#array = train.copy()
results = list()    

from sklearn.naive_bayes import GaussianNB
    
for tr, te in kfold.split(array):
    train_cv = array[tr]
    test_cv = array[te]
    
    response_train = train_cv[:,-1]
    response_test = test_cv[:,-1]
    
    train_cv = np.delete(train_cv, -1, axis=1)
    test_cv = np.delete(test_cv, -1, axis=1)
    
    iter_results = dict()
    
    gau_nb = GaussianNB()

    gau_nb.fit(train_cv, response_train)
    
    gau_nb_pred = gau_nb.predict(test_cv)
    
    accuracy = accuracy_score(response_test, gau_nb_pred)
    iter_results['accuracy'] = accuracy
    
    roc = roc_auc_score(response_test, gau_nb_pred)
    iter_results['auc_roc'] = roc
    
    kappa = cohen_kappa_score(response_test, gau_nb_pred)
    iter_results['kappa'] = kappa
    
    conf_matrix = confusion_matrix(response_test, gau_nb_pred)
    
    results.append(iter_results)
    

###########################################################
    
from sklearn.model_selection import KFold    
from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

kfold = KFold(n_splits=10, shuffle=True, random_state=1986)

array = np.array(train, copy=True)
#array = train.copy()
results_ber_nb = list()    

from sklearn.naive_bayes import BernoulliNB
    
for tr, te in kfold.split(array):
    train_cv = array[tr]
    test_cv = array[te]
    
    response_train = train_cv[:,-1]
    response_test = test_cv[:,-1]
    
    train_cv = np.delete(train_cv, -1, axis=1)
    test_cv = np.delete(test_cv, -1, axis=1)
    
    iter_results = dict()
    
    ber_nb = BernoulliNB(alpha=1.0, binarize=0.1, fit_prior=True)

    ber_nb.fit(train_cv, response_train)
    
    ber_nb_pred = ber_nb.predict(test_cv)
    
    accuracy = accuracy_score(response_test, ber_nb_pred)
    iter_results['accuracy'] = accuracy
    
    roc = roc_auc_score(response_test, ber_nb_pred)
    iter_results['auc_roc'] = roc
    
    kappa = cohen_kappa_score(response_test, ber_nb_pred)
    iter_results['kappa'] = kappa
    
    conf_matrix = confusion_matrix(response_test, ber_nb_pred)
    
    results_ber_nb.append(iter_results)
    

####################################
    
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold    

from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score, confusion_matrix

kfold = KFold(n_splits=2, shuffle=True, random_state=1986)

array = np.array(train, copy=True)
#array = train.copy()
results_rf = list()    

    
for tr, te in kfold.split(array):
    train_cv = array[tr]
    test_cv = array[te]
    
    response_train = train_cv[:,-1]
    response_test = test_cv[:,-1]
    
    train_cv = np.delete(train_cv, -1, axis=1)
    test_cv = np.delete(test_cv, -1, axis=1)
    
    iter_results = dict()
    
    rf = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1,
                                       n_jobs = -1, oob_score=True)
    rf.fit(train_cv, response_train)
    
    rf_pred = rf.predict(test_cv)
    
    accuracy = accuracy_score(response_test, rf_pred)
    iter_results['accuracy'] = accuracy
    
    roc = roc_auc_score(response_test, rf_pred)
    iter_results['auc_roc'] = roc
    
    kappa = cohen_kappa_score(response_test, rf_pred)
    iter_results['kappa'] = kappa
    
    conf_matrix = confusion_matrix(response_test, rf_pred)
    
    results_rf.append(iter_results)