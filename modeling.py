from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
from util.utils import printDivider

def create_model_data(negative_samples, positive_samples, label_dict={ "negative": "cohort1", "positive": "cohort3" }, verbose=True):
    Xneg = negative_samples.loc[:, negative_samples.columns != 'PatientKey'].values
    Xpos = positive_samples.loc[:, positive_samples.columns != 'PatientKey'].values
    X = np.vstack([Xpos, Xneg])
    y = np.zeros(X.shape[0],)
    y[:Xpos.shape[0]] = 1
    if verbose:
        print(label_dict)
        print(label_dict["negative"], ":", Xneg.shape)
        print(label_dict["positive"], ":", Xpos.shape)
        print("X:", X.shape)
        print("y", y.shape)
    
    return (X, y)

def train_logistic(X_train, y_train):
    lr = linear_model.LogisticRegression()
    lr.fit(X_train, y_train)
    return lr

def eval_model(model, X_test, y_test):
    ypred_discrete = model.predict(X_test)
    ypred = model.predict_proba(X_test)[:, 1]

    cm = metrics.confusion_matrix(y_test, ypred_discrete)
    auc = metrics.roc_auc_score(y_test, ypred)
    f1 = metrics.f1_score(y_test, ypred_discrete)    
    return (cm, auc, f1)

def train_random_forest(X_train, y_train):
    best_auc = 0
    best_n_estimators = 0
    for i in range(10, 110, 10):
        rf_clf = RandomForestClassifier(n_estimators=i)
        # use cross validation to find the best n_estimators
        # auc = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='f1').mean()
        auc = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
        if auc > best_auc:
            best_auc = auc
            best_n_estimators = i
    # train model with best estimator
    best_model = RandomForestClassifier(n_estimators=best_n_estimators)
    best_model.fit(X_train, y_train)
    
    return (best_model, best_n_estimators)

def show_feature_importance(rf_clf, features):
    forest = rf_clf
    importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #             axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(len(features)):
        print("%d. feature %s (%f)" % (f + 1, features[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    # plt.bar(range(len(features)), importances[indices],
    #     color="r", yerr=std[indices], align="center")
    plt.bar(range(len(features)), importances[indices],
        color="r", align="center")
    # plt.bar(range(len(features)), importances[indices],
    #     color="r", yerr=std[indices], align="center")
    plt.bar(range(len(features)), importances[indices],
        color="r", align="center")
    plt.xticks(range(len(features)), indices)
    plt.xlim([-1, len(features)])
    plt.show()

def train_gradient_boosting(X_train, y_train):
    best_auc = 0
    best_n_estimators = 0
    for i in range(10, 110, 10):
        gb_clf = GradientBoostingClassifier(n_estimators=i)
        # use cross validation to find the best n_estimators
        # auc = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='f1').mean()
        auc = cross_val_score(gb_clf, X_train, y_train, cv=5, scoring='roc_auc').mean()
        if auc > best_auc:
            best_auc = auc
            best_n_estimators = i
    # train model with best estimator
    best_model = GradientBoostingClassifier(n_estimators=best_n_estimators)
    best_model.fit(X_train, y_train)
    
    return (best_model, best_n_estimators)

def build_models(
        cohort13_data,
        cohort2_data,
        test_size=0.2,
        random_state=123,
        label_dict={ "negative": "cohort13", "positive": "cohort2" }
    ):
    print("start modeling")
    print()
    features = cohort13_data.columns[1:]
    print("using features")
    print(features)

    # prepare data
    X, y = create_model_data(cohort13_data, cohort2_data, label_dict=label_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print('X_train:', X_train.shape)
    print('y_train:', X_train.shape)
    printDivider()

    # linear regression
    lr = train_logistic(X_train, y_train)
    cm, auc, f1, ap = eval_model(lr, X_test, y_test)
    print("Linear Regression")
    print()
    print(cm)
    print('auc:', auc)
    print('f1:', f1)
    printDivider()

    # random forest
    rf_clf, rf_n_estimators = train_random_forest(X_train, y_train)
    cm, auc, f1, ap = eval_model(rf_clf, X_test, y_test)
    print("Random Forest")
    print()
    print('confusion matrix')
    print(cm)
    print('n_estimators:', rf_n_estimators)
    print('auc:', auc)
    print('f1:', f1)
    show_feature_importance(rf_clf, features)
    printDivider()

    # gradient boosting
    gb_clf, gb_n_estimators = train_gradient_boosting(X_train, y_train)
    cm, auc, f1, ap = eval_model(gb_clf, X_test, y_test)
    print("Gradient Boosting")
    print()
    print('confusion matrix')
    print(cm)
    print('n_estimators:', gb_n_estimators)
    print('auc:', auc)
    print('f1:', f1)
    show_feature_importance(gb_clf, features)
    printDivider()

    return (lr, rf_clf, gb_clf)

def cross_validate_model(cohort13, cohort2, clf):
    X, y = create_model_data(cohort13, cohort2)
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    
def build_12vs3_models(cohort12_data,cohort3_data,test_size=0.2,random_state=123,label_dict={ "negative": "cohort12", "positive": "cohort3" }):
    print("start modeling")
    print()
    features = cohort12_data.columns[1:]
    print("using features")
    print(features)

    # prepare data
    X, y = create_model_data(cohort12_data, cohort3_data, label_dict=label_dict)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    print('X_train:', X_train.shape)
    print('y_train:', X_train.shape)
    printDivider()

    # linear regression
    lr = train_logistic(X_train, y_train)
    cm, auc, f1 = eval_model(lr, X_test, y_test)
    print("Linear Regression")
    print()
    print(cm)
    print('auc:', auc)
    print('f1:', f1)
    printDivider()

    # random forest
    rf_clf, rf_n_estimators = train_random_forest(X_train, y_train)
    cm, auc, f1 = eval_model(rf_clf, X_test, y_test)
    print("Random Forest")
    print()
    print('confusion matrix')
    print(cm)
    print('n_estimators:', rf_n_estimators)
    print('auc:', auc)
    print('f1:', f1)
    show_feature_importance(rf_clf, features)
    printDivider()

    # gradient boosting
    gb_clf, gb_n_estimators = train_gradient_boosting(X_train, y_train)
    cm, auc, f1 = eval_model(gb_clf, X_test, y_test)
    print("Gradient Boosting")
    print()
    print('confusion matrix')
    print(cm)
    print('n_estimators:', gb_n_estimators)
    print('auc:', auc)
    print('f1:', f1)
    show_feature_importance(gb_clf, features)
    printDivider()

    return (lr, rf_clf, gb_clf)