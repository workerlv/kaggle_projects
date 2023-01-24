from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
# from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
import numpy as np
import time
import json
import pandas as pd


class GridSearchAV:
    def __init__(self, X_train, X_test, y_train, y_test, final_df_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.final_df_test = final_df_test
        self.valid_scores = pd.DataFrame()
        self.clf_best_params = {}
        self.LR_grid = {'penalty': ['l2'],
                   'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                   'max_iter': [50, 100, 150]}

        self.KNN_grid = {'n_neighbors': [3, 5, 7, 9],
                    'p': [1, 2]}

        self.SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']}

        self.RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
                   'max_depth': [4, 6, 8, 10, 12]}

        self.boosted_grid = {'n_estimators': [50, 100, 150, 200],
                        'max_depth': [4, 8, 12],
                        'learning_rate': [0.05, 0.1, 0.15]}

        self.NB_grid = {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7]}

        # Dictionary of all grids
        self.grid = {
            "LogisticRegression": self.LR_grid,
            "KNN": self.KNN_grid,
            # "SVC" : self.SVC_grid,
            "RandomForest": self.RF_grid,
            "XGBoost": self.boosted_grid,
            "LGBM": self.boosted_grid,
            "CatBoost": self.boosted_grid,
            "NaiveBayes": self.NB_grid
        }

        self.classifiers = {}
        self.best_classifiers = {}
        self.best_params = {}
        self.preds = np.zeros(len(self.final_df_test))

        with open('SpaceShipTitanic/best_params_from_grid_search.json', 'r') as f:
            read_json_data = json.load(f)

        self.clf_best_params = dict(read_json_data)

    def add_logistic_regression(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["LogisticRegression"] = LogisticRegression(**self.clf_best_params["LogisticRegression"],
                                                                             random_state=0)
        else:
            self.classifiers["LogisticRegression"] = LogisticRegression(random_state=0)

    def add_KNN(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["KNN"] = KNeighborsClassifier(**self.clf_best_params["KNN"])
        else:
            self.classifiers["KNN"] = KNeighborsClassifier()

    def add_random_forest(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["RandomForest"] = RandomForestClassifier(**self.clf_best_params["RandomForest"],
                                                                           random_state=0)
        else:
            self.classifiers["RandomForest"] = RandomForestClassifier(random_state=0)

    def add_XGBoost(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["XGBoost"] = XGBClassifier(**self.clf_best_params["XGBoost"],
                                                             random_state=0,
                                                             use_label_encoder=False,
                                                             eval_metric='logloss')
        else:
            self.classifiers["XGBoost"] = XGBClassifier(random_state=0,
                                                        use_label_encoder=False,
                                                        eval_metric='logloss')

    # def add_LGBM(self, cross_validation=False):
    #     if cross_validation:
    #
    #     else
    #     self.classifiers["LGBM"] = LGBMClassifier(random_state=0)

    def add_catboost(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["CatBoost"] = CatBoostClassifier(**self.clf_best_params["CatBoost"],
                                                                   random_state=0,
                                                                   verbose=False)
        else:
            self.classifiers["CatBoost"] = CatBoostClassifier(random_state=0,
                                                              verbose=False)

    def add_naivebayes(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["NaiveBayes"] = GaussianNB(**self.clf_best_params["NaiveBayes"])
        else:
            self.classifiers["NaiveBayes"] = GaussianNB()

    def add_svc(self, cross_validation=False):
        if cross_validation:
            self.best_classifiers["SVC"] = SVC(**self.clf_best_params["SVC"],
                                               random_state=0,
                                               probability=True)
        else:
            self.classifiers["SVC"] = SVC(random_state=0,
                                          probability=True)

    def start_grid_search(self):
        i = 0
        self.clf_best_params = self.classifiers.copy()
        self.valid_scores = pd.DataFrame({'Classifer': self.classifiers.keys(),
                                     'Validation accuracy': np.zeros(len(self.classifiers)),
                                     'Training time': np.zeros(len(self.classifiers))})

        for key, classifier in self.classifiers.items():
            start = time.time()
            clf = GridSearchCV(estimator=classifier, param_grid=self.grid[key], n_jobs=-1, cv=None)

            # Train and score
            clf.fit(self.X_train, self.y_train)
            self.valid_scores.iloc[i, 1] = clf.score(self.X_test, self.y_test)

            # Save trained model
            self.clf_best_params[key] = clf.best_params_

            # Print iteration and training time
            stop = time.time()
            self.valid_scores.iloc[i, 2] = np.round((stop - start) / 60, 2)

            print('Model:', key)
            print('Training time (mins):', self.valid_scores.iloc[i, 2])
            print('')
            i += 1

        save_best_params_to_json = json.dumps(self.clf_best_params)
        converted_dict = json.loads(save_best_params_to_json)

        with open("SpaceShipTitanic/best_params_from_grid_search.json", "w") as f:
            json.dump(converted_dict, f, indent=6)

    def get_valid_scores(self):
        return self.valid_scores

    def get_clf_best_params(self):
        return self.clf_best_params

    def start_cross_validation(self, folds):
        FOLDS = folds

        classifiers_for_sum = ""
        scores_for_sum = ""
        score_sum = 0
        calsif_count = 0

        for key, classifier in self.best_classifiers.items():
            start = time.time()

            cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=0)

            score = 0
            for fold, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                # Get training and validation sets
                X_train_, X_test_ = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_train_, y_test_ = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]

                # Train model
                clf = classifier
                clf.fit(X_train_, y_train_)

                # Make predictions and measure accuracy
                self.preds += clf.predict_proba(self.final_df_test)[:, 1]
                score += clf.score(X_test_, y_test_)

            # Average accuracy
            score = score / FOLDS

            classifiers_for_sum += f"{key} / "
            scores_for_sum += f"{score.round(2)} / "
            score_sum += score
            calsif_count += 1
            # Stop timer
            stop = time.time()

            # Print accuracy and time
            print(f"Model: {key}")
            print(f"Average validation accuracy: {np.round(100 * score, 2)}")
            print(f"Training time (mins): {np.round((stop - start) / 60, 2)}")
            print(f"")

        # Ensemble predictions
        self.preds = self.preds / (FOLDS * len(self.classifiers))

        return classifiers_for_sum, scores_for_sum, score_sum, calsif_count

    def get_predictions(self):
        print("PREDS:")
        print(self.preds)
        return self.preds






