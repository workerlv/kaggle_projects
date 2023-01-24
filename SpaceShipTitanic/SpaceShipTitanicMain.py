import StreamlitAV as ST
import SpaceShipTitanic as SST
import PrepareForCalcualtionAV as PFC
import GridSearchCatAV as GS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import os
import seaborn as sns
sns.set()

# from lightgbm import LGBMClassifier
# import lazypredict
# from lazypredict.Supervised import LazyClassifier

DEPENDENT_VARIABLE_NAME = "Transported"

raw_data_from_file = pd.read_csv("SpaceShipTitanic/train.csv")
raw_test_data_from_file = pd.read_csv("SpaceShipTitanic/test.csv")

sst = SST.SpaceShipTitanic(train_data=raw_data_from_file, test_data=raw_test_data_from_file)

ST.write_h1("Spaceship titanic")
ST.write_h2("RAW DATA")
ST.sidebar("Columns")

selected_numeric_values = ST.selected_filter_values("Numeric columns", sst.data_numeric)
selected_categorical_values = ST.selected_filter_values("Categorical columns", sst.data_categorical)
selected_binary_values = ST.selected_filter_values("Binary columns", sst.data_binary)

grid_search = ST.st.sidebar.checkbox("Grid search")
all_saved_dataframes = ST.st.sidebar.checkbox("Saved dataframes")

with ST.st.expander("raw data from tables"):

    # DATA ABOUT REMOVED COLUMNS
    ST.display_dataframe(sst.raw_summary(), "raw summary")
    ST.filter_func(sst.data_numeric, "Numeric data", selected_numeric_values)
    ST.filter_func(sst.data_categorical, "Categorical data", selected_categorical_values)
    ST.filter_func(sst.data_binary, "Binary data", selected_binary_values)

    if ST.st.button('raw_data correlation'):
        fig_corr_matrix = px.imshow(sst.raw_data.corr(), text_auto=True, aspect="auto", width=800, height=800)
        ST.st.plotly_chart(fig_corr_matrix)

with ST.st.expander("Dependent variable"):

    ST.display_dataframe(sst.missing_dependent_values(), "dependent variable summary")

    fig1, ax1 = plt.subplots(figsize=(15, 5))
    ax1.pie(sst.dependent_variable.value_counts(),
            explode=(0.1, 0),
            labels=sst.dependent_variable.value_counts().index,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    ST.st.pyplot(fig1)

    ST.display_dataframe(sst.dependent_variable, "Dependent variable")

with ST.st.expander("Numeric data"):
    ST.display_dataframe(sst.is_null_table_numeric(), "is null values")

    ST.filter_func(sst.data_numeric.describe().applymap(lambda x: f"{x:0.2f}"), "Numeric data describe",
                   selected_numeric_values)

    ST.filter_func(sst.data_numeric_test.describe().applymap(lambda x: f"{x:0.2f}"), "Numeric test data describe",
                   selected_numeric_values)

with ST.st.expander("Categorical data"):
    ST.display_dataframe(sst.is_null_table_categorical(), "is null values")

    ST.filter_func(sst.data_categorical.describe(), "Categorical data describe", selected_categorical_values)
    ST.filter_func(sst.data_categorical.describe(), "Categorical test data describe", selected_categorical_values)

    number_of_unique_values = ST.st.number_input('Insert a number', value=5)
    if ST.st.button('Show unique values'):
        ST.display_dataframe(sst.get_unique_categorical_values(number_of_unique_values), "unique cat values")

with ST.st.expander("Binary data"):
    ST.display_dataframe(sst.is_null_table_binary(), "is null values")

    ST.filter_func(sst.data_binary.describe(), "Binary data describe", selected_binary_values)
    ST.filter_func(sst.data_binary.describe(), "Binary data test data describe", selected_binary_values)

ST.write_h2("Feature engineering")

with ST.st.expander("Feature engineering"):

    # passanger id
    split_passengerId = sst.data_categorical["PassengerId"].str.split("_", n=1, expand=True)
    split_passengerId_test = sst.data_categorical_test["PassengerId"].str.split("_", n=1, expand=True)

    sst.data_numeric["Passenger_within_group"] = split_passengerId[1]
    sst.data_numeric_test["Passenger_within_group"] = split_passengerId_test[1]

    ST.display_dataframe(split_passengerId, "Passenger id split")

    ST.display_dataframe(split_passengerId.nunique(), "additional")
    split_passengerId[DEPENDENT_VARIABLE_NAME] = sst.dependent_variable
    split_passengerId.drop([0], axis=1, inplace=True)
    ST.display_dataframe(split_passengerId, "new df")

    fig_t, ax_t = plt.subplots(figsize=(15, 5))
    sns.histplot(data=split_passengerId, x=1, bins=8, kde=True, hue=DEPENDENT_VARIABLE_NAME)
    ST.st.pyplot(fig_t)

ST.write_h2("PLOTS")

add_numeric_hist_plots = ST.st.checkbox(f"Numeric histplots")

if add_numeric_hist_plots:

    with ST.st.expander("Numeric histplot"):

        num_hist_df = pd.concat([sst.data_numeric, sst.dependent_variable], axis=1)

        for i, var_name in enumerate(selected_numeric_values):
            bins = ST.st.slider(f"{var_name} bins", 0, 100, 30)

            need_limit = ST.st.checkbox(f"Insert {var_name} y-axis limit")
            need_log = ST.st.checkbox(f"Show {var_name} log plot")

            if need_limit:
                y_limit = ST.st.number_input(f"{var_name} y-limit", value=100)
                plt.ylim([0, y_limit])

            fig_hist, ax_hist = plt.subplots(figsize=(15, 5))
            sns.histplot(data=num_hist_df, x=var_name, bins=bins, kde=True, hue=DEPENDENT_VARIABLE_NAME)
            ST.st.pyplot(fig_hist)

            if need_log:
                log_val = np.log(num_hist_df)
                fig_hist_log, ax_hist_log = plt.subplots(figsize=(15, 5))
                sns.histplot(data=log_val, x=var_name, bins=bins, kde=True, hue=DEPENDENT_VARIABLE_NAME)
                ST.st.pyplot(fig_hist_log)

add_cat_hist_plots = ST.st.checkbox(f"Categorical countplots")

if add_cat_hist_plots:

    cat_countplot_df = pd.concat([sst.data_categorical, sst.dependent_variable], axis=1)
    for i, var_name in enumerate(selected_categorical_values):
        fig_countplot = plt.figure(figsize=(10, 5))
        sns.countplot(data=cat_countplot_df, x=var_name, hue=DEPENDENT_VARIABLE_NAME)
        ST.st.pyplot(fig_countplot)

add_binary_count_plots = ST.st.checkbox(f"Binary countplots")

if add_binary_count_plots:

    binary_countplot_df = pd.concat([sst.data_binary, sst.dependent_variable], axis=1)
    for i, var_name in enumerate(selected_binary_values):
        fig_countplot = plt.figure(figsize=(10, 5))
        sns.countplot(data=binary_countplot_df, x=var_name, hue=DEPENDENT_VARIABLE_NAME)
        ST.st.pyplot(fig_countplot)


ST.write_h2("DATA CLEANING")

with ST.st.expander("Clean numeric data"):

    imputer_strategy = ST.selectbox("Choose imputer", ["median", "mean", "most_frequent", "constant"])
    sst.summary["num_null_val_strategy"] = imputer_strategy

    if imputer_strategy == "constant":
        number = ST.st.number_input('Insert constant value', value=0)
        sst.numeric_value_imputer(imputer_strategy, constant_value=number)
    else:
        sst.numeric_value_imputer(imputer_strategy)

    imbalance_percent_num = ST.st.slider('Imbalance percent (numeric)', 0, 100, 80)
    ST.write_h5(f"imbalanced numeric columns - {sst.imbalance_numeric_columns(imbalance_percent_num / 10)}")
    ST.display_dataframe(sst.is_null_table_numeric(), "is numeric null")

    if ST.st.button('Create numeric data table'):
        ST.filter_func(sst.data_numeric, "Numeric data", selected_numeric_values)
        ST.filter_func(sst.data_numeric_test, "Numeric test data", selected_numeric_values)

with ST.st.expander("Clean categorical data"):
    ST.write_h5("na values replaced with 'None'")

    drop_categorical_columns = ST.multiselct("Dropping following columns",
                                             sst.data_categorical.columns,
                                             ["PassengerId", "Cabin", "Name"])
    sst.summary["cat_data_removed_col"] = len(drop_categorical_columns)

    sst.drop_categorical_columns(drop_categorical_columns)

    imbalance_percent = ST.st.slider('Imbalance percent', 0, 100, 80)
    ST.write_h5(f"imbalanced categorical columns - {sst.imbalance_categorical_columns(imbalance_percent / 10)}")

    radio_option_cat = ST.radio("Encoding",
                                ["One hot encoding", "No"])
    sst.summary["cat_encoding"] = radio_option_cat

    sst.encoding_categorical_values(radio_option_cat)

    if ST.st.button('Create categorical data table'):
        ST.display_dataframe(sst.data_categorical, "Categorical columns after cleanup")
        ST.display_dataframe(sst.data_categorical_test, "Categorical columns after cleanup")

with ST.st.expander("Clean binary data data"):

    radio_option_binary = ST.radio("Choose encoding option", ["False: 0, True: 1, None: 0", "False: 0, True: 1, None: 1"])
    sst.summary["binary_nan_strategy"] = radio_option_binary

    if radio_option_binary == "False: 0, True: 1, None: 0":
        sst.replace_binary_missing_values(0)
    else:
        sst.replace_binary_missing_values(1)

    ST.display_dataframe(sst.is_null_table_binary(), "is binary null")
    ST.display_dataframe(sst.binary_table_with_binary_values(), "Counts")

ST.write_h2("COMBINE DATA")

with ST.st.expander("Combine data"):
    pfc = PFC.PrepareForCalculation(final_df=sst.combine_data(),
                                    final_test_df=sst.combine_test_data(),
                                    dependent_var_name=DEPENDENT_VARIABLE_NAME)
    final_df = pfc.final_df
    final_df_test = pfc.final_df_test

    ST.display_dataframe(pfc.combined_info(), "Combined data check")

    if ST.st.button('Create data table'):
        ST.display_dataframe(final_df, "final_df")
        ST.display_dataframe(final_df_test, "final_df_test")

    sst.summary["final_df_shape"] = f"{final_df.shape}"
    sst.summary["final_df_test_shape"] = f"{final_df_test.shape}"

ST.write_h2("PREPARE FOR CALCULATIONS")

with ST.st.expander("Train test split"):

    test_size = ST.st.slider('Test size', 1, 99, 20)
    y = pfc.y_value
    X_train, X_test, y_train, y_test = pfc.trains_test_split(test_size)

    ST.st.write(f"Test size = {test_size} %")
    ST.display_dataframe(pfc.splited_data_info(), "Split data check")

with ST.st.expander("Numeric value scaling"):

    radio_option_scaling = ST.radio("Scaling", ["MinMaxScaler", "Standard_scaler", "No"])
    numeric_columns = list(sst.data_numeric.columns)

    sst.summary["scaler"] = radio_option_scaling

    if radio_option_scaling == "MinMaxScaler":
        pfc.add_min_max_scaler(list(sst.data_numeric.columns))
    elif radio_option_scaling == "Standard_scaler":
        pfc.add_standard_scaler(list(sst.data_numeric.columns))

    if ST.st.button('Scaled data check'):
        ST.display_dataframe(pfc.X_train[numeric_columns], "X_train")
        ST.display_dataframe(pfc.X_test[numeric_columns], "X_test")
        ST.display_dataframe(pfc.final_df_test[numeric_columns], "final_df_test")

ST.write_h2("CALCULATIONS")

if grid_search:
    grid_search = GS.GridSearchAV(X_train=X_train,
                                  X_test=X_test,
                                  y_train=y_train,
                                  y_test=y_test,
                                  final_df_test=final_df_test)

    ST.write_h4("Grid search")
    sst.summary["calculation_strategy"] = "grid search"

    with ST.st.expander("Model selection"):
        is_logistic_regression = ST.st.checkbox("Logistic regression")
        is_KNN = ST.st.checkbox("KNN")
        is_RandomForest = ST.st.checkbox("RandomForest")
        is_XGBoost = ST.st.checkbox("XGBoost")
        # is_LGBM = ST.st.checkbox("LGBM")
        is_CatBoost = ST.st.checkbox("CatBoost")
        is_NaiveBayes = ST.st.checkbox("NaiveBayes")
        is_SVC = ST.st.checkbox("SVC")

        if is_logistic_regression:
            grid_search.add_logistic_regression()

        if is_KNN:
            grid_search.add_KNN()

        if is_RandomForest:
            grid_search.add_random_forest()

        if is_XGBoost:
            grid_search.add_XGBoost()

        # if is_LGBM:
        #     grid_search.add_LGBM()

        if is_CatBoost:
            grid_search.add_catboost()

        if is_NaiveBayes:
            grid_search.add_naivebayes()

        if is_SVC:
            grid_search.add_svc()

    with ST.st.expander("Grid search"):
        clf_best_params = {}

        if ST.st.button('Start grid search'):
            grid_search.start_grid_search()

            ST.display_dataframe(grid_search.get_valid_scores(), "Valid scores")
            ST.display_dataframe(grid_search.get_clf_best_params(), "Best params")

    with ST.st.expander("Best params for training with cross validation"):

        start_cross_validation = ST.st.checkbox("Start cross validation")
        if start_cross_validation:

            if is_logistic_regression:
                grid_search.add_logistic_regression(True)

            if is_KNN:
                grid_search.add_KNN(True)

            if is_RandomForest:
                grid_search.add_random_forest(True)

            if is_XGBoost:
                grid_search.add_XGBoost(True)

            # if is_LGBM:
            #     grid_search.add_LGBM(True)

            if is_CatBoost:
                grid_search.add_catboost(True)

            if is_NaiveBayes:
                grid_search.add_naivebayes(True)

            if is_SVC:
                grid_search.add_svc(True)

            classifiers_for_sum, scores_for_sum, score_sum, calsif_count = grid_search.start_cross_validation(5)
            sst.summary["cross_fold_models"] = classifiers_for_sum
            sst.summary["cross_fold_results"] = scores_for_sum
            sst.summary["cross_fold_results_avg"] = (score_sum / calsif_count).round(2)

            plt.figure(figsize=(10, 4))
            sns.histplot(grid_search.get_predictions(), binwidth=0.01, kde=True)
            plt.title('Predicted probabilities')
            plt.xlabel('Probability')
            ST.st.pyplot(plt)

ST.write_h4("Individual models")

ST.write_h1("SUBMISSION")

if all_saved_dataframes:
    ST.write_h2("ALL SAVED DATAFRAMES")

    with ST.st.expander("Check all summaries"):
        path = 'SpaceShipTitanic/results'
        file_names = os.listdir(path)
        all_dataframes = pd.DataFrame()

        for file_name in file_names:
            raw_data = pd.read_csv(f"SpaceShipTitanic/results/{file_name}")
            all_dataframes = pd.concat([all_dataframes, raw_data])

        ST.display_dataframe(all_dataframes, "ALL DATAFRAMES")

with ST.st.expander("Summary"):
    true_false_border = ST.st.slider("True/ False border", 0.0, 1.0, 0.5, 0.05)
    sst.summary["true_false_border"] = true_false_border

    kaggle_score = ST.st.text_input('Kaggle score')
    any_comments = ST.st.text_input("Any comments?")

    sst.summary["kaggle_score"] = kaggle_score
    sst.summary["comment"] = any_comments

    ST.display_dataframe(sst.summary_table(), "Summary")

    if grid_search:
        if ST.st.button("Save summary"):
            sst.summary_table().to_csv(f"SpaceShipTitanic/results/results_{datetime.now().strftime('%Y-%m-%d_%H:%M')}.csv", index=False)

        rounded_predictions = list((grid_search.get_predictions() >= true_false_border).astype(int))
        pred_labels = list(set(rounded_predictions))
        pred_values = [rounded_predictions.count(pred_labels[0]), rounded_predictions.count(pred_labels[1])]

        fig1, ax1 = plt.subplots(figsize=(15, 5))
        ax1.pie(pred_values,
                labels=pred_labels,
                autopct='%1.1f%%',
                shadow=True,
                startangle=90)
        ax1.axis('equal')

        ST.st.pyplot(fig1)


if grid_search:
    with ST.st.expander("Create submission"):

        if ST.st.button("Create submission file"):
            raw_test_data_s = pd.read_csv("SpaceShipTitanic/test.csv")
            submission = pd.DataFrame(raw_test_data_s["PassengerId"])
            final_predictions = (grid_search.get_predictions() >= true_false_border).astype(int)
            submission["Transported"] = final_predictions
            submission["Transported"] = submission["Transported"].replace({0: False, 1: True})

            csv_f = submission.to_csv(index=False).encode('utf-8')

            ST.st.download_button(
                label="Download data as CSV",
                data=csv_f,
                file_name='submission.csv',
                mime='text/csv',
            )



