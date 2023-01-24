import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class PrepareForCalculation:
    def __init__(self, final_df, final_test_df, dependent_var_name):
        self.final_df = final_df
        self.final_df_test = final_test_df
        self.dependent_var_name = dependent_var_name
        self.y_value = self.final_df[self.dependent_var_name]
        self.X_values = self.final_df.drop(dependent_var_name, axis=1)
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()

    def combined_info(self):
        combined_data_info = {"train": [self.final_df.shape, self.final_df.isnull().sum().sum()],
                              "test": [self.final_df_test.shape, self.final_df_test.isnull().sum().sum()]}

        return pd.DataFrame(combined_data_info, index=["Shape", "is null"])

    def trains_test_split(self, test_size_percents):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_values,
                                                                                self.y_value,
                                                                                random_state=123,
                                                                                test_size=test_size_percents/100)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def splited_data_info(self):
        split_data_info_dict = {"X_train": [self.X_train.shape],
                                "X_test": [self.X_test.shape],
                                "y_train": [self.y_train.shape],
                                "y_test": [self.y_test.shape]
                                }

        return pd.DataFrame(split_data_info_dict, index=["Shape"])

    def add_min_max_scaler(self, numeric_columns):
        minmax_scaler = MinMaxScaler()
        data_numeric_t1 = minmax_scaler.fit_transform(self.X_train[numeric_columns])
        data_numeric_t2 = minmax_scaler.transform(self.X_test[numeric_columns])
        data_numeric_test = minmax_scaler.transform(self.final_df_test[numeric_columns])
        self.X_train[numeric_columns] = data_numeric_t1
        self.X_test[numeric_columns] = data_numeric_t2
        self.final_df_test[numeric_columns] = data_numeric_test

    def add_standard_scaler(self, numeric_columns):
        standard_scaler = StandardScaler()
        data_numeric_t1 = standard_scaler.fit_transform(self.X_train[numeric_columns])
        data_numeric_t2 = standard_scaler.transform(self.X_test[numeric_columns])
        data_numeric_test = standard_scaler.transform(self.final_df_test[numeric_columns])
        self.X_train[numeric_columns] = data_numeric_t1
        self.X_test[numeric_columns] = data_numeric_t2
        self.final_df_test[numeric_columns] = data_numeric_test




