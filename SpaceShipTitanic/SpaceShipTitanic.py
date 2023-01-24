import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder


class SpaceShipTitanic:
    def __init__(self, train_data, test_data):
        super().__init__()
        self.raw_data = train_data
        self.raw_test_data = test_data
        self.dependent_variable = self.raw_data["Transported"]
        self.dependent_variable = self.dependent_variable.replace({False: 0, True: 1})
        self.data_numeric = self.raw_data.select_dtypes("number")
        self.data_categorical = self.raw_data.select_dtypes("object")
        self.data_binary = self.raw_data[["CryoSleep", "VIP"]]
        self.data_categorical = self.data_categorical.drop(["CryoSleep", "VIP"], axis=1)
        self.data_numeric_test = self.raw_test_data.select_dtypes("number")
        self.data_categorical_test = self.raw_test_data.select_dtypes("object")
        self.data_binary_test = self.raw_test_data[["CryoSleep", "VIP"]]
        self.data_categorical_test = self.data_categorical_test.drop(["CryoSleep", "VIP"], axis=1)
        self.summary = {}
        self.missing_categorical_to_none()

    def total_missing_values(self):
        return self.raw_data.isnull().sum().sum()

    def total_missing_test_values(self):
        return self.raw_test_data.isnull().sum().sum()

    def raw_summary(self):
        raw_summary = {
            "Data type": ["Train", "Test"],
            "RAW data shape": [self.raw_data.shape, self.raw_test_data.shape],
            "Numeric shape": [self.data_numeric.shape, self.data_numeric_test.shape],
            "Categorical shape": [self.data_categorical.shape, self.data_categorical_test.shape],
            "Binary shape": [self.data_binary.shape, self.data_binary_test.shape],
            "Total missing values": [self.total_missing_values(), self.total_missing_test_values()]
        }

        return pd.DataFrame(raw_summary)

    def missing_dependent_values(self):
        dependent_info_dict = {
            "null values": self.dependent_variable.isnull().sum()
        }
        return pd.DataFrame(dependent_info_dict, index=["summary"])

    @staticmethod
    def is_null_table(dataframe_, table1, table2):
        missing_num_values_df = pd.DataFrame()
        missing_num_values_df[table1] = dataframe_.isnull().sum()
        missing_num_values_df[table2] = round(dataframe_.isnull().sum() / len(dataframe_) * 100, 2)
        result = missing_num_values_df[missing_num_values_df[table2] > 0]
        result = result.sort_values(by=[table2], ascending=False)
        return result

    def is_null_table_numeric(self):
        data_ = self.is_null_table(self.data_numeric, "Missing value count", "Missing value %")
        data_test_ = self.is_null_table(self.data_numeric_test, "Missing test value count", "Missing test value %")
        return pd.concat([data_, data_test_], axis=1)

    def is_null_table_categorical(self):
        data_ = self.is_null_table(self.data_categorical, "Missing value count", "Missing value %")
        data_test_ = self.is_null_table(self.data_categorical_test, "Missing test value count", "Missing test value %")
        return pd.concat([data_, data_test_], axis=1)

    def is_null_table_binary(self):
        data_ = self.is_null_table(self.data_binary, "Missing value count", "Missing value %")
        data_test_ = self.is_null_table(self.data_binary_test, "Missing test value count", "Missing test value %")
        return pd.concat([data_, data_test_], axis=1)

    def get_unique_categorical_values(self, num):
        unique_cat_values = pd.DataFrame()

        for col in self.data_categorical.columns:
            temp_dict = {col: self.data_categorical[col].unique()}
            new_column = pd.DataFrame(temp_dict)
            unique_cat_values = pd.concat([unique_cat_values, new_column], axis=1)

        return unique_cat_values.head(num)

    @staticmethod
    def is_null_column_names(dataframe_):
        missing_num_values_df = pd.DataFrame()
        missing_num_values_df["Missing value count"] = dataframe_.isnull().sum()
        result = missing_num_values_df[missing_num_values_df["Missing value count"] > 0]
        return result.index.to_list()

    def numeric_value_imputer(self, imputer_strategy, constant_value=0):
        null_columns_to_impute = self.is_null_column_names(self.data_numeric)
        null_columns_to_impute_test = self.is_null_column_names(self.data_numeric_test)

        if imputer_strategy == "constant":
            imputer = SimpleImputer(strategy=imputer_strategy, fill_value=constant_value)
        else:
            imputer = SimpleImputer(strategy=imputer_strategy)

        imputer.fit(self.data_numeric[null_columns_to_impute])
        self.data_numeric[null_columns_to_impute] = imputer.transform(self.data_numeric[null_columns_to_impute])
        self.data_numeric_test[null_columns_to_impute_test] = imputer.transform(self.data_numeric_test[null_columns_to_impute_test])

    def missing_categorical_to_none(self):
        self.data_categorical.fillna("None", inplace=True)
        self.data_categorical_test.fillna("None", inplace=True)

    def drop_categorical_columns(self, columns):
        if len(columns) > 0:
            for col in columns:
                self.data_categorical.drop(col, axis=1, inplace=True)
                self.data_categorical_test.drop(col, axis=1, inplace=True)

    def encoding_categorical_values(self, encoding_option):
        if encoding_option == "One hot encoding":
            transformer = make_column_transformer(
                (OneHotEncoder(drop='first'), list(self.data_categorical.columns)), remainder='passthrough')

            transformed = transformer.fit_transform(self.data_categorical).toarray()
            transformed_test = transformer.fit_transform(self.data_categorical_test).toarray()

            self.data_categorical = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
            self.data_categorical_test = pd.DataFrame(transformed_test, columns=transformer.get_feature_names_out())

    def replace_binary_missing_values(self, value_to_replace):
        self.data_binary.fillna("None", inplace=True)
        self.data_binary_test.fillna("None", inplace=True)

        for col in self.data_binary.columns:
            self.data_binary[col] = self.data_binary[col].replace({False: 0, True: 1, "None": value_to_replace})
            self.data_binary_test[col] = self.data_binary_test[col].replace({False: 0, True: 1, "None": value_to_replace})

    def binary_table_with_binary_values(self):
        temp_df1 = pd.DataFrame()

        for idx, col in enumerate(self.data_binary.columns):
            temp_dict = {f"{col}-0": (self.data_binary[col] == 0).sum()}
            temp_dict2 = {f"{col}-1": (self.data_binary[col] == 1).sum()}
            new_column = pd.DataFrame(temp_dict, index=["Train"])
            new_column2 = pd.DataFrame(temp_dict2, index=["Train"])
            temp_df1 = pd.concat([temp_df1, new_column], axis=1)
            temp_df1 = pd.concat([temp_df1, new_column2], axis=1)

        temp_df2 = pd.DataFrame()

        for idx, col in enumerate(self.data_binary.columns):
            temp_dict3 = {f"{col}-0": (self.data_binary_test[col] == 0).sum()}
            temp_dict4 = {f"{col}-1": (self.data_binary_test[col] == 1).sum()}
            new_column3 = pd.DataFrame(temp_dict3, index=["Test"])
            new_column4 = pd.DataFrame(temp_dict4, index=["Test"])
            temp_df2 = pd.concat([temp_df2, new_column3], axis=1)
            temp_df2 = pd.concat([temp_df2, new_column4], axis=1)

        return pd.concat([temp_df1, temp_df2], axis=0)

    @staticmethod
    def imbalance_columns(dataframe_, imbalance_percent):
        imbalance_col = []
        for col in dataframe_:
            if dataframe_[col].value_counts().max() > (dataframe_.shape[0] * imbalance_percent):
                imbalance_col.append(col)
        return imbalance_col

    def imbalance_numeric_columns(self, imbalance_percent):
        return self.imbalance_columns(self.data_numeric, imbalance_percent)

    def imbalance_categorical_columns(self, imbalance_percent):
        return self.imbalance_columns(self.data_categorical, imbalance_percent)

    def combine_data(self):
        return pd.concat([self.data_numeric, self.data_categorical, self.data_binary, self.dependent_variable], axis=1)

    def combine_test_data(self):
        return pd.concat([self.data_numeric_test, self.data_categorical_test, self.data_binary_test], axis=1)

    def summary_table(self):
        self.summary["raw_dat_shape"] = f"{self.raw_data.shape}"
        self.summary["data_numeric_shape"] = f"{self.data_numeric.shape}"
        self.summary["data_categorical_shape"] = f"{self.data_categorical.shape}"
        self.summary["data_binary_shape"] = f"{self.data_binary.shape}"
        self.summary["raw_test_dat_shape"] = f"{self.raw_test_data.shape}"
        self.summary["data_test_numeric_shape"] = f"{self.data_numeric_test.shape}"
        self.summary["data_test_categorical_shape"] = f"{self.data_categorical_test.shape}"
        self.summary["data_test_binary_shape"] = f"{self.data_binary_test.shape}"
        self.summary["total_missing_values"] = self.total_missing_values()
        self.summary["total_missing_test_values"] = self.total_missing_test_values()
        self.summary["dependent_variable_value_counts"] = f"{self.dependent_variable.value_counts()[0]} / {self.dependent_variable.value_counts()[1]}"
        self.summary["cat_na_val"] = "replaced with None"

        return pd.DataFrame(self.summary, index=[datetime.now().strftime('%Y-%m-%d_%H:%M')])


