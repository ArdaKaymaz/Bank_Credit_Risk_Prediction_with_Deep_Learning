import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_th(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=1):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_th(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

def base_model_results(df, cat_cols, target_col="Churn", drop_first=True, random_state=12345):
    # Create a copy of the DataFrame
    dff = df.copy()

    # Perform one-hot encoding on categorical columns
    dff = pd.get_dummies(dff, columns=[col for col in cat_cols if col != target_col], drop_first=drop_first)

    # Separate features and target variable
    y = dff[target_col]
    X = dff.drop([target_col, "customerID"], axis=1)

    # Define models
    models = [('LR', LogisticRegression(random_state=random_state)),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier(random_state=random_state)),
              ('RF', RandomForestClassifier(random_state=random_state)),
              ('XGB', XGBClassifier(random_state=random_state)),
              ("LightGBM", LGBMClassifier(random_state=random_state)),
              ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

    # Iterate over models and perform cross-validation
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
        print(f"########## {name} ##########")
        print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
        print(f"AUC: {round(cv_results['test_roc_auc'].mean(), 4)}")
        print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
        print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
        print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
        print("\n")


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def standart_scaler_func(dataframe, num_cols):
    ss = StandardScaler()
    dataframe[num_cols] = ss.fit_transform(dataframe[[num_cols]])
    return dataframe


def min_max_scaler_func(dataframe, num_cols):
    mms = MinMaxScaler()
    dataframe[num_cols] = mms.fit_transform(dataframe[[num_cols]])
    return dataframe


def robust_scaler_func(dataframe, num_cols):
    rs = RobustScaler()
    dataframe[num_cols] = rs.fit_transform(dataframe[[num_cols]])
    return dataframe