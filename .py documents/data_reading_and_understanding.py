import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def check_data(dataframe):
    print(20 * "-" + "Information".center(20) + 20 * "-")
    print(dataframe.info())
    print(20 * "-" + "Data Shape".center(20) + 20 * "-")
    print(dataframe.shape)
    print(20 * "-" + "Nunique".center(20) + 20 * "-")
    print(dataframe.nunique())
    print("\n" + 20 * "-" + "The First 5 Data".center(20) + 20 * "-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print((dataframe.isnull().sum()).sort_values(ascending=False))
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
  It returns categorical, numerical and cardinal variables names and quantity
  Parameters
  ----------
  dataframe: dataframe
      the dataframe where we take the variables names
  cat_th: int, float
      class threshold value for variables that are numeric but categorical
  car_th: int, float
      class threshold value for variables that are categorical but cardinal

  Returns
  -------
      cat_cols: list
          Categorical variables list
      num_cols: list
          Numerical variables list
      cat_but_car: list
          List of cardinal variables with categorical view
  Notes
  -----
  cat_cols + num_cols + cat_but_car = total variables
  num_but_cat in cat_cols

  """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    #cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    #num_but_cat = [col for col in dataframe.columns if pd.Series(dataframe[col]).nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        plt.figure(figsize=(12, 8))
        sns.countplot(x=dataframe[col_name], data=dataframe)
        #plt.xticks(rotation=30)
        plt.show()


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def correlation_analysis(dataframe, num_cols, target_col):
    """
    This function calculates correlation matrix, visualizes it with a heatmap,
    and computes the correlation of each numeric column with the target column.

    Parameters:
    - dataframe: DataFrame, the dataset
    - num_cols: list of strings, column names of numeric features
    - target_col: string, the target column name

    Returns:
    - None
    """

    # Correlation Matrix
    correlation_matrix = dataframe[num_cols].corr()

    # Visualization
    f, ax = plt.subplots(figsize=[18, 13])
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()

    # Correlation with Target Column
    correlation_with_target = dataframe[num_cols].corrwith(dataframe[target_col]).sort_values(ascending=False)
    print(correlation_with_target)












