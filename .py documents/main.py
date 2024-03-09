### Libraries

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from credit_score import data_reading_and_understanding as dr
from credit_score import feature_engineering as fe
from credit_score import model as md
from credit_score import variable_evaluations as ve
from sklearn.neighbors import LocalOutlierFactor
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

from warnings import filterwarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore")


### Options

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 350)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
warnings.simplefilter(action="ignore")


### Datasets

# Reading datasets

app = pd.read_csv("credit_score/Datasets/application_record.csv")
credit = pd.read_csv("credit_score/Datasets/credit_record.csv")


# Checking for duplicated observations

app_control = app.drop("ID", axis=1)
(app_control.duplicated() == 1).sum() # 348472 duplicated observations out of 438557 observations
not_duplicated_app = app[app_control.duplicated() == 0] # Not duplicated observations


# Defining customers' age and merging with application data

customer_age = pd.DataFrame(credit.groupby(["ID"])["MONTHS_BALANCE"].agg(min)) # Due to 0 is the current month and -1 is the previous month, minimum "MONTHS_BALANCE" observation will result as customer's starting point.
customer_age = customer_age.rename(columns={"MONTHS_BALANCE": "TENURE"}) # Renaming "MONTHS_BALANCE" as "TENURE" will describe variable better.
data = not_duplicated_app.merge(customer_age, on="ID", how="inner") # We only take in account the customers who has record in "app" and "credit" datasets.


# Classifying customers' paying status (0: No risk, 1: Risk) and merging with application data

credit["STATUS"].value_counts()

credit_risk = credit[credit["STATUS"] != "X"] # Since "X" value in "STATUS" variable means "no loan for that month", we will not consider these values. Nevertheless, for a different calculation method of "STATUS", which means risk, X values would be useful.

map_status = {"C": 0,
              "0": 0.1,
              "1": 0.2,
              "2": 0.4,
              "3": 0.6,
              "4": 0.8,
              "5": 1}

# By reason of the case do not have labels this part of the case has a crucial importance
# and should be considered as crossroads. Giving different weights to paying status' regarding their delay
# comparing each other, every paying status have a relative penalty, in other words, risk value.
# Rather than binary scoring, calculating mean risk value of every customer by considering their record
# from starting point to current time pays attention to customers'
# all "debt paying behaviour" will assess customers by their loyalty to debt.

credit_risk["STATUS"] = credit_risk["STATUS"].map(map_status)

credit_target = pd.DataFrame(credit_risk.groupby(["ID"])["STATUS"].mean())

credit_target["STATUS"] = credit_target["STATUS"].apply(lambda x: 0 if x < 0.1 else 1)

credit_target = credit_target.rename(columns={"STATUS": "RISK"})

credit_target.value_counts()

df = data.merge(credit_target, on="ID", how="inner")


### Exploratory Data Analysis

dr.check_data(df)

df.rename(columns={"CODE_GENDER": "gender",
                   "FLAG_OWN_CAR": "own_car",
                   "FLAG_OWN_REALTY": "own_realty",
                   "DAYS_BIRTH": "birthday",
                   "DAYS_EMPLOYED": "employment_date",
                   "FLAG_MOBIL": "own_mobile",
                   "CNT_CHILDREN": "num_child",
                   "AMT_INCOME_TOTAL": "annual_income",
                   "NAME_EDUCATION_TYPE": "education",
                   "NAME_FAMILY_STATUS": "marital_status",
                   "NAME_HOUSING_TYPE": "house_type",
                   "FLAG_EMAIL": "email",
                   "NAME_INCOME_TYPE": "income_type",
                   "FLAG_WORK_PHONE": "work_phone",
                   "FLAG_PHONE": "phone",
                   "CNT_FAM_MEMBERS": "family_size",
                   "OCCUPATION_TYPE": "occu_type",
                   "TENURE": "tenure",
                   "RISK": "risk"
                   }, inplace=True)


# Catching categorical and numerical Variables

cat_cols, num_cols, cat_but_car = dr.grab_col_names(df, cat_th=20, car_th=50)

num_cols.remove("ID")


# Summary for Categorical Variables

for col in cat_cols:
    dr.cat_summary(df, col, plot=True)

for col in cat_cols:
    dr.target_summary_with_cat(df, "risk", col)


# Summary for Numerical Variables

for col in num_cols:
    dr.num_summary(df, col, plot=True)

for col in num_cols:
    dr.target_summary_with_num(df, "risk", col)


# Almost none of the variables has variation between each other,
# thus there is a considerable need for feature extraction to explain causing effects.


# Correlation Analysis

dr.correlation_analysis(df, num_cols, "risk")

# There is a negative correlation between "tenure" and "risk", which is the target variable, in this case, tenure would
# be useful for detecting risky customers.


### Data Preparation

# Since there is no variation between observations regarding "own_mobile" variable, we drop "own_mobile".

df = df.drop(["ID", "own_mobile"], axis=1)


# Missing Data Analysis

dr.missing_values_table(df) # There are 2761 missing values in "occu_type".


# Filling in missing data

df.loc[((df["occu_type"].isnull()) & (df["employment_date"] < 0)), "occu_type"] = "Unknown"

df.loc[((df["occu_type"].isnull()) & (df["employment_date"] > 0)), "occu_type"] = "Unemployed"


### Feature Extraction/Interactions

# Creating Age variable by Birthday variable

df["age"] = round((df["birthday"] / 365) * -1).astype(int)


# Creating categorical age variable by "age"

bins = [df["age"].min(), 34, 45, 57, df["age"].max()]
labels = ["21-34", "35-45", "46-57", "58-69"]
df["age_category"] = pd.cut(df["age"], bins=bins, labels=labels, include_lowest=True)


# Creating "experience" variable by "employment_date" variable

df["experience"] = round(df["employment_date"] / 30)

df["experience"] = df["experience"].apply(lambda x: int(x * -1) if x < 0 else 0)


# Creating experience categorical variable from experience by using spec. function asdfawebgasdgs

def map_experience_to_category(experience):
    exp_by_year = experience / 12
    if 0 <= exp_by_year < 1:
        return "0-1 years_exp"
    elif 1 <= exp_by_year < 3:
        return "1-3 years_exp"
    elif 3 <= exp_by_year < 5:
        return "3-5 years_exp"
    elif 5 <= exp_by_year < 10:
        return "5-10 years_exp"
    elif 10 <= exp_by_year < 20:
        return "10-20 years_exp"
    else:
        return "20+ years_exp"

df["exp_category"] = df["experience"].apply(map_experience_to_category)


# Creating occupation stage by "exp_category"

df.loc[(df["exp_category"] == "0-1 years_exp") |
       (df["exp_category"] == "1-3 years_exp") |
       (df["exp_category"] == "3-5 years_exp"), "occu_stage"] = "early_stage"

df.loc[(df["exp_category"] == "5-10 years_exp") |
       (df["exp_category"] == "10-20 years_exp"), "occu_stage"] = "mid_stage"

df.loc[(df["exp_category"] == "20+ years_exp"), "occu_stage"] = "late_stage"


# Turning "tenure" into positive

df["tenure"] = df["tenure"] * -1


# Creating new month_balance categorical variable from month_balance by using spec. function sdgfetrgbra

def map_month_to_category(month):
    if 0 <= month < 6:
        return "0_6_customer"
    elif 6 <= month < 12:
        return "6_12_customer"
    elif 12 <= month < 24:
        return "12_24_customer"
    elif 24 <= month < 36:
        return "24_36_customer"
    elif 36 <= month < 48:
        return "36_48_customer"
    else:
        return "48+_customer"

df["cat_tenure"] = df["tenure"].apply(map_month_to_category)

df["cat_tenure"].value_counts()

df.groupby("cat_tenure")["risk"].mean()


# Creating categorical income variable by "annual_income"

df["income_category"] = pd.qcut(df["annual_income"], q=4, labels=["low", "medium", "high", "very_high"])


# Level-based Occupation Segmentation by "occu_type"

df.loc[(df['occu_type'] == 'Cleaning staff') |
       (df['occu_type'] == 'Cooking staff') |
       (df['occu_type'] == 'Drivers') | 
       (df['occu_type'] == 'Private service staff') |
       (df['occu_type'] == 'Laborers') |
       (df['occu_type'] == 'Security staff') |
       (df['occu_type'] == 'Waiters/barmen staff'), "level_based_occu"] = "labor_worker"


df.loc[(df['occu_type'] == 'Accountants') |
       (df['occu_type'] == 'HR staff') |
       (df['occu_type'] == 'Medicine staff') |
       (df['occu_type'] == 'Realty agents') |
       (df['occu_type'] == 'Sales staff') |
       (df['occu_type'] == 'Secretaries'), "level_based_occu"] = "office_worker"


df.loc[(df['occu_type'] == 'Managers') |
       (df['occu_type'] == 'High skill tech staff') |
       (df['occu_type'] == 'IT staff') |
       (df['occu_type'] == 'Core staff'), "level_based_occu"] = "high_worker"


df.loc[(df['occu_type'] == 'Unemployed'), "level_based_occu"] = "unemployed"


df.loc[(df['occu_type'] == 'Unknown'), "level_based_occu"] = "unknown"


# Laborers

df.loc[(df["occu_type"] == "Laborers") | (df["occu_type"] == "Low-skill Laborers"), "occu_type"] = "Laborers"


# Industry-based Occupation Segmentation by "occu_type"

df.loc[(df['occu_type'] == 'Cleaning staff') |
       (df['occu_type'] == 'Cooking staff') |
       (df['occu_type'] == 'Private service staff') |
       (df['occu_type'] == 'Waiters/barmen staff') |
       (df['occu_type'] == 'Medicine staff') |
       (df['occu_type'] == 'Drivers'), "ind_based_occu"] = "service_industry"

df.loc[(df['occu_type'] == 'Accountants') |
       (df['occu_type'] == 'HR staff') |
       (df['occu_type'] == 'Secretaries') |
       (df['occu_type'] == 'Core staff') |
       (df['occu_type'] == 'Managers') |
       (df['occu_type'] == 'Sales staff') |
       (df['occu_type'] == 'Security staff') |
       (df['occu_type'] == 'Realty agents'), "ind_based_occu"] = "pro_serv_industry"

df.loc[(df['occu_type'] == 'IT staff') |
       (df['occu_type'] == 'High skill tech staff'), "ind_based_occu"] = "tech_industry"

df.loc[(df['occu_type'] == 'Unemployed'), "ind_based_occu"] = "unemployed"

df.loc[(df['occu_type'] == 'Unknown'), "ind_based_occu"] = "unknown"

df.loc[(df['occu_type'] == 'Laborers'), "ind_based_occu"] = "laborers"


# Creating categorical "active_worker" variable by "employment_date"

df["active_worker"] = df["employment_date"].apply(lambda x: 1 if x < 0 else 0)


# Creating categorical "parent" variable by "num_child"

df["parent"] = df["num_child"].apply(lambda x: 0 if x == 0 else 1)


# Creating categorical "life_stage" customer variable by "age"

df["life_stage"] = df["age"].apply(lambda x: "young" if x <= 25 else ("mature" if 25 < x <= 50 else "elder"))



# Creating education segmentation by "education"

df.loc[(df["education"] == "Lower secondary") |
       (df["education"] == "Secondary / secondary special") |
       (df["education"] == "Incomplete higher"), "edu_segment"] = "low_edu"

df.loc[(df["education"] == "Higher education") | (df["education"] == "Academic degree"), "edu_segment"] = "high_edu"


# Creating annual_income / family_size variable

df["income/fam_size"] = df["annual_income"] / df["family_size"]


### Encoding

df1 = df.copy()

cat_cols, num_cols, cat_but_car = dr.grab_col_names(df1, cat_th=20, car_th=45)

cat_cols = [col for col in cat_cols if col not in ["risk"]]

df1 = fe.one_hot_encoder(df1, cat_cols, drop_first=True)


# Local Outlier Factor (LOF) Method

clf = LocalOutlierFactor(n_neighbors=5)
clf.fit_predict(df1)

df1_scores = clf.negative_outlier_factor_
df1_scores[0:5]
df1_scores = -df1_scores
np.sort(df1_scores)[0:5]
scores = pd.DataFrame(np.sort(df1_scores))

th = np.sort(df1_scores)[3]
df1[df1_scores < th]
df1[df1_scores < th].shape
df1.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
df1[df1_scores < th].index

df1 = df1[df1_scores > th]

# There are 3 outliers


### Modeling

y = df1["risk"]
X = df1.drop(["risk"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

X_train = X_train.astype("float32")
y_train = y_train.astype("float32")
X_test = X_test.astype("float32")
y_test = y_test.astype("float32")

print(f"x_train boyutu: {X_train.shape}")
print(f"x_test boyutu: {X_test.shape}")

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Reshape the arrays
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train.shape
X_test.shape

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1, 97)),
    tf.keras.layers.Dense(units=64, activation="relu", name="hidden_layer"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=1, activation="sigmoid", name="output_layer")])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, verbose=1, mode="min"),
    ModelCheckpoint(filepath="class_model.h5", monitor="val_loss", mode="min",
                    save_best_only=True, save_weights_only=False, verbose=1)]

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), "accuracy"])

model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=callbacks)

loss, precision, recall, acc = model.evaluate(X_test, y_test, verbose=False)

print("\nTest Accuracy: %.1f%%" % (100.0 * acc))
print("\nTest Loss: %.1f%%" % (100.0 * loss))
print("\nTest Precision: %.1f%%" % (100.0 * precision))
print("\nTest Recall: %.1f%%" % (100.0 * recall))


# Model Success Evaluation

#---------------------
# Accuracy
# ---------------------
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], color="b", label='Training Accuracy')
plt.plot(history.history['val_accuracy'], color="r", label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy', fontsize=16)

#-------------------------
#Loss
# --------------------------
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], color="b", label='Training Loss')
plt.plot(history.history['val_loss'], color="r", label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.ylim([0, max(plt.ylim())])
plt.title('Training and Validation Loss', fontsize=16)
plt.show()