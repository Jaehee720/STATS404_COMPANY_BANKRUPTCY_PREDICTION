#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# import data
file_name = "bankrupt_data.csv"


# In[2]:


df = pd.read_csv(filepath_or_buffer=file_name, encoding="latin-1")


# In[3]:


# check row and columns
df.shape


# In[4]:


# check if there are missing values
df.info()
# There is no missing value


# In[5]:


# remove space from columns

a = df.columns.tolist()
a = [x.strip(" ") for x in a]

df.columns = a


# In[6]:


# Since we have too many features, we will select some features that are meaningful using one of the feature selection methods, filter method.

# step1. Remove the features that have a low correlation with the target value.

X = df.iloc[:, 1:]
y = df["Bankrupt?"]

importances = X.apply(lambda x: x.corr(y))
indices = np.argsort(importances)
print(importances[indices])

names = df.columns[1:]

plt.figure(figsize=(15, 15))

plt.title("correlation with Bankrupt")
plt.barh(range(len(indices)), importances[indices], color="g", align="center")
plt.yticks(range(len(indices)), [names[i] for i in indices])
plt.xlabel("Relative Importance")
plt.show()


# In[7]:


# We will save the features that have a correlation higher than the absolute value of 0.10. Therefore, I will set the treshold to the absolute value of 0.10.
a = []

for i in range(0, len(indices)):
    if np.abs(importances[i]) > 0.10:
        a.append(names[i])
        print(names[i])

print(len(a))

X = df.loc[:, a]


# In[8]:


# Step2: Remove the features that have a correlation higher than 90% with other independent variables.

corr = X.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):
    for j in range(i + 1, corr.shape[0]):
        if corr.iloc[i, j] >= 0.9:
            if columns[j]:
                columns[j] = False

selected_columns = X.columns[columns]

print(len(selected_columns))
# Now we only have 21 features. I will create a new df with these selected columns and the features that I think it's important.

new_df = pd.concat(
    [
        df[
            [
                "Bankrupt?",
                "net income to total assets",
                "Cash flow to total assets",
                "tax Pre-net interest rate",
                "inventory and accounts receivable/net value",
            ]
        ],
        X[selected_columns],
    ],
    axis=1,
)
print(new_df.columns)
print(new_df.shape)

# We have 25 features and target.


# In[9]:


# split the dataset into training and test sets


x = new_df.iloc[:, 1:]
y = new_df["Bankrupt?"]

# use K-fold function to split data. 80% of the data will be training set and the rest of it will be test set.
skf = StratifiedKFold(n_splits=5, shuffle=False)

for train_index, test_index in skf.split(x, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    original_xtrain, original_xtest = x.iloc[train_index], x.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# In[10]:


# check if the percentage of bankruptcy is similar between the train set and the test set
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print(train_counts_label / len(original_ytrain))
print(test_counts_label / len(original_ytest))

# The distribution of bankruptcy is similar to the original dataset.


# In[11]:


# export a training set and a test set as csv
original_xtrain.to_csv(
    r"C:\Users\jeongjaehui\Documents\Stats_404\JEONG-JAEHEE\bankrupt_training_set.csv",
    index=False,
    header=True,
)
original_xtest.to_csv(
    r"C:\Users\jeongjaehui\Documents\Stats_404\JEONG-JAEHEE\bankrupt_test_set.csv",
    index=False,
    header=True,
)


# In[12]:


original_xtrain.shape


# In[13]:


# splite the data into training and valiation sets


x_train, x_val, y_train, y_val = train_test_split(
    original_xtrain,
    original_ytrain,
    test_size=0.1,
    random_state=1,
    stratify=original_ytrain,
    shuffle=True,
)


# In[14]:


# generate a logistics regression model

log_reg = LogisticRegression(class_weight="balanced", max_iter=10000)
log_model = log_reg.fit(x_train, y_train)
log_pred = log_model.predict(x_val)


# check a classification
print(classification_report(y_val, log_pred))

print(
    pd.DataFrame(
        confusion_matrix(y_val, log_pred),
        columns=["Predicted Nagative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[15]:


log_pred = log_model.predict(original_xtest)

# check a classification
print(classification_report(original_ytest, log_pred))

print(
    pd.DataFrame(
        confusion_matrix(original_ytest, log_pred),
        columns=["Predicted Nagative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[16]:


# check coefficient
log_model.coef_[0]


# In[17]:


# fit the best model
cat = CatBoostClassifier(
    iterations=500,
    loss_function="Logloss",
    depth=6,
    l2_leaf_reg=1e-20,
    eval_metric="Accuracy",
    leaf_estimation_iterations=10,
    use_best_model=True,
    logging_level="Silent",
    random_seed=42,
    class_weights=(1, 30),
)

cat_model = cat.fit(x_train, y_train, eval_set=(x_val, y_val))


# In[18]:


cat_pred = cat_model.predict(x_val)


# In[19]:


print(classification_report(y_val, cat_pred))

# create a confusion matrix using the validation set
print(
    pd.DataFrame(
        confusion_matrix(y_val, cat_pred),
        columns=["Predicted Nagative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[20]:


cat_pred = cat_model.predict(original_xtest)

print(classification_report(original_ytest, cat_pred))

# create a confusion matrix using the validation set
print(
    pd.DataFrame(
        confusion_matrix(original_ytest, cat_pred),
        columns=["Predicted Nagative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[21]:


# generate a precision recall curve to check if we can change the threshold to increase FalseNegative.

probs = cat_model.predict_proba(original_xtest)
positive_probs = probs[:, 1]


precision, recall, thresholds = precision_recall_curve(original_ytest, positive_probs)
# retrieve probability of being 1(in second column of probs_y)
pr_auc = metrics.auc(recall, precision)

plt.title("Precision-Recall vs Threshold Chart")
plt.plot(thresholds, precision[:-1], "b--", label="Precision")
plt.plot(thresholds, recall[:-1], "r--", label="Recall")
plt.ylabel("Precision, Recall")
plt.xlabel("Threshold")
plt.legend(loc="lower left")
plt.ylim([0, 1])


# In[22]:


THRESHOLD = 0.37
thre_37_preds_cat = np.where(cat_model.predict_proba(x_val)[:, 1] > THRESHOLD, 1, 0)

# check a classification
print(classification_report(y_val, thre_37_preds_cat))

print(
    pd.DataFrame(
        confusion_matrix(y_val, thre_37_preds_cat),
        columns=["Predicted Nagative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[23]:


thre_37_preds_cat = np.where(
    cat_model.predict_proba(original_xtest)[:, 1] > THRESHOLD, 1, 0
)

# check a classification
print(classification_report(original_ytest, thre_37_preds_cat))

print(
    pd.DataFrame(
        confusion_matrix(original_ytest, thre_37_preds_cat),
        columns=["Predicted Negative", "Predicted Positive"],
        index=["Actual Negative", "Actual Positive"],
    )
)


# In[24]:


# check features' importance of this model
feat_imp_cat = cat_model.get_feature_importance(prettified=True)

important_features = feat_imp_cat.loc[feat_imp_cat["Importances"] > 0]
# important_features[0] = important_features[0].iloc[:,0]

important_features


# In[ ]: