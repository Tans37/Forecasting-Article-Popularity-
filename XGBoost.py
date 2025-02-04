#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import random
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt

#%%
############################ Random Search for Hyper-parameters ############################
file_path = '/Users/exrt/Desktop/Reg_dataset.csv'
data = pd.read_csv(file_path)
# data = data.sample(frac=0.5, random_state=42)

X = data.drop('shares', axis=1)
y = data['shares']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

base_score_value = y_train.mean()
xgb_reg = xgb.XGBRegressor(objective='count:poisson', base_score=base_score_value)
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'colsample_bytree': [0.3, 0.4, 0.5, 0.6, 0.7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
}
random_search = RandomizedSearchCV(xgb_reg, param_distributions=param_dist, 
                                   n_iter=25, scoring='neg_mean_squared_error', 
                                   cv=5, verbose=1, random_state=42)

random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)
# print("Best score: ", random_search.best_score_)




#%%
############################ Boostraping Hypothesis Tests ############################
data = pd.read_csv("/Users/exrt/Desktop/Reg_dataset.csv")
random.seed(1502)
np.random.seed(1502)

sample_size = int(0.7 * len(data))
data = data.sample(n=sample_size)

#%%
# params = {
#     'subsample': 0.6,
#     'max_depth': 4,
#     'learning_rate': 0.05,
#     'colsample_bytree': 0.5,
#     'eta': 0.01,
#     'objective': 'count:poisson'
# }
params = random_search.best_params_

#%%
############################# First Hypothesis ############################
impvar = ["data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world"]
# impvar = impvar[0:2]

bsn = 100
MSE_H1 = []
MSE_H0 = []

average_log_label = np.log(data["shares"].mean())
params['base_score'] = average_log_label
scaler = MinMaxScaler()

for i in tqdm(range(bsn), desc="Processing"):
    resampled_data = data.sample(frac=1, replace=True)

    X = resampled_data.drop('shares', axis=1)
    C = X.columns.tolist()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X,columns = C)
    y = resampled_data['shares']
    y = np.log(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    base_score_value = y_train.mean()
    # H1
    model_H1 = train(params, dtrain, num_boost_round=300)
    predictions_H1 = model_H1.predict(dtest)
    MSE_H1.append(np.sqrt(mean_squared_error(y_test, predictions_H1)))

    # H0
    X_train_H0 = X_train.drop(impvar, axis=1)
    X_test_H0 = X_test.drop(impvar, axis=1)
    dtrain_H0 = DMatrix(X_train_H0, label=y_train)
    dtest_H0 = DMatrix(X_test_H0, label=y_test)

    model_H0 = train(params, dtrain_H0, num_boost_round=300)
    predictions_H0 = model_H0.predict(dtest_H0)
    MSE_H0.append(np.sqrt(mean_squared_error(y_test, predictions_H0)))

p_value1 = np.mean(np.array(MSE_H0) < np.array(MSE_H1))
print(p_value1)

h_statistic, p_value_h = stats.kruskal(MSE_H1, MSE_H0)
print(f'Kruskal-Wallis H test statistic: {h_statistic}')
print(f'P-value: {p_value_h}')

print(np.mean(MSE_H0))
print(np.mean(MSE_H1))


#%%
impvar = ["is_weekend", "day_count"]

bsn = 100
MSE_H1 = []
MSE_H0 = []

average_log_label = np.log(data["shares"].mean())
params['base_score'] = average_log_label
scaler = MinMaxScaler()


for i in tqdm(range(bsn), desc="Processing"):

    resampled_data = data.sample(frac=1, replace=True)
    
    X = resampled_data.drop('shares', axis=1)
    C = X.columns.tolist()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns = C)
    y = resampled_data['shares']
    y = np.log(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    dtrain = DMatrix(X_train, label=y_train)
    dtest = DMatrix(X_test, label=y_test)

    base_score_value = y_train.mean()
    # H1
    model_H1 = train(params, dtrain, num_boost_round=300)
    predictions_H1 = model_H1.predict(dtest)
    MSE_H1.append(np.sqrt(mean_squared_error(y_test, predictions_H1)))

    # H0 
    X_train_H0 = X_train.drop(impvar, axis=1)
    X_test_H0 = X_test.drop(impvar, axis=1)
    dtrain_H0 = DMatrix(X_train_H0, label=y_train)
    dtest_H0 = DMatrix(X_test_H0, label=y_test)

    model_H0 = train(params, dtrain_H0, num_boost_round=300)
    predictions_H0 = model_H0.predict(dtest_H0)
    MSE_H0.append(np.sqrt(mean_squared_error(y_test, predictions_H0)))

p_value2 = np.mean(np.array(MSE_H0) < np.array(MSE_H1))
print(p_value2)
#%%
h_statistic, p_value_h = stats.kruskal(MSE_H1, MSE_H0)
print(f'Kruskal-Wallis H test statistic: {h_statistic}')
print(f'P-value: {p_value_h}')

# %%
################################# Final Model RMSE #####################################
data = pd.read_csv("/Users/exrt/Desktop/Reg_dataset.csv")
scaler = MinMaxScaler()

X = data.drop('shares', axis=1)
X = scaler.fit_transform(X)
y = data['shares']
y = np.log(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
dtrain = DMatrix(X_train, label=y_train)
dtest = DMatrix(X_test, label=y_test)

base_score_value = y_train.mean()
Final= train(params, dtrain, num_boost_round=300)
predictions = Final.predict(dtest)
RMSE = np.sqrt(mean_squared_error(y_test, predictions))
print(RMSE)

feature_importances = Final.get_score(importance_type='weight')

sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]

top_features = [feature for feature, _ in sorted_features]
print(top_features)

feature_names = data.columns
mapped_feature_names = [feature_names[int(f[1:])] for f in top_features]
print(mapped_feature_names)

#%%
Final.get_dump()
Final.get_params()
# %%
importances = [importance for _, importance in sorted_features]

plt.figure(figsize=(10, 6))
plt.barh(mapped_feature_names, importances, color='lightgrey')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis() 
plt.show()
# %%
# plt.figure(figsize=(500, 200))
# xgb.plot_tree(Final, num_trees=len(Final.get_dump()) - 1)
# plt.savefig("/Users/exrt/Desktop/figure_1.png", dpi=500, bbox_inches='tight')
# plt.savefig("/Users/exrt/Desktop/figure_1.pdf", bbox_inches='tight')
# plt.close() 


# %%
