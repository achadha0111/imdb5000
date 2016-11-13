import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr

import os
import pickle

train = [];


path_to_file = os.getcwd()+"movie_metadata.csv"
train = pd.read_csv(path_to_file, header=0)
gross = pd.DataFrame({"gross":train["gross"], "log(price + 1)":np.log1p(train["gross"])})
train["gross"] = np.log1p(train["gross"])
numeric_features = train.dtypes[train.dtypes != "object"].index
skewed_features = train[numeric_features].apply(lambda x: skew(x.dropna()))
skewed_features = skewed_features[skewed_features > 0.75]
skewed_features = skewed_features.index

train[skewed_features] = np.log1p(train[skewed_features])
train = train.dropna()
train = train.drop([ 'color', 'language', 'imdb_score', 'num_user_for_reviews', 'country', 'title_year', 'cast_total_facebook_likes', 'facenumber_in_poster', 'imdb_score', 'num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes', 'actor_2_name', 'actor_1_facebook_likes', 'num_voted_users', 'movie_title', 'actor_3_name', 'plot_keywords', 'movie_imdb_link', 'content_rating', 'actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes'], axis=1)
print train.head(n=5)
train = pd.get_dummies(train, prefix=['genres','actor_1_name', 'director_name'], columns=['genres', 'actor_1_name', 'director_name'])
	#train = train.fillna(train.mean())
print train.shape








X_train = train[:train.shape[0]]

y = train.gross

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, SGDRegressor
from sklearn.model_selection import cross_val_score


# model = sklearn.linear_model.LinearRegression(copy_X=True)


"""def rmse_cv(model):
	rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5))
	return(rmse)
	

model_ridge = Ridge()
alphas = [0.00005, 0.00015, 0.00045, 0.0135]
cv_ridge = [rmse_cv(SGDRegressor(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show() """

from sklearn import linear_model

alf = linear_model.SGDRegressor(shuffle=True, n_iter=200)
alf.fit(X_train, y)
d = {}
for key in range(5, 3835):
	#if key not in ['director_name', 'genres', 'actor_1_name', 'budget', 'gross']:
	d[key] = 0
director_name = "James Cameron"
genre = "Action"
lead = "Sam Worthington"
budget = np.log1p(10000000)
print budget
d2 = {'director_name' : pd.Series(director_name, index=[1]),
     'genres' : pd.Series(genre, index=[1]),
     'actor_1_name': pd.Series(lead, index=[1]),
     'budget': pd.Series(budget, index=[1])}
d.update(d2)
df = pd.DataFrame(d)

df = pd.get_dummies(df, prefix=['genres','actor_1_name', 'director_name'], columns=['genres', 'actor_1_name', 'director_name'])
print df
#print df.set_value(1, [5::3834], 0)
prediction = alf.predict(df)
print alf.coef_
print map(abs, prediction)









