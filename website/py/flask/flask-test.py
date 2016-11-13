import flask
#import os

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper

import numpy as np
import pandas as pd
#import seaborn as sns
#import matplotlib

#import matplotlib.pyplot as plt
from scipy.stats import skew
#from scipy.stats.stats import pearsonr

import os
#import pickle

train = [];


path_to_file = "/var/www/html/py/flask/movie_metadata.csv"
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

clf = linear_model.SGDRegressor(shuffle=True, n_iter=20)
clf.fit(X_train, y)
d = {}
for key in range(5, 3835):
	#if key not in ['director_name', 'genres', 'actor_1_name', 'budget', 'gross']:
	d[key] = 0

#print map(abs, prediction)



def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        print f
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

app = flask.Flask(__name__)  # specifies the root

# When the index of the route argument is requested, the function index() is called, and will
# return some value
# It's convention to name the function the same as the directory specified by the route


@app.route('/flask/<genre>/<actorname>/<budget>/<directorname>', methods=['GET'])
@crossdomain(origin='*')
def flask(genre, actorname, budget, directorname):

    #return genre + actorname + budget + directorname
    budgetnormalize = np.log1p(int(budget))

    # print budget
    d2 = {'director_name' : pd.Series(directorname, index=[1]),
         'genres' : pd.Series(genre, index=[1]),
         'actor_1_name': pd.Series(actorname, index=[1]),
         'budget': pd.Series(budgetnormalize, index=[1])}
    d.update(d2)
    df = pd.DataFrame(d)

    df = pd.get_dummies(df, prefix=['genres','actor_1_name', 'director_name'], columns=['genres', 'actor_1_name', 'director_name'])
    # print df
    #print df.set_value(1, [5::3834], 0)
    prediction = clf.predict(df)
    return repr(map(abs, prediction)[0])

if __name__ == "__main__":
    app.run(
    host=os.getenv('LISTEN', '0.0.0.0'),
    port=int(os.getenv('PORT', '8080'))
    )

