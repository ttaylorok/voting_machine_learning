import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

voting = pd.read_csv('countypres_2000-2016.csv').dropna()
voting['FIPS'] = voting['FIPS'].astype(int)
voting = voting[voting['year'] == 2016]
vg = voting.groupby(['state','FIPS','party'])['candidatevotes'].sum()
vgu = vg.unstack()
vgu.reset_index(inplace = True)
vgu['party'] = np.where(vgu['democrat'] > vgu['republican'], 'democrat', 'republican')
vgu['FIPS2'] = np.where(vgu['FIPS'] < 10000, '0'+vgu['FIPS'].astype(str), vgu['FIPS'].astype(str))

r = pd.read_csv('restaurants_with_counties.csv', dtype = {'county' : str})
r = r[r['name'] != 'none']

rg = r.groupby('county')['county'].count()
m = pd.merge(rg,vgu,left_index=True,right_on='FIPS2')
s = m.sort_values('county', ascending = False)

r['num_restaurants'] = r.groupby('county')['county'].transform('count')
r['nr_inv'] = 1 / r['num_restaurants']*100
r2 = r[r['num_restaurants'] >= 50]

r2['num_name'] = r2.groupby('name')['name'].transform('count')
r2['is_chain'] = np.where(r2['num_name'] > 3, 'yes', 'no')
r2['name_rank'] = r2['num_name'].rank(method = 'dense', ascending = False)

r2['num_cuisine'] = r2.groupby('cuisine_new')['cuisine_new'].transform('count')
r2['cuisine_rank'] = r2['num_cuisine'].rank(method = 'dense', ascending = False)

r2['num_ff'] = r2.groupby(['fastfood','name'])['name'].transform('count')
r2['ff_rank'] = r2.groupby('fastfood')['num_ff'].rank(method = 'dense', ascending = False)

#r2[r2['fastfood']=='no'].sort_values(by='ff_rank', ascending=True)[['name','ff_rank']]

r3 = r2.groupby(['county','is_chain'])['is_chain'].count()

# fastfood rank
r4 = r2[(r2['ff_rank'] <=10) & (r2['fastfood'] == 'yes')].groupby(['county','name'])['nr_inv'].sum()
r4u = r4.unstack()
#r4u.drop(columns = ['Dunkin\' Donuts','Chick-fil-A','Chipotle'], inplace = True)

# non-fastfood rank
r5 = r2[(r2['ff_rank'] <=10) & (r2['fastfood'] == 'no')].groupby(['county','name'])['nr_inv'].sum()
r5u = r5.unstack()
#r5u.drop(columns = ['Denny\'s','Waffle House'], inplace = True)

# cuisine rank
r6 = r2[r2['cuisine_rank'] <= 10].groupby(['county','cuisine_new'])['nr_inv'].sum()
r6u = r6.unstack()

r7 = r2.groupby(['county','is_chain'])['is_chain'].count()
r7u = r7.unstack()
r7u['perc_chain'] = r7u['yes'] / 100
r7u.drop(columns = ['no','yes'], inplace=True)

rm1 = pd.merge(r4u, r5u, on = 'county')
rm2 = pd.merge(rm1, r6u, on = 'county')
rm3 = pd.merge(rm2, r7u, on = 'county')

import plotly.graph_objects as go
fig = go.Figure()
for c in r4u.columns:
    fig.add_trace(go.Box(
                    y=r4u[c],
                    name=c,
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=1
))
fig.update_layout(showlegend=False, margin={"r":0,"t":0,"l":15,"b":15})
fig.update_xaxes(tickangle=90)
fig.write_html('box_plot.html')


m = pd.merge(vgu, rm3, left_on = 'FIPS2', right_on = 'county').fillna(value=0)

training = m.sample(frac=0.5, random_state = 753).fillna(value=0)
validation = m.drop(index = training.index).fillna(value=0)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(m[m.columns[6:]],
                                                    m['party'],
                                                    test_size=0.5,
                                                    random_state=0)

bl = m.groupby('party')['party'].count()
print('baseline: %f' % (bl.max()/len(m.index)))


#r['num_cuisine'] = r.groupby('cuisine_new')['cuisine_new'].transform('count')

# %% fit model
from sklearn.ensemble import RandomForestClassifier

# initialize model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_pred_rf_correct = np.where(y_pred_rf != y_test,0,1)
a = np.sum(y_pred_rf_correct)
print('accuracy rf: %f' % (a/len(y_pred_rf)))

i = rf.feature_importances_
#print(i)

# %% cross validation
import sklearn
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
# param_grid = {
#     #'bootstrap': [True],
#     'max_depth': [2,4],
#     'max_features': [2, 6, 10],
#     'min_samples_leaf': [2, 4],
#     # 'min_samples_split': [4,6,8],
#     # 'n_estimators': [100,200]
# }
# x = ('max_depth = 5')
# y = {'max_depth' : 5}

# def grid_cv(param_grid):
#     lst = list(param_grid.keys())
#     for i in np.arange(len(lst)):
#         print(lst[i])
#         #print(param_grid[lst[i]])
#         for j in param_grid[lst[i]]:
#             #print(lst[i],j)
#             for k in np.arange(i,len(param_grid)):
#                 for l in param_grid[lst[k]]:
#                     print(lst[k],l)
                
#     return 
# grid_cv(param_grid)  

param_grid = {
    #'bootstrap': [True],
    'max_depth': [2,4,8],
    'max_features': [5, 10, 15, 20],
    #'min_samples_leaf': [3, 7],
    'min_samples_split': [2,8,16],
    #'n_estimators': [50,100]
}  

import numpy as np
def grid_cv(param_grid):
    out = []
    cur = {}        
    def grid_cv_recur(param_grid,lvl):
        if lvl == len(param_grid):
            out.append(cur.copy())
            return
        first = list(param_grid.keys())[lvl]
        for i in param_grid[first]:
            cur[first] = i
            #print('  '*lvl,first,lvl,i)
            grid_cv_recur(param_grid,lvl+1)
    grid_cv_recur(param_grid,0) 
    return out  

x = grid_cv(param_grid)

dfv = pd.DataFrame(x)

from sklearn.model_selection import cross_validate
#scores_all = []
#dfv['scores_all'] = 0
dfv['scores_mean'] = 0
for s,r in enumerate(x):
    r2 = RandomForestClassifier(**r)
    r2.fit(X_train,y_train)
    y_pred = r2.predict(X_test)
    print(r2.score(X_test,y_test))
    a = sklearn.metrics.accuracy_score(y_test, y_pred)
    
    scores = cross_validate(r2, X_train, y_train, cv=5)
    #scores_all.append(scores['test_score'])
    #dfv.loc[s,'scores_all'] = [scores['test_score']]
    dfv.loc[s,'scores_mean'] = scores['test_score'].mean()
    #print(scores['test_score'])
    
    #print(a)

# %% old   

RandomForestClassifier(**y)
# Create a based model
rf2 = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = RandomForestClassifier(),
                           param_grid = param_grid,)
                           #scoring = ['roc_auc'],
                           # cv = 2,
                           # verbose = 10,
                           #param_grid={'max_depth': [1,2]},
                           # n_jobs=None)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
#grid_search.best_params_
grid_search.cv_results_

# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy

best_grid = grid_search.best_estimator_
#grid_accuracy = evaluate(best_grid, X_test, y_test)
#p = best_grid.predict(X_test)

y_pred_rfo = best_grid.predict(X_test)
y_pred_rfo_correct = np.where(y_pred_rfo != y_test,0,1)
aa = np.sum(y_pred_rfo_correct)
print('accuracy rfo: %f' % (aa/len(y_pred_rfo)))

# %% naive bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
y_pred_gnb_correct = np.where(y_pred_gnb != y_test,0,1)
b = np.sum(y_pred_gnb_correct)
print('accuracy gnb: %f' % (b/len(y_pred_gnb)))

# %% unsupervised neural networks
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)

nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                      hidden_layer_sizes=(5, 2), random_state=1)

nn.fit(scaler.transform(X_train), y_train)
y_pred_nn = nn.predict(scaler.transform(X_test))
y_pred_nn_correct = np.where(y_pred_nn != y_test,0,1)
c = np.sum(y_pred_nn_correct)
print('accuracy nn: %f' % (c/len(y_pred_nn)))

# %% logistic regression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_pred_lr_correct = np.where(y_pred_lr != y_test,0,1)
d = np.sum(y_pred_lr_correct)
print('accuracy lr: %f' % (d/len(y_pred_lr)))
