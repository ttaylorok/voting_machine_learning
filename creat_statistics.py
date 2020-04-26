import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import cross_validate   
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

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

# models = [[10,0,0],
#            [0,10,0],
#            [0,0,10],
#            [5,5,5]]

#['Arby's', 'Burger King', 'Chick-fil-A', 'Dairy Queen', 'Five Guys', 'McDonald's', 'Sonic', 'Taco Bell']
#['Applebee's', 'Buffalo Wild Wings', 'Cracker Barrel', 'Golden Corral', 'IHOP', 'Olive Garden', 'Pizza Hut', 'Waffle House']
#['breakfast', 'burger', 'indian', 'italian', 'sandwich', 'thai', 'vietnamese']

filt = ['Arby\'s', 'Burger King', 'Chick-fil-A', 'Dairy Queen', 'Five Guys', 'McDonald\'s', 'Sonic', 'Taco Bell'] \
    + ['Applebee\'s', 'Buffalo Wild Wings', 'Cracker Barrel', 'Golden Corral', 'IHOP', 'Olive Garden', 'Pizza Hut', 'Waffle House'] \
        + ['breakfast', 'burger', 'indian', 'italian', 'sandwich', 'thai', 'vietnamese']


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

#for mm,model in enumerate(models):
#print(model)
# fastfood rank
r4 = r2[(r2['ff_rank'] <= 10) & (r2['fastfood'] == 'yes')].groupby(['county','name'])['nr_inv'].sum()
r4u = r4.unstack()
#r4u.drop(columns = ['Dunkin\' Donuts','Chick-fil-A','Chipotle'], inplace = True)

# non-fastfood rank
r5 = r2[(r2['ff_rank'] <= 10) & (r2['fastfood'] == 'no')].groupby(['county','name'])['nr_inv'].sum()
r5u = r5.unstack()
#r5u.drop(columns = ['Denny\'s','Waffle House'], inplace = True)

# cuisine rank
r6 = r2[r2['cuisine_rank'] <= 10].groupby(['county','cuisine_new'])['nr_inv'].sum()
r6u = r6.unstack()

r7 = r2.groupby(['county','is_chain'])['is_chain'].count()
r7u = r7.unstack()
r7u['perc_chain'] = r7u['yes'] / 100
r7u.drop(columns = ['no','yes'], inplace=True)

rm1 = pd.merge(r4u, r5u, on = 'county', how = 'outer')
rm2 = pd.merge(rm1, r6u, on = 'county', how = 'outer')
rm3 = pd.merge(rm2, r7u, on = 'county', how = 'outer')

import plotly.graph_objects as go
fig = go.Figure()
for c in r6u.columns:
    fig.add_trace(go.Box(
                    y=r6u[c],
                    name=c,
                    boxpoints='all',
                    jitter=0.5,
                    whiskerwidth=0.2,
                    marker_size=2,
                    line_width=1
))
fig.update_layout(showlegend=False,
                  margin={"r":0,"t":30,"l":15,"b":15},
                  yaxis_title="Presence (% of all Restaurants)",
                  title="Presence of Cuisine Types per County")
fig.update_xaxes(tickangle=90)
fig.update_yaxes(range=[0,200])
fig.write_html('box_plot_cuisines_x.html', include_plotlyjs = 'cdn')


m = pd.merge(vgu, rm3, left_on = 'FIPS2', right_on = 'county').fillna(value=0)

models = {'fast-food'     : m[m.columns[6:16]],
          'sit-down'      : m[m.columns[16:26]],
          'cuisine types' : m[m.columns[26:36]],
          'all'           : m[m.columns[6:36]]}

# training = m.sample(frac=0.5, random_state = 753).fillna(value=0)
# validation = m.drop(index = training.index).fillna(value=0)


# X_train, X_test, y_train, y_test = train_test_split(m[m.columns[6:]],
#                                                     m['party'],
#                                                     test_size=0.5,
#                                                     random_state=0)

bl = m.groupby('party')['party'].count()
print('baseline: %f' % (bl.max()/len(m.index)))




#r['num_cuisine'] = r.groupby('cuisine_new')['cuisine_new'].transform('count')

# %% fit model
from sklearn.ensemble import RandomForestClassifier

# initialize model

params_rf = {
    #'bootstrap': [True],
    'max_depth': [4,8,16],
    #'max_features': [10, 40, 100],
    #'min_samples_leaf': [3, 7],
    'min_samples_split': [2,8,16],
    #'min_impurity_decrease' : [0.1,0.05,0.02,0.01]
    'n_estimators': [10,40,100]
} 

param_grid_rf = grid_cv(params_rf)

df_pg_rf = pd.DataFrame(param_grid_rf)

scores_all = np.ones([len(param_grid_rf), len(models)])
for j,model in enumerate(models.keys()):
    X_train, X_test,\
        y_train, y_test = train_test_split(models[model],
                                            m['party'],
                                            test_size=0.5,
                                            random_state=0)
   
    for i,pg in enumerate(param_grid_rf):
        rf = RandomForestClassifier(**pg)
        rf.fit(X_train, y_train)
        
        scores = cross_validate(rf, X_train, y_train, cv=5)
        print(scores['test_score'])
        scores_all[i,j] = scores['test_score'].mean()
        
dfs = pd.DataFrame(scores_all, columns=list(models.keys()))
df_pg_rf2 = pd.concat([df_pg_rf, dfs], axis=1)

df_pg_rf2.max()


#df_pg_rf2['max_score'] = df_pg_rf2['all'].max()
mr = max_rows = df_pg_rf2[df_pg_rf2['all'] == df_pg_rf2['all'].max()].iloc[0,:]
na = dict(mr[:-len(models)].astype(int))
rf2 = RandomForestClassifier(**na)
rf2.fit(X_train, y_train)
y_pred = rf2.predict(X_test)
accuracy_score(y_test, y_pred)

fig = go.Figure(
    data=[go.Table(
    header=dict(values=['max depth',
                        'min samples split',
                        'n estimators'] \
                + list(df_pg_rf2.columns[3:])),
    cells=dict(values=[df_pg_rf2['max_depth'],
                       df_pg_rf2['min_samples_split'],
                       df_pg_rf2['n_estimators'],
                       df_pg_rf2['fast-food'].round(4),
                       df_pg_rf2['sit-down'].round(4),
                       df_pg_rf2['cuisine types'].round(4),
                       df_pg_rf2['all'].round(4)]))
                     ])
fig.update_layout(margin = {"r":20,"t":0,"l":0,"b":0})
fig.show()
fig.write_html('plotly_table_rf.html', include_plotlyjs = 'cdn')

# y_pred_rf = rf.predict(X_test)
# y_pred_rf_correct = np.where(y_pred_rf != y_test,0,1)
# a = np.sum(y_pred_rf_correct)
# print('accuracy rf: %f' % (a/len(y_pred_rf)))

#i = rf.feature_importances_
#print(i)

# %% cross validation
# import sklearn
# from sklearn.model_selection import GridSearchCV
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






# %% fit model
from sklearn.tree import DecisionTreeClassifier

# initialize model

params_dc = {
    #'bootstrap': [True],
    'max_depth': [4,8,16],
    #'max_features': [10, 40, 100],
    #'min_samples_leaf': [3, 7],
    'min_samples_split': [2,8,16],
    'min_impurity_decrease' : [0.1,0.05,0.02,0.01]
    #'n_estimators': [10,40,100]
} 

param_grid_dc = grid_cv(params_dc)

df_pg_dc = pd.DataFrame(param_grid_dc)

scores_all = np.ones([len(param_grid_dc), len(models)])
for j,model in enumerate(models.keys()):
    X_train, X_test,\
        y_train, y_test = train_test_split(models[model],
                                            m['party'],
                                            test_size=0.5,
                                            random_state=0)
   
    for i,pg in enumerate(param_grid_dc):
        dc = DecisionTreeClassifier(**pg)
        dc.fit(X_train, y_train)
        
        scores = cross_validate(dc, X_train, y_train, cv=5)
        print(scores['test_score'])
        scores_all[i,j] = scores['test_score'].mean()
        
dfs = pd.DataFrame(scores_all, columns=list(models.keys()))
df_pg_dc2 = pd.concat([df_pg_dc, dfs], axis=1)

df_pg_dc2.max()

X_train, X_test,\
    y_train, y_test = train_test_split(models['cuisine types'],
                                        m['party'],
                                        test_size=0.5,
                                        random_state=0)

mr = df_pg_dc2[df_pg_dc2['cuisine types'] == df_pg_dc2['cuisine types'].max()].iloc[0,:]
na = dict(mr[:-len(models)])
dc2 = DecisionTreeClassifier(max_depth=4, min_samples_split=2, min_impurity_decrease=0.5)
dc2.fit(X_train, y_train)
y_pred = dc2.predict(X_test)
accuracy_score(y_test, y_pred)

fig = go.Figure(
    data=[go.Table(
    header=dict(values=['max depth',
                        'min samples split',
                        'min impurity decrease'] \
                + list(df_pg_dc2.columns[3:])),
    cells=dict(values=[df_pg_dc2['max_depth'],
                       df_pg_dc2['min_samples_split'],
                       df_pg_dc2['min_impurity_decrease'],
                       df_pg_dc2['fast-food'].round(4),
                       df_pg_dc2['sit-down'].round(4),
                       df_pg_dc2['cuisine types'].round(4),
                       df_pg_dc2['all'].round(4)]))
                     ])
fig.update_layout(margin = {"r":20,"t":0,"l":0,"b":0})
fig.show()
fig.write_html('plotly_table_dc.html', include_plotlyjs = 'cdn')

# %% write to html
        



    
    #print(a)
    
# %% PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pca = PCA()
r4 = r2[(r2['ff_rank'] <= 10) & (r2['fastfood'] == 'yes')].groupby(['county','name'])['nr_inv'].sum()
r4u = r4.unstack()
#r4us = StandardScaler().fit_transform(r4u.fillna(value=0))
r4us = StandardScaler().fit_transform(m[m.columns[6:]])

pca_scores = pd.DataFrame(pca.fit_transform(r4us))
#pca_scores.index = r4u.index
# pca_scores_county = pd.concat([r4u.index,
#                                pd.DataFrame(pca_scores)],
#                                axis=1)

# m = pd.merge(vgu, pca_scores,
#              left_on = 'FIPS2', right_index = True)


from sklearn.model_selection import train_test_split
X_train, X_test,\
    y_train, y_test = train_test_split(pca_scores[pca_scores.columns[0:10]],
                                                    m['party'],
                                                    test_size=0.5,
                                                    random_state=0)

# initialize model
rfp = RandomForestClassifier()
rfp.fit(X_train, y_train)

y_pred_rfp = rfp.predict(X_test)
y_pred_rfp_correct = np.where(y_pred_rfp != y_test,0,1)
a = np.sum(y_pred_rfp_correct)
print('accuracy rfp: %f' % (a/len(y_pred_rfp)))

i = rf.feature_importances_


# pca = PCA()
# pca_comps = pca.fit(r4us)
# expl_var = pca.explained_variance_ratio_
# sing_values = pca.singular_values_

# #pca_scores = (pca_comps.components_)

# pca_scores = pca.fit_transform(r4us)

# %% old   

# RandomForestClassifier(**y)
# # Create a based model
# rf2 = RandomForestClassifier()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = RandomForestClassifier(),
#                            param_grid = param_grid,)
#                            #scoring = ['roc_auc'],
#                            # cv = 2,
#                            # verbose = 10,
#                            #param_grid={'max_depth': [1,2]},
#                            # n_jobs=None)

# # Fit the grid search to the data
# grid_search.fit(X_train, y_train)
# #grid_search.best_params_
# grid_search.cv_results_

# # def evaluate(model, test_features, test_labels):
# #     predictions = model.predict(test_features)
# #     errors = abs(predictions - test_labels)
# #     mape = 100 * np.mean(errors / test_labels)
# #     accuracy = 100 - mape
# #     print('Model Performance')
# #     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
# #     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
# #     return accuracy

# best_grid = grid_search.best_estimator_
# #grid_accuracy = evaluate(best_grid, X_test, y_test)
# #p = best_grid.predict(X_test)

# y_pred_rfo = best_grid.predict(X_test)
# y_pred_rfo_correct = np.where(y_pred_rfo != y_test,0,1)
# aa = np.sum(y_pred_rfo_correct)
# print('accuracy rfo: %f' % (aa/len(y_pred_rfo)))

# %% naive bayes
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_gnb = gnb.predict(X_test)
# y_pred_gnb_correct = np.where(y_pred_gnb != y_test,0,1)
# b = np.sum(y_pred_gnb_correct)
# print('accuracy gnb: %f' % (b/len(y_pred_gnb)))

from sklearn.naive_bayes import GaussianNB

# initialize model

params_gnb = {
    'var_smoothing' : [1e-9,0.001,0.1,1,10]
} 

param_grid_gnb = grid_cv(params_gnb)

df_pg_gnb = pd.DataFrame(param_grid_gnb)

scores_all = np.ones([len(param_grid_gnb), len(models)])
for j,model in enumerate(models.keys()):
    X_train, X_test,\
        y_train, y_test = train_test_split(models[model],
                                            m['party'],
                                            test_size=0.5,
                                            random_state=0)
   
    for i,pg in enumerate(param_grid_gnb):
        gnb = GaussianNB(**pg)
        gnb.fit(X_train, y_train)
        
        scores = cross_validate(gnb, X_train, y_train, cv=5)
        print(scores['test_score'])
        scores_all[i,j] = scores['test_score'].mean()
        
dfs = pd.DataFrame(scores_all, columns=list(models.keys()))
df_pg_gnb2 = pd.concat([df_pg_gnb, dfs], axis=1)
print(df_pg_gnb2)

df_pg_gnb2.max()

X_train, X_test,\
    y_train, y_test = train_test_split(models['cuisine types'],
                                        m['party'],
                                        test_size=0.5,
                                        random_state=0)

mr = df_pg_gnb2[df_pg_gnb2['all'] == df_pg_gnb2['all'].max()].iloc[0,:]
na = dict(mr[:-len(models)])
gnb2 = GaussianNB()
gnb2.fit(X_train, y_train)
y_pred = gnb2.predict(X_test)
accuracy_score(y_test, y_pred)

fig = go.Figure(
    data=[go.Table(
    header=dict(values=['var smoothing'] \
                + list(df_pg_gnb2.columns[1:])),
    cells=dict(values=[df_pg_gnb2['var_smoothing'],
                       df_pg_gnb2['fast-food'].round(4),
                       df_pg_gnb2['sit-down'].round(4),
                       df_pg_gnb2['cuisine types'].round(4),
                       df_pg_gnb2['all'].round(4)]))
                     ])
fig.update_layout(margin = {"r":20,"t":0,"l":0,"b":0})
fig.show()
fig.write_html('plotly_table_gnb.html', include_plotlyjs = 'cdn')

import plotly.graph_objects as go
#animals=['giraffes', 'orangutans', 'monkeys']

fig = go.Figure(data=[
    go.Bar(name='Democrat', x=m.columns[26:36], y=gnb2.theta_[0], error_y=1),
    go.Bar(name='Republican', x=m.columns[26:36], y=gnb2.theta_[1])
])
# Change the bar mode

fig.update_layout(barmode='group',
                  margin={"r":0,"t":30,"l":0,"b":0},
                  title="Cuisine Preferences vs Voting Outcome",
                  legend=dict(x=0.8, y=0.9))
fig.update_xaxes(tickangle=-90)
fig.show()
fig.write_html("theta_output.html", include_plotlyjs = 'cdn')

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
# from sklearn.linear_model import LogisticRegression

# lr = LogisticRegression(random_state=0)
# lr.fit(X_train, y_train)
# y_pred_lr = lr.predict(X_test)
# y_pred_lr_correct = np.where(y_pred_lr != y_test,0,1)
# d = np.sum(y_pred_lr_correct)
# print('accuracy lr: %f' % (d/len(y_pred_lr)))

from sklearn.linear_model import LogisticRegression

# initialize model

params_lr = {
    'solver' :['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter' : [100,200],
    'C' : [1,0.1]
} 

param_grid_lr = grid_cv(params_lr)

df_pg_lr = pd.DataFrame(param_grid_lr)

scores_all = np.ones([len(param_grid_lr), len(models)])
for j,model in enumerate(models.keys()):
    X_train, X_test,\
        y_train, y_test = train_test_split(models[model],
                                            m['party'],
                                            test_size=0.5,
                                            random_state=0)
   
    for i,pg in enumerate(param_grid_lr):
        lr = LogisticRegression(**pg)
        lr.fit(X_train, y_train)
        
        try:
            scores = cross_validate(lr, X_train, y_train, cv=5)
            print(scores['test_score'])
            scores_all[i,j] = scores['test_score'].mean()
        except:
            scores_all[i,j] = np.nan
        
        
dfs = pd.DataFrame(scores_all, columns=list(models.keys()))
df_pg_lr2 = pd.concat([df_pg_lr, dfs], axis=1)
print(df_pg_lr2)

df_pg_lr2.max()

X_train, X_test,\
    y_train, y_test = train_test_split(models['cuisine types'],
                                        m['party'],
                                        test_size=0.5,
                                        random_state=0)

mr = df_pg_lr2[df_pg_lr2['cuisine types'] == df_pg_lr2['cuisine types'].max()].iloc[0,:]
na = dict(mr[:-len(models)])
lr2 = LogisticRegression(**na)
lr2.fit(X_train, y_train)
y_pred = lr2.predict(X_test)
accuracy_score(y_test, y_pred)

fig = go.Figure(
    data=[go.Table(
    header=dict(values=['solver',
                        'max iter',
                        'C'] + list(df_pg_lr2.columns[3:])),
    cells=dict(values=[df_pg_lr2['solver'],
                       df_pg_lr2['max_iter'],
                       df_pg_lr2['C'],
                       df_pg_lr2['fast-food'].round(4),
                       df_pg_lr2['sit-down'].round(4),
                       df_pg_lr2['cuisine types'].round(4),
                       df_pg_lr2['all'].round(4)]))
                     ])
fig.update_layout(margin = {"r":20,"t":0,"l":0,"b":0})
fig.show()
fig.write_html('plotly_table_lr.html', include_plotlyjs = 'cdn')
