#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-

import pandas as pd
import statsmodels.api as sm

tweet = pd.read_csv('1_data/4_features_csv/features_500.csv', sep=';', encoding='utf-8', index_col=0)

X, y = [], []
# see leverage of 4 features: Diversité des lemmes, Diversité des temps verbaux, Moyenne de la fréquence des mots and Root_dependency
for i in range(500):
    f = tweet.iloc[i, 10:14].to_list()
    X.append(f)
# familier/courant/soutenu/poubelle
variables = tweet.Poubelle.to_numpy()
def transform_labels(y):
    y[y>0]=1
    return y
variables = transform_labels(variables)
y.append(variables)

# Ajout d'une dimension 1 qui permet à l'algorithme de savoir où commencent les features
X = sm.add_constant(X)

# Configuration du modèle
model = sm.Logit(variables, X)
# result = model.fit(method='bfgs')
# result = model.fit(method='basinhopping')
result = model.fit(method='ncg')
result.pred_table()
result.summary()
