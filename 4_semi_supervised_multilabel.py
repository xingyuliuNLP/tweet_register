#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.models import load_model

from normalization import *

# select reliable text
def transform_prediction(predictions):
    predictions[predictions>=0.5] = 1
    predictions[predictions<0.5] = 0
    return predictions

# model construction
def keras_RN(X_train,
             y_train,
             nb_classes,
             hidden_layer_sizes=10,
             dropout=.5,
             epochs=50,
             verbose=1,
             input_dim=2):
    """
    input: X_train, Y_train
    ouput: predict values [[proba1, proba2, proba3, proba4]]
    """
    model = Sequential()
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_sizes, input_dim=input_dim, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='CategoricalCrossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=100, shuffle=True )
    return model


def get_reliable_data_inter_classifier(y_pred, y_train, X_train):
    """
    """
    y_to_add = list()
    x_to_add = list()
    for i in range(len(y_pred)):
        if sum(y_pred[i]) != 0:
            y_to_add.append(y_pred[i])
            x_to_add.append(X_predict[i])
    if len(y_to_add) != 0 :
        y_train = np.append(y_train, y_to_add, axis=0)
        X_train = np.append(X_train, x_to_add, axis=0)
    return X_train, y_train, len(y_to_add)


if __name__ == "__main__":

    # tagged data
    train_graine = get_data("1_data/4_features_csv/features_500.csv")

    # data to tag automatically
    predict_all = get_data("1_data/4_features_csv/features_20000.csv", sep='\t')
    predict_all.iloc[:, 1:2]

    # vectorized text
    X = train_graine.iloc[:,6:15].to_numpy()
    X_predict = predict_all.iloc[:, 2:].to_numpy()

    # label
    y = train_graine[["Familier", "Courant", "Soutenu", "Poubelle"]].to_numpy()

    # percentage to binary
    y[y > 0.25] = 1
    y[y <= 0.25] = 0

    #RN parameter
    nb_classes = 4
    hidden_layer_sizes = 10
    dropout = .2
    epochs = 30 #100
    verbose = 1
    input_dim = len(X[0])
    #model parameter
    seuil_reliability = 0.7

    nbr_iter = 8
    set_features = list()
    set_fold = list()
    set_mse = list()
    set_mae = list()
    set_accuracy = list()
    set_cp_text_added = list()

    fld = KFold(n_splits=5, shuffle=True, random_state=42)

    fold = 0

    for train, test in fld.split(X=X, y=y):

        all_id = list(train) + list(test)
        train_index = round((len(all_id)/5) * 3)
        test_index = int(round((len(all_id)/5) / 2, 0))

        train = all_id[:train_index]
        test = all_id[-(test_index*2):-test_index]
        dev = all_id[-test_index:]

        X_train = X[train]
        X_test = X[test]
        X_dev = X[dev]

        # familier
        y_train = y[train]
        y_test = y[test]
        y_dev = y[dev]

        print("-----"*10)
        print("\t\tfold : {}/5".format(fold+1))
        print("-----"*10)

        for iter in range(nbr_iter):

            print("\n\titer : {}/{}".format(iter+1, nbr_iter))
            set_fold.append(fold)

            if iter < nbr_iter-1:

                print("-"*70)
                print("\tfold : {}/5 \titer : {}/{}".format(fold+1, iter+1, nbr_iter))
                print("-"*70)

                predictor = keras_RN(X_train,
                             y_train,
                             nb_classes,
                             hidden_layer_sizes,
                             dropout,
                             epochs,
                             verbose,
                             input_dim)

                y_pred = predictor.predict(X_predict)
                y_pred = transform_pred(np.array(y_pred), seuil_reliability)
                y_pred_test = predictor.predict(X_test)
                y_pred_test = transform_pred(np.array(y_pred_test), seuil_reliability)

                set_mse.append(mean_squared_error(y_pred_test, y_test))
                set_mae.append(mean_absolute_error(y_pred_test, y_test))

                X_train, y_train, cp_text_added = get_reliable_data_inter_classifier(y_pred, y_train, X_train)

                set_cp_text_added.append(cp_text_added)

            if iter == nbr_iter-1:

                print("-"*70)
                print("\tfold : {}/5 \titer : {}/{}".format(fold+1, iter+1, nbr_iter))
                print("-"*70)

                predictor = keras_RN(X_train,
                             y_train,
                             nb_classes,
                             hidden_layer_sizes,
                             dropout,
                             epochs,
                             verbose,
                             input_dim)

                y_pred_test = predictor.predict(X_dev)
                y_pred_test = transform_pred(np.array(y_pred_test))

    #             set_accuracy.append()
    #             set_cp_text_added.append
                set_mse.append(mean_squared_error(y_pred_test, y_dev))
                set_mae.append(mean_absolute_error(y_pred_test, y_dev))

        fold+=1

    predictor.save("semi_supervised_multilabels-expertFeatures_RN.h5")

    dict_out = dict()
    dict_out["fold"] = set_fold
    dict_out["text added"] = set_cp_text_added
    dict_out["mse"] = set_mse
    dict_out["mae"] = set_mae
    dict_out["accuracy"] = set_accuracy

    df = pd.DataFrame.from_dict(dict_out, orient="index").fillna(0)
    name_to_save = "semi_supervised_multilabels-expertFeatures_epoch-{}_seuil-{}_iter-{}".format(epochs, str(seuil_reliability).replace(".","-"), nbr_iter)
    df.to_csv("{}.csv".format(name_to_save), sep="\t", encoding="utf-8")