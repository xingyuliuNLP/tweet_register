#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def get_results_by_folds(nbr_folds, df):
    list_finale_metrics_dev = list()
    list_text_added = list()
    list_all_metrics_test = list()
    for fold in range(nbr_folds):
        df_fold = df.loc[df['fold'] == fold]
        list_finale_metrics_dev.append(np.array(df_fold.iloc[-1:,-2:].values.tolist()[0]))
        list_all_metrics_test.append(df_fold.iloc[:,-2:].values.tolist())
    return np.array(list_finale_metrics_dev), np.array(list_text_added), np.array(list_all_metrics_test)

def get_finale_results(nbr_folds, df):
    metrics, text_added, all_metrics_test = get_results_by_folds(nbr_folds, df)
    evolution_text_added = np.sum(text_added, axis=0)/nbr_folds
    evolution_metrics_test = np.sum(all_metrics_test, axis=0)/nbr_folds
    finale_metrics_dev = np.sum(metrics, axis=0)/nbr_folds

    mse_evolution = evolution_metrics_test[:,0]
    mae_evolution = evolution_metrics_test[:,1]
    accuracy = evolution_metrics_test[:,2]
    mse_finale = finale_metrics_dev[0]
    mae_finale = finale_metrics_dev[1]

    return evolution_text_added, mse_evolution, mae_evolution, mse_finale, mae_finale, accuracy

if __name__ == "__main__":

    df = pd.read_csv("../projetAA/semi_supervised_multilabels-expertFeatures_epoch-30_seuil-0-6_iter-8.csv", sep="\t", index_col=0)

    evolution_text_added, mse_evolution, mae_evolution, mse_finale, mae_finale, accuracy = get_finale_results(5, df)

    title = "Variation des performances durant les itérations"
    fig = plt.figure()
    plt.title(title)
    x = [el for el in range(len(mae_evolution))]
    plt.xlabel("itérations")
    plt.ylabel("mesure")

    plt.plot(x, mse_evolution, color="green", label='MSE')
    plt.plot(x, mae_evolution, color="orange", label='MAE')
    plt.plot(x, accuracy, color="blue", label='Accuracy')

    plt.legend(loc='upper left')
    plt.savefig('../{}.png'.format(title.replace("\n", "").replace(" : ","_").replace(" ", "_").replace(",", "-").strip()), dpi=300, format='png', bbox_inches='tight')