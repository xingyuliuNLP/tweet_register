#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plot


df = pd.read_csv("../projetAA/corpus_annote.csv", sep="\t")
df_ = df[["Familier", "Courant", "Soutenu", "Poubelle"]].values.tolist()
registres = [predict.index(max(predict)) for predict in df_]

familier = registres.count(0)
courant = registres.count(1)
soutenu = registres.count(2)
poubelle = registres.count(3)

title = 'Répartition des registres en première place'

plot.hist(registres,
          range = (0, 4),
          bins = 4,
          color = 'orange',
          edgecolor="yellow")

plot.xlabel('valeurs')
plot.ylabel('nombres')
plot.title(title)
plot.savefig(os.path.join('../{}.png'.format(title.replace(" ", "_").replace("\n", "_"))), dpi=300, format='png', bbox_inches='tight')
plot.show()