#!/anaconda3/bin/python3.7
# -*- coding: UTF-8 -*-
import pandas as pd
import os
from normalization import *

# Tag data by talismane
def tag_talismane(path_csv="1_data/3_normalized_data/normalized_data.csv"):
    """
    input: csv of normalized data
    output: str(data annotated in CONLL-U format)
    """
    conll_output = ""
#     df = get_data(path_csv)
    df = pd.read_csv(path_csv, encoding="utf-8", sep=";")
    for text in df["Tweet"].values.tolist():
        cmd = f"echo {text} > temp.txt"
        os.system(cmd)
        os.system("java -Xmx1G -Dconfig.file=talismane-fr-5.0.4.conf -jar talismane-core-5.1.2.jar --analyse --sessionId=fr --encoding=UTF8 --inFile=temp.txt --outFile=temp.tal")
        # Convert Talismane into a format usable by Grew (a sed script)
        with open("temp.tal", "r", encoding="utf-8") as f:
            conll_output += f.read()
    return conll_output

# write tagged data in a file
def conll_to_file(conll_data, path="1_data/4_data_tagged/data_tagged.conll"):
    """
    input: str(conll_data)
    """
    with open(path, "w", encoding="utf-8") as file_out:
        file_out.write(conll_data)