import pandas as pd 

file = "/home/asurite.ad.asu.edu/zhaonan2/Blind_VQA/verl/data/blind500/train.parquet"
data = pd.read_parquet(file)
print(data.head())
print(data.iloc[0].keys())
print(data.iloc[0]["prompt"])
print(len(str(data.iloc[0]["prompt"])))