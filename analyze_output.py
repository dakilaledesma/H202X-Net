import pandas as pd

output = pd.read_csv("out_naive_postpro.csv")

distinct_classes = set(output["Predicted"])
print(len(distinct_classes))