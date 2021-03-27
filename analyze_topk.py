import pandas as pd

output = pd.read_csv("submission_files/out3bw.csv")

distinct_classes = set(output["Predicted"])
print(len(distinct_classes))