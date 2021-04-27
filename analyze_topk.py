import pandas as pd

output = pd.read_csv("submission_files/out33-srx50bw_b-l.csv")

distinct_classes = set(output["Predicted"])
print(len(distinct_classes))