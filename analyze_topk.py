import pandas as pd

output = pd.read_csv("submission_files/out23_spp4.csv")

distinct_classes = set(output["Predicted"])
print(len(distinct_classes))