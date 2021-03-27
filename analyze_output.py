import pandas as pd

output = open("submission_files/topk_ids.csv")
output_lines = output.readlines()

categories = set()
for line in output_lines:
    line = line.strip().split(',')
    vals = [int(v) for v in line[1:]]
    print(vals)
    
    for v in vals:
        categories.add(v)
        
print(len(list(categories)))
    