import random
from tqdm import tqdm

results = open("submission_files/topk_ids.csv")
result_lines = results.readlines()

best_outlines = []
best_numclasses = 0
patience_counter = 0

rand_seed = 0
while True:
    out_lines = ["Id,Predicted"]
    out_dict = {i: 0 for i in range(64500)}

    # random.seed(7231970)
    # random.seed(12251997)

    random.seed(rand_seed)
    indices = random.sample(range(0, len(result_lines)), len(result_lines))
    for index in indices:
        line = result_lines[index].replace(".jpg", '')
        vals = line.split(",")

        top_k = [int(x) for x in vals[1:]]

        for i in top_k:
            if out_dict[i] < 10:
                out_dict[i] += 1
                out_lines.append(f"{vals[0]},{i}")
                break
        else:
            out_lines.append(f"{vals[0]},{top_k[0]}")

    numclasses = len(set([key for key, value in out_dict.items() if value > 0]))
    if numclasses > best_numclasses:
        best_numclasses = numclasses
        best_outlines = out_lines
        patience_counter = 0
    else:
        patience_counter += 1

    print(f"Run {rand_seed}: {numclasses} | Best: {best_numclasses} | Patience: {patience_counter}")
    if patience_counter >= 1000:
        break

    rand_seed += 1


print(best_numclasses)
out_lines = '\n'.join(best_outlines)
out_file = open("out8_naive_postpro.csv", 'w')
out_file.write(out_lines)
out_file.close()
