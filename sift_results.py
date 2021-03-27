results = open("submission_files/topk_ids.csv")
result_lines = results.readlines()

out_lines = ["Id,Predicted"]
for line in result_lines:
    line = line.replace(".jpg", '')
    vals = line.split(",")

    out_lines.append(f"{vals[0]},{vals[1]}")

out_lines = '\n'.join(out_lines)

out_file = open("submission_files/out3bw.csv", 'w')
out_file.write(out_lines)
out_file.close()