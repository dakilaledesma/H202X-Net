output_file = open("outputs/eb3-7-squished-bottleneck-10-cf.txt")
output_lines = output_file.readlines()[1:]

output_classes = set()
for line in output_lines:
    output_class = int(line.replace("\n", '').split(",")[1])
    output_classes.add(output_class)

print(len(output_classes))