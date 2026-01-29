import matplotlib.pyplot as plt
import csv

x = []
y = []
size = []
lines = 0
with open('自主呼吸时长.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        lines += 1
        if lines == 1:
            continue
        x.append(int(row[0]))
        y.append(int(row[1]))
        size.append(int(row[2]) * 5)
print(x)
plt.scatter(y, x, s=size, alpha=0.5)

plt.savefig('自主呼吸时长.png')
plt.show()

