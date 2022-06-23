from collections import defaultdict
import statistics,math
d = defaultdict(list)

with open('inf_time') as f:
#with open('time_taken') as f:
#with open('read_time') as f:
    lines = f.readlines()
    for l in lines:
        x = l.split()
        d[x[0].split(':')[0]].append(math.ceil(float(x[2])*1000))

a = [statistics.mean(x) for k,x in d.items()]
b = [statistics.median(x) for k,x in d.items()]
message=''
for i in range(len(a)):
    message += f'{a[i]},{b[i]} '

print(message)
