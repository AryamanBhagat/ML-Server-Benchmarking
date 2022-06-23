#import
import subprocess
import os
import sys
import subprocess
import time
#declare parameters
connections = [16, 16, 16, 16, 16]
duration = [300, 300,300, 300, 300]
threads = [16, 16, 16, 16, 16]
rate = [10, 25, 50, 75, 100]
names = ["hazard_vest_16t16c10rps", "hazard_vest_16t16c25rps", "hazard_vest_16t16c50rps", "hazard_vest_16t16c75rps", "hazard_vest_16t16c100rps"]

count = len(connections)

#move into data folder.

os.chdir('data')


files = ""

for i in range(count):
    files += "./" + names[i] + ".out "
    output = subprocess.check_output(['wrk', '-t'+str(threads[i]), '-c'+str(connections[i]), '-d' +str(duration[i]) + 's', '-R'+str(rate[i]), '-L', '-s', '../postCommand1.lua', 'http://172.31.22.216:5000/hazard'])
    #print(output)
    output = output.decode('UTF-8')
    with open(names[i] + '.out', 'w') as f:
        print(output, file = f)
    time.sleep(10)
print(files)   

os.system("hdr-plot --output hazard_vest_16t16cplot.png --title \"hazard vest 16 thread 16 connection latency plot\" "+files)

