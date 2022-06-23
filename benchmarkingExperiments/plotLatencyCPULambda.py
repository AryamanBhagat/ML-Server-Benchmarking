#import
import subprocess
import os
import sys
import subprocess
import time
#declare parameters
connections = [16]
duration = [3600]
threads = [16]
rate = [1000]
names = ["16t16c1000rpsLambda"]

count = len(connections)

#move into data folder.

os.chdir('data')


files = ""

for i in range(count):
    
    files += "./" + names[i] + ".out "
    #print("wrkRunning")
    output = subprocess.check_output(['wrk', '-t'+str(threads[i]), '-c'+str(connections[i]), '-d' +str(duration[i]) + 's', '-R'+str(rate[i]), '-L', '-s', '../postCommand2.lua', 'https://h0fx1aj2m8.execute-api.ap-south-1.amazonaws.com/Prod/yolo'])
    #print(output)
    output = output.decode('UTF-8')
    with open(names[i] + '.out', 'w') as f:
        print(output, file = f)
    #time.sleep(600)
print(files)   

os.system("hdr-plot --output 16t16cplotLambda1000rps.png --title \"16 thread 16 connection latency plot\" "+files)

