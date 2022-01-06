from matplotlib import pyplot as plt 
import json
import os
import numpy as np

with open("compile-pointwise-args.json", "r") as read_file:
    data = json.load(read_file)

datatype='float'
myshape=[25, 100]
global_workitems = 1024
iterations=10

dataset=[]

for local_workitems_per_CU in {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}:
# for local_workitems_per_CU in {1, 2, 117}:
    # print('iter=', local_workitems_per_CU)

    f = open('./temp_file.json', 'w')

    cp  = {  }  # compile_pointwise
    cp["lambda"] = "[](auto x, auto y, auto z) { return x+y+z; }"

    # inputs is a list of dict
    inputs = []
    for ii in range(4):
        # 4 identical items
        item = {"type": datatype, "lens": myshape}
        inputs.append(item)

    cp["inputs"] = inputs

    cp["global"] = global_workitems
    cp["local"] = local_workitems_per_CU
    cp["iterations"] = iterations

    outer_item={"compile_pointwise": cp}

    f.write(json.dumps( outer_item, indent=4))
    f.close()

    ##################################################
    #   run gpu-driver
    ##################################################

    stream = os.popen('~/AMDMIGraphX/build/bin/gpu-driver ~/AMDMIGraphX/temp_file.json')
    output = stream.read()

    # parse out the time (ms)
    startp = output.find("]: ")
    endp = output.find("ms", startp)
    mytime = float(output[startp+3:endp])

    print(local_workitems_per_CU, mytime)
    dataset.append((local_workitems_per_CU, mytime))
    stream.close()

dataset = sorted(dataset, key=lambda item: item[0])
print("dataset is ", dataset)

# Create a plot

fig = plt.figure()
x, y = np.array(dataset).T
print("x", x)
print('y', y)
plt.plot(x, y)
plt.xlabel('local workitems per CU')
plt.ylabel('time(ms)')
plt.title('time vs local, data shape= ' + json.dumps(myshape))

outputfile='./plot.png'
plt.savefig(outputfile)
print ('your results have been plotted as ', outputfile)