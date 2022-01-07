from matplotlib import pyplot as plt 
import json
import os
import numpy as np
import math

# with open("compile-pointwise-args.json", "r") as read_file:
#     data = json.load(read_file)

datatype='float'
myshape=[2500,100000]
global_workitems = 1024*16
iterations=10



dataset=[[], [], [], []]
# Create a plot

fig = plt.figure()

# loop over independent variable of interest
for global_iter in range(4):
    dataset[global_iter]=[]

    # Apparently any value larger than 1024 is illegal
    for local_workitems_per_CU in { 32, 48, 64, 128, 256, 512, 1024}:
    # for local_workitems_per_CU in {64, 128, 192, 4*64, 5*64, 6*64, 8*64, 10*64, 12*64, 14*64,  1024}:
        print('local work items=', local_workitems_per_CU)
        if (local_workitems_per_CU > global_workitems):
            continue

        #############################################
        #  Generate a temporary JSON file for the gpu-driver
        #############################################

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

        # Format as JSON and write to file
        f.write(json.dumps( outer_item, indent=4))
        f.close()

        ##################################################
        #   run gpu-driver
        ##################################################

        print('run the gpu-driver...')
        stream = os.popen('~/AMDMIGraphX/build/bin/gpu-driver ~/AMDMIGraphX/temp_file.json')
        output = stream.read()

        # parse out the time (ms)
        startp = output.find("]: ")
        endp = output.find("ms", startp)
        print(output)
        mytime = float(output[startp+3:endp])

        print(local_workitems_per_CU, mytime)
        dataset[global_iter].append((local_workitems_per_CU, mytime))
        stream.close()

    dataset[global_iter] = sorted(dataset[global_iter], key=lambda item: item[0])
    x, y = np.array(dataset[global_iter]).T

    print("dataset is ", dataset[global_iter])
    ax = fig.add_subplot(2, 2, global_iter+1)
    plt.xscale('log')
    plt.xlabel('local workitems per CU')
    plt.ylabel('time(ms)')
    plt.ylim(bottom=0, top=20)

    ax.plot(x, y, marker='d', label='global workitems ' + str(global_workitems))
    ax.legend()

    global_workitems = global_workitems * 4


plt.suptitle('time vs local, data shape= ' + json.dumps(myshape))

outputfile='./plot.png'
plt.savefig(outputfile)
print ('your results have been plotted as ', outputfile)