from matplotlib import pyplot as plt 
import json
import os
import numpy as np
import math
from datetime import datetime

#                 test_pointwise_perf.py

# this script runs the gpu-driver while iterating over various combinations of 
# tensor lens; global work items; and local work items.  It plots the results.
# It saves output to a JSON File as well, so that it can be re-plotted by 
# other tools.

# This script version is for comparing speeds of simple pointwise operators,
# our simplest use case.




# with open("compile-pointwise-args.json", "r") as read_file:
#     data = json.load(read_file)

datatype='float'
iterations=10

remark=' Searching for optimal global, local number of work items '

# A dictionary which we will use to serialize results to a file
json_data={"comment": remark, "iterations": iterations, "datatype": datatype}


#iterate over tensor sizes (one plot file per size)

json_data["shapes"] = [] # this should be a list of dict containing "shape": xxx, "global_list":
# for myshape in [[32,32]]:
# for myshape in [ [512, 16*16*4], [512, 16*16*16],  [512, 16*16*16*4],  [512, 16*16*16*16] ]:
for myshape in [  [7680], [7681],  [7680, 30], [200000], [2*1024, 1024], [1024*1024, 30]]:

    tensorsize = 1
    for x in myshape:
        tensorsize = tensorsize * x

    dataset=[]
    # Create a plot
    fig = plt.figure(figsize=(10,8))

    #####################################################################
    #      loop over independent variables of interest
    #####################################################################

    # loop over global_workitems.  Range is relative to tensor size

    global_output_item_list=[]
    json_data["shapes"].append({"shape": myshape, "global_outputs":     global_output_item_list})

    global_workitems = max(2,tensorsize/32)

    # I don't know the exact limitation on global_workitems, but if it's bigger than roughly 1G it
    # will cause a program crash. 
    while (global_workitems <= tensorsize and global_workitems < 512*1024*1024):
    # for global_workitems in {32, 64, 128, 512, 1024, 4*1024, }:
        global_workitems = int(global_workitems*2)
        dataset=[]

        this_global = []

        # json_data["shapes"][myshape].append(this_global)


        # Any value larger than 1024 is rejected by hip library
        # for local_workitems_per_CU in { 32, 48, 64, 1024}:
        for local_workitems_per_CU in { 64, 66, 128,130, 192, 4*64,  6*64, 6*64+5, 8*64, 10*64, 12*64, 14*64,  1024}:
        # for local_workitems_per_CU in [64, 128]:
            print('local work items=', local_workitems_per_CU)
            if (local_workitems_per_CU > global_workitems):
                continue

            #############################################
            #  Generate a temporary JSON file for the gpu-driver
            #############################################

            f = open('./temp_file.json', 'w')

            cp  = {  }  # compile_pointwise

            # this lambda with more looping makes it easier to distinguish differences
            cp["lambda"] = "[](auto x, auto y, auto z) {for (int i = 1; i < 200; i++){ z = sqrt(abs(z+y));} return x+y+z; }"

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

            # print('run the gpu-driver...')
            stream = os.popen('~/AMDMIGraphX/build/bin/gpu-driver ~/AMDMIGraphX/temp_file.json')
            output = stream.read()

            # parse out the time (ms)
            startp = output.find("]: ")
            endp = output.find("ms", startp)

            # an error here probably means we gave the gpu-driver an arg or args out of range.  Skip the bad data point
            if(startp == -1 or endp == -1):
                continue
            # print(output)
            mytime = float(output[startp+3:endp])

            print(local_workitems_per_CU, mytime)
            dataset.append((local_workitems_per_CU, mytime))
            data_point=(local_workitems_per_CU, mytime)
            stream.close()

        dataset = sorted(dataset, key=lambda item: item[0])
        global_output_item_dict={'global_workitem_count': global_workitems, 
                'comment': 'Each data item represents one test run as [local_workitems, time]', 
                'data': dataset}
        
        global_output_item_list.append(global_output_item_dict)
          
        x, y = np.array(dataset).T

        plt.xscale('log')
        plt.xlabel('local workitems per CU')
        plt.ylabel('time(ms)')
        # scale y to make the smaller numbers (the only important ones) visible
        ymax = min(y)
        ymax = ymax*200

        ymax = (round(ymax/10., 1)+1)
        ymax = ymax / 4

        plt.ylim(bottom=0, top=ymax)
        # plt.ylim(bottom=0)
        plt.plot(x, y, marker='d', label='global workitems ' + str(global_workitems) \
            + '  t/g=' + "{:.1f}".format(float(tensorsize)/global_workitems))
        #  str(float(tensorsize)/global_workitems))
        plt.legend()

    # printing to file
    #  first, prepare to save a backup copy of this script
    save_python_dir='./test/py/pointwise/'
    # get a unique digit string that will be used in the filenames of both the plot and backup script
    unique_no = datetime.now().microsecond
    save_python_filename = save_python_dir +os.path.splitext (os.path.basename(__file__))[0] +'_t' + str(unique_no) + '.py' 


    plt.suptitle('time vs local,  ' + remark + '   data shape= ' + json.dumps(myshape) 
        + '   tensor size= '+ str(tensorsize)
        + '   data=' + str(datatype)
        + '\nsource: ' + save_python_filename,  x=0.4, y=.98, fontsize='small')

    output_plot_file=save_python_dir + 'plot_' + str(tensorsize) + '_' + str(unique_no) + '.png'

    # copy this Python script to a distinctively named backup.
    # This lets us review the exact Python code used for each script run.
    os.popen('mkdir ' + save_python_dir)
    os.popen('cp ' + __file__ + ' ' + save_python_filename)
    # Save the plot to a file
    plt.savefig(output_plot_file)
    print ('your results have been plotted as ', output_plot_file, '  see also ' + save_python_filename )

#   Write all the data from all the runs to a single JSON file.  Its name contains the last unique number in our set
output_data_file = save_python_dir + 'pointwise_perf_results_' + str(unique_no) + '.json'
f = open(output_data_file, 'w')
# Format as JSON and write to file
f.write(json.dumps( json_data, indent=4))
f.close()

print("\nDone.  dataset is saved as " + output_data_file)
     