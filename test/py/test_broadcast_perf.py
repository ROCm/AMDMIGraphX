from matplotlib import pyplot as plt 
import json
import os
import numpy as np
import math
from datetime import datetime

#                 test_broadcast_perf.py
#                 Author:  Brian Pickrell, AMD copyright 2022

#
# this script runs the test program gpu-driver while iterating over 
# tensor lens; global work items; and local work items.  It plots the results.
# It saves output to a JSON File as well, so that it can be re-plotted by 
# other tools.

# this script is simlar to test_pointwise_perf.py, except that its
# Json arguments include "strides".  Setting one element in a stride to 0
# cause a tensor to have one axis broadcast in gpu-driver.
#
# This script version is for comparing speeds of operators when broadcasting 
# tensor data.
#
# Engineers using this script should expect to modify it frequently in the course of testing.
# The script is backed up each time a plot is made, to allow recreating the exact parameters/conditoins
# used in that test run.

datatype='float'
iterations=10

# This remark appears in plot header.  Update it each time you make a change.
remark=' Pointwise op test, no stride with MI100 GPU'

# A dictionary which we will use to serialize results to a file
json_data={"comment": remark, "iterations": iterations, "datatype": datatype}


#iterate over tensor sizes (one plot file per size)

json_data["tensors"] = [] 

# for the default case with no broadcasting, stride would be [cols, 1] where myshape=[rows, cols]
# See src/include/migraphx/op/broadcast.hpp line 17 for description of indexing.
# For normal indexing in a tensor, strides contains a running product of the dimension sizes of 
# the tensor, which are given in the variable lens--but offset by 1 place.  Example: for a 4-dimensional tensor where
# lens is [3, 2, 240, 1021] (not a real-world case), strides would be
# [1, 3, 6, 1440]  <== 1440 = 3*2*240
#
# For broadcasting, set one value in strides to 0 and then that axis will be ignored (broadcast) by the operator.
# setting strides to empty { } causes it to be defaulted to the non-broadcast case.
# stride=[1, 0]
stride={}

# this will hold the results in tabular format
result=[]

# myshape populates the lens parameter.  Its list size must match stride (if not defaulted), 2 items for 2d tensors.

# [rows, cols]
for myshape in [  [1024, 240] ]:
    print('processing tensor ', myshape)

    if len(myshape) != len(stride):
        print('Warning!  tensor ' , myshape, ' doesn\'t have the same number of dimensions as stride ' , stride, ', no broadcast.')
        stride={}

    # multiply all the dimensions together
    tensorsize = np.prod(myshape)

    dataset=[]
    # Create a plot
    fig = plt.figure(figsize=(10,8))

    #####################################################################
    #      loop over independent variables of interest
    #####################################################################

    # loop over global_workitems.  Range is relative to tensor size

    global_output_item_list=[]
    json_data["tensors"].append({"shape": myshape, "strides": stride, "global_outputs":     global_output_item_list})

    global_workitems = max(2, tensorsize/32)

    # construct an "interesting" range for global_workitems
    gl_range=[]
    xx = tensorsize/16
    for i in range(5):
        vv = int(3*i*xx)
        ww = int(2**i*xx)
        if vv > 63:
            gl_range.append(vv)
            gl_range.append(ww)
    gl_range.sort()
    print(tensorsize, '------------------>gl_range=', gl_range)

    # I don't know the exact limitation on global_workitems, but if it's bigger than roughly 1G it
    # will cause a program crash. 
    # while (global_workitems <= tensorsize and global_workitems < 512*1024*1024):
    for global_workitems in gl_range:
        # global_workitems = int(global_workitems*2)

        # result items to save in JSON form
        dataset=[]

        # result items to plot immediately (not save)
        plot_set=[]

        this_global = []

        # json_data["tensors"][myshape].append(this_global)


        # Any value larger than 1024 is rejected by hip library
        # for local_workitems_per_CU in { 32, 48, 64, 1024}:
        for local_workitems_per_CU in range(64, 1025,64):
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

            # 4 inputs, of which 2 are broadcasted if stride is set
            item = {"type": datatype, "lens": myshape, "strides": {}}
            inputs.append(item)

            item = {"type": datatype, "lens": myshape, "strides": stride}
            inputs.append(item)

            item = {"type": datatype, "lens": myshape, "strides": stride}
            inputs.append(item)

            item = {"type": datatype, "lens": myshape, "strides": {}}
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
            # todo: change abs. paths to paths relative to AMDMIGraphX
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

            # print(local_workitems_per_CU, mytime)

            # as an unlabeled local,time pair
            plot_set.append((local_workitems_per_CU, mytime))

           # data point to be exported to JSON file
            # As a 4-tuple of values, very compact
            data_point=((tensorsize, global_workitems, local_workitems_per_CU, mytime))
            result.append(data_point)

            # As a dict including the full shape of the tensor
            data_point={'tensor': myshape, 'tensorsize': tensorsize, 'global': global_workitems, 'local': local_workitems_per_CU, 'result, ms': mytime}
            dataset.append(data_point)

            # data_point=(local_workitems_per_CU, mytime)
            stream.close()

        # check for empty data (some inputs out of range, usually).  You must update this check
        # if you change the containers holding the results!
        # if(len(plot_set) == 0 or len(dataset) == 0 or len(result) == 0):
        if(len(plot_set) == 0 ):
            print('Warning: no output for tensor ', myshape, " global items " + str(global_workitems))
            continue

        # dict contents:  sort by key
        dataset = sorted(dataset, key=lambda item: item['tensorsize'])

        global_output_item_dict={'global_workitem_count': global_workitems, 
                'comment': 'Each data item represents one test run', 
                'data': dataset}
        
        global_output_item_list.append(global_output_item_dict)
          
        x, y = np.array(plot_set).T

        # plt.xscale('log')
        plt.xlabel('local workitems per CU')
        plt.ylabel('time(ms)')
        # scale y to make the smaller numbers (the only important ones) visible
        ymax = min(y)
        ymax = ymax*240

        ymax = (round(ymax/10., 1)+1)
        ymax = ymax / 8

        plt.ylim(bottom=0, top=ymax)
        # plt.ylim(bottom=0)
        plt.plot(x, y, marker='d', label='global workitems ' + str(global_workitems) \
            + '  t/g=' + "{:.1f}".format(float(tensorsize)/global_workitems))
        #  str(float(tensorsize)/global_workitems))
        plt.legend()

    # printing to file
    #  first, prepare to save a backup copy of this script
    save_python_dir='./test/py/pointwise/rome6_nostride/'  # should end with trailing /
    # get a unique digit string that will be used in the filenames of both the plot and backup script
    unique_no = datetime.now().microsecond
    save_python_filename = save_python_dir +os.path.splitext (os.path.basename(__file__))[0] +'_t' + str(unique_no) + '.py' 


    plt.suptitle('time vs local,  ' + remark + '   data shape= ' + json.dumps(myshape) 
        + '   tensor size= '+ str(tensorsize) + '   stride= ' + json.dumps(stride)
        + '   data=' + str(datatype)
        + '\nsource: ' + save_python_filename,  x=0.5, y=.98, fontsize='small')

    output_plot_file=save_python_dir + 'plot_' + str(tensorsize) + '_' + str(unique_no) + '.png'

    # copy this Python script to a distinctively named backup.
    # This lets us review the exact Python code used for each script run.
    os.popen('mkdir ' + save_python_dir)
    os.popen('cp ' + __file__ + ' ' + save_python_filename)
    # Save the plot to a file
    plt.savefig(output_plot_file)
    print ('your results have been plotted as ', output_plot_file, '  see also ' + save_python_filename )

# add the tabular array we created
json_data["result"] = result

#   Write all the data from all the runs to a single JSON file.  Its name contains the last unique number in our set

output_data_file = save_python_dir + 'pointwise_perf_results_' + str(unique_no) + '.json'
f = open(output_data_file, 'w')
# Format as JSON and write to file
f.write(json.dumps( json_data, indent=4))
f.close()

print("\nDone.  dataset is saved as " + output_data_file)
     