from matplotlib import pyplot as plt 
from mpl_toolkits.axes_grid1 import Divider, Size
import json
import os
import numpy as np
import math
from datetime import datetime

# with open("compile-pointwise-args.json", "r") as read_file:
#     data = json.load(read_file)

datatype='float'
iterations=100

remark=' local as multiples of 64, iter=100'

#iterate over tensor sizes (one plot file per size)

# for myshape in [[1024]]:
for myshape in [ [512, 16*16*16],  [512, 16*16*16*16], [512,16*16*16*16, 15] ]:

    tensorsize = 1
    for x in myshape:
        tensorsize = tensorsize * x

    dataset=[]
    # Create a plot

    fig = plt.figure(figsize=(10,5))

    # loop over independent variables of interest

    # loop over global_workitems.  Range is relative to tensor size

    global_workitems = max(2,tensorsize/32)
            # I added max check on  because migraphx throws exceptions if global gets out of range, e.g. 
            #     In file included from main.cpp:2:
            # ./migraphx/kernels/index.hpp:18:16: error: integer literal is too large to be represented in a signed integer type, interpreting as unsigned [-Werror,-Wimplicitly-unsigned-literal]
            #         return MIGRAPHX_NGLOBAL;
            #             ^
            # <command line>:1:26: note: expanded from here
            # #define MIGRAPHX_NGLOBAL 18446744071562067968
            #                         ^
            # 1 error generated when compiling for gfx908.
            # terminate called after throwing an instance of 'migraphx::version_1::exception'
            # what():  /home/bpickrel/AMDMIGraphX/src/compile_src.cpp:41: compile: Output file missing: main.o
            # Aborted (core dumped)
            # local work items= 32
            # In file included from main.cpp:2:
            # ./migraphx/kernels/index.hpp:18:16: error: integer literal is too large to be represented in a signed integer type, interpreting as unsigned [-Werror,-Wimplicitly-unsigned-literal]
            #         return MIGRAPHX_NGLOBAL;
            #             ^
            # <command line>:1:26: note: expanded from here
            # #define MIGRAPHX_NGLOBAL 18446744071562067968
    while (global_workitems <= 2*tensorsize and global_workitems < 512*1024*1024):
    # for global_workitems in {32, 64, 128, 512, 1024, 4*1024, }:
        global_workitems = int(global_workitems*2)
        dataset=[]


        # Apparently any value larger than 1024 is illegal
        # for local_workitems_per_CU in { 32, 48, 64, 1024}:
        for local_workitems_per_CU in {8, 11, 32, 64, 128, 192, 4*64, 5*64, 6*64, 8*64, 10*64, 12*64, 14*64,  1024}:
        # for local_workitems_per_CU in [64, 128]:
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
            stream.close()

        dataset = sorted(dataset, key=lambda item: item[0])
        x, y = np.array(dataset).T

        # print("dataset is ", dataset, "\nbest y is  ***** ", min(y))
        plt.xscale('log')
        plt.xlabel('local workitems per CU')
        plt.ylabel('time(ms)')
        # scale y to make the smaller numbers (the only important ones) visible
        ymax = min(y)
        ymax = ymax*200

        ymax = (round(ymax/10., 1)+1)
        ymax = ymax / 20

        plt.ylim(bottom=0, top=ymax)

        plt.plot(x, y, marker='d', label='global workitems ' + str(global_workitems))
        plt.legend()

    # printing to file
    #  first, prepare to save a backup copy of this script
    save_python_dir='./test/py/pointwise/'
    # get a unique digit string that will be used in the filenames of both the plot and backup script
    unique_no = datetime.now().microsecond
    save_the_python_file = save_python_dir +os.path.splitext (os.path.basename(__file__))[0] +'_t' + str(unique_no) + '.py' 


    plt.suptitle('time vs local,  ' + remark + '   data shape= ' + json.dumps(myshape) 
        + '   tensor size= '+ str(tensorsize)
        + '   data=' + str(datatype)
        + '\nsource: ' + save_the_python_file,  x=0.4, y=.98, fontsize='small')

    plt.text(0.5, 0.7, 'hello',
         horizontalalignment='center',
         fontsize='small')

    outputfile=save_python_dir + 'plot_' + str(tensorsize) + '_' + str(unique_no) + '.png'

    # copy this Python script to a distinctively named backup.
    # This lets us review the exact Python code used for each script run.
    os.popen('mkdir ' + save_python_dir)
    os.popen('cp ' + __file__ + ' ' + save_the_python_file)
    # Save the plot to a file
    plt.savefig(outputfile)
    print ('your results have been plotted as ', outputfile, '  see also ' + save_the_python_file )

    