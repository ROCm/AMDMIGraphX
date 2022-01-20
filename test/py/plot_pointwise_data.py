
# plot_pointwise_data.py :  ingest the JSON files created by test runs 
# from either test_pointwise_perf or test_broadcast_perf.py.
# Add your own code to make pyplot charts. 


from matplotlib import pyplot as plt 
import json
import os
import sys
import numpy as np
import pandas

if len(sys.argv) < 2:
    print('syntax: ' + __file__ + " ./test/py/pointwise/pointwise_perf_results_807270.json")
    exit(-1)

with open(sys.argv[1], "r") as read_file:
    data = json.load(read_file)

print('keys are^^^^^ ',data.keys())
# shapes = data['shapes']
# print('length of shapes=', len(shapes))

# myshape = shapes[0]

# glo = myshape['global_outputs']
# print('num of globals=', len(glo))

# my_local = glo[0]['data'][0]  # a local/time pair
# print('local val=', my_local)

################################################################################################################
#
#         Demonstration of loading and filtering the data in tabular form.  Column headers are hardcoded here.
#
################################################################################################################


result = data["result"]
df = pandas.DataFrame(result, columns=['tensorsize', 'global_workitems', 'local_workitems_per_CU', 'time' ])

print(df)



# Given three dataframe columns (column labels), plot one as a function of the other two.
#
# the_df  pandas.DataFrame 
# param_a string is the dependent variable.  (column label) 
# param_b string is the categorical variable, i.e. there is one stacked line for each unique value of b (column label) 
# param_c string  is the independent variable (the x axis) (column label)
#
def plot_vs_parameter(the_df, param_a, param_b, param_c):
    the_df = the_df.sort_values(by=[param_b, param_c])
    # make a list of unique global values

    # issue: we get a different range of param_b (global) for different param_a (tensor size)
    # each tensorsize should get its own legend??
    ss = np.unique(the_df[param_b])
    # print('ss=',ss)

    # for each unique value of param_b, plot a separate line
    for uniq in ss:
        subs = the_df[the_df[param_b] == uniq]
        x = subs[param_c]
        y = subs['time']
        plt.plot(x,y, marker='o', label='label')
    plt.legend([param_b + ' = ' + str(sss) for sss in ss])
    plt.xlabel(param_c)
    plt.ylabel('time, ms')

# plot local vs. time, stacked lines for each value of global
# for all tensor sizes
fig = plt.figure(figsize=(10,8))
plot_vs_parameter(df, 'tensorsize', 'global_workitems', 'local_workitems_per_CU')
plt.savefig('./test/py/pointwise/a447.png')


# plot local vs. time, stacked lines for each tensor size
# filtering out only points where global=7500 (there's only one tensor size for this)
fig = plt.figure(figsize=(10,8))
plot_vs_parameter(df[df['global_workitems']==7500], 'global_workitems',  'tensorsize', 'local_workitems_per_CU')

plt.savefig('./test/py/pointwise/a446.png')


# new test:  
#
#     for each unique tensor size
#         find arg min(time)
#         list local, global for best time  ==>  append to results
tensorsizes = np.unique(df['tensorsize'])
tensorsizes = np.sort(tensorsizes)
results=pandas.DataFrame(columns=df.columns)
for ts in tensorsizes:
    # filter for that size
    filt_tens = df[df['tensorsize']==ts]
    amin =  np.argmin(filt_tens['time'])
    filt_tens.iloc[[amin]]
    
    # Append the row at the index given in amin.  Note the double [[]] and also that the "index" value is from the original df, not filt_tens
    results = results.append(filt_tens.iloc[[amin]])
print('Optimal global/local values are:\n',results.iloc[:, [ True, True, True, False]],'-------\n')

#      End   Demonstration of loading and filtering the data in tabular form.  Column headers are hardcoded here.