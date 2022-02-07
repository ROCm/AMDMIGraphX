
#                 plot_pointwise_data.py
#                 Author:  Brian Pickrell, AMD copyright 2022

#   ingest the JSON files created by  test_broadcast_perf.py
#   and print optimum local/global inputs for the test set.
#   Use this code as an example to make pyplot charts or
#   otherwise do follow-up analysis of the data. 
#
#   syntax:  python3 plot_pointwise_data.py <input JSON file>


from matplotlib import pyplot as plt 
import json
import os
import sys
import numpy as np
import pandas

if len(sys.argv) < 2:
    print('syntax: python3 ' + __file__ + " ./test/py/pointwise/pointwise_perf_results_807270.json")
    exit(-1)

with open(sys.argv[1], "r") as read_file:
    data = json.load(read_file)

# Find the directory of the input file; output will go there
output_dir = os.path.abspath(os.path.dirname(sys.argv[1])) 

# Example:  to view the keys of the top level JSON nodes,
# print('keys are ', data.keys())

# Example:  to get a JSON subnode,
# tensors = data['tensors']
# print('length of tensors=', len(tensors))


################################################################################################################
#
#         Demonstration of loading and filtering the data in tabular form.  Column headers are hardcoded here.
#
################################################################################################################


result = data["result"]
df = pandas.DataFrame(result, columns=['tensorsize', 'global_workitems', 'local_workitems_per_CU', 'time' ])

# print(df)



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
        y = subs[param_a]
        plt.plot(x,y, marker='o', label='label')
    plt.legend([param_b + ' = ' + str(sss) for sss in ss])
    plt.xlabel(param_c)
    plt.ylabel(param_a)

# Example:
# plot local vs. time, stacked lines for each value of global
# for all tensor sizes
fig = plt.figure(figsize=(10,8))
plot_vs_parameter(df, 'time', 'global_workitems', 'local_workitems_per_CU')
plt.savefig(os.path.join(output_dir, 'local_vs_time.png'))


# Example:  plot local_workitems_per_CU vs. global_workitems, stacked lines for each tensor size
# filtering out only points where global=61440 or global=36864 (spans two tensor sizes)
# This is provided only as an example of filtering and changing axes; the result is not very useful. 

fig = plt.figure(figsize=(10,8))
plot_vs_parameter(df[(df['global_workitems']==61440) | (df['global_workitems']==36864)], 'global_workitems',  'tensorsize', 'local_workitems_per_CU')
plt.savefig(os.path.join(output_dir, 'local_vs_global.png'))


# Example:  printing a table of optimum global/local values
#
#     for each unique tensor size
#         find arg min(time)
#         list local, global for best time  ==>  append to results

# tensorsizes = np.unique(df['tensorsize'])
# tensorsizes = np.sort(tensorsizes)
# results=pandas.DataFrame(columns=df.columns)
# for ts in tensorsizes:
#     # filter for that size
#     filt_tens = df[df['tensorsize']==ts]
#     amin =  np.argmin(filt_tens['time'])
#     filt_tens.iloc[[amin]]
    
#     # Append the row at the index given in amin.  Note the double [[]] and also that the "index" value is from the original df, not filt_tens
#     results = results.append(filt_tens.iloc[[amin]])
# print('Optimal global/local values are:\n',results.iloc[:, [ True, True, True, False]],'-------\n')


# same optimum table, adding a column for  ratio tensorsize/global items
df_new = df
df_new['tens/glo'] = df_new['tensorsize']/df_new['global_workitems']
tensorsizes = np.unique(df_new['tensorsize'])
tensorsizes = np.sort(tensorsizes)
results=pandas.DataFrame(columns=df_new.columns)
for ts in tensorsizes:
    # filter for that size
    filt_tens = df_new[df_new['tensorsize']==ts]
    amin =  np.argmin(filt_tens['time'])
    filt_tens.iloc[[amin]]
    
    # Append the row at the index given in amin.  Note the double [[]] and also that the "index" value is from the original df, not filt_tens
    results = results.append(filt_tens.iloc[[amin]])
print('Optimal global/local values are:\n',results.iloc[:, [ True, True, True, False, True]],'-------\n')


#      End   Demonstration of loading and filtering the data in tabular form. 
