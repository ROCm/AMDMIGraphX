import json
import argparse
import os
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parser for MIGraphX ROCTX Markers")
    parser.add_argument('--json-path',
                        type=str,
                        metavar='json_path',
                        help='path to json file')
    parser.add_argument('--out',
                        type=str,
                        metavar='out',
                        help='output directory for run')
    parser.add_argument(
        '--study-name',
        type=str,
        metavar='study-name',
        help='study name is used for naming the output CSV file.')
    parser.add_argument('--repeat',
                        type=int,
                        metavar='repeat',
                        help='defines number of runs',
                        default=2)
    parser.add_argument('--parse', default=False, action='store_true')
    parser.add_argument('--run',
                        type=str,
                        metavar='run',
                        help='enables run and fetches run configs.')
    parser.add_argument('--debug', default=False, action='store_true')

    args = parser.parse_args()
    return args


def parse(file):
    args = parse_args()
    with open(file, "r") as read_file:
        data = json.load(read_file)

    #Get marker names
    list_names = []
    for i in data:
        if (i):
            if ("Marker start:" in i['name']) and (
                    i['name'] not in list_names):
                list_names.append(i['name'])

    # Get timing information for each marker name
    list_times_per_names = []
    for name in list_names:
        temp_list = []
        for entry in data:
            if (entry) and (
                    name == entry['name']
            ):  # name can match on gpu or cpu side, for gpu, we need data from gpu markers.
                if (("gpu::" in name)
                        and ("UserMarker frame:" in entry['args']['desc'])
                    ):  #gpu side information
                    temp_list.append(int(entry.get('dur')))
                elif (("gpu::" not in name)
                      and ("Marker start:" in entry['args']['desc'])
                      ):  #cpu side information
                    temp_list.append(int(entry.get('dur')))
        list_times_per_names.append(temp_list)

    sum_per_name = []
    for list in list_times_per_names:
        sum_per_name.append(sum(list))

    count_per_name = []
    for list in list_times_per_names:
        try:
            count_per_name.append(len(list))
        except:
            count_per_name.append(0)

    max_per_name = []
    for list in list_times_per_names:
        try:
            max_per_name.append(max(list))
        except:
            max_per_name.append(0)

    min_per_name = []
    for list in list_times_per_names:
        try:
            min_per_name.append(min(list))
        except:
            min_per_name.append(0)

    max_index_per_name = []
    for list in list_times_per_names:
        try:
            max_index_per_name.append(list.index(max(list)))
        except:
            max_index_per_name.append(0)

    max_occur_per_name = []
    for list in list_times_per_names:
        try:
            max_occur_per_name.append(list.count(max(list)))
        except:
            max_occur_per_name.append(0)

    total_time = sum(sum_per_name)

    d = {
        'SUM': sum_per_name,
        'MIN': min_per_name,
        'MAX': max_per_name,
        'COUNT': count_per_name,
        'MAX_INDEX': max_index_per_name,
        'MAX_OCCUR': max_occur_per_name
    }
    df2 = pd.DataFrame(d)
    df2.index = list_names
    df2.sort_values(by=['SUM'], inplace=True, ascending=False)

    if (args.debug):
        print(df2)
        print("\nTOTAL TIME: %s us" % total_time)
    return df2, total_time


def run():
    args = parse_args()
    repeat_count = args.repeat
    if (repeat_count == 0 or repeat_count == float('inf') or not repeat_count):
        raise Exception(
            "Repeat count is either, 0, infinity or not defined. Quitting.")
    run_args = args.run
    #configurations
    configs = '--hip-trace --roctx-trace --flush-rate 10ms --timestamp on'
    output_dir = '-d %s' % args.out
    executable = '/opt/rocm/bin/migraphx-driver roctx %s' % run_args
    process_args = configs + ' ' + output_dir + ' ' + executable
    for i in range(repeat_count):
        os.system('rocprof ' + process_args)
    print("RUN COMPLETE.")


def main():

    args = parse_args()
    print(args)
    file = args.json_path

    if (args.study_name):
        filename = args.study_name + ".csv"
    else:
        filename = "output" + datetime.now().strftime(
            "%Y_%m_%d-%I:%M:%S_%p") + ".csv"

    with open(filename, 'a') as f:
        f.write(args.run)
        f.write('\n')

    if (args.run):
        curr = os.path.abspath(os.getcwd())
        if not os.path.exists('/tmp/rocmProfileData'):
            print("rocmProfileData does not exist. Cloning.")
            os.system(
                'git clone https://github.com/ROCmSoftwarePlatform/rocmProfileData.git /tmp/rocmProfileData'
            )

        os.chdir("/tmp/rocmProfileData/rocpd_python/")
        os.system('python setup.py install')
        os.chdir("/tmp/rocmProfileData/")
        os.chdir(curr)
        run()
        os.chdir(curr + "/%s/" % args.out)
        out_path = os.popen("ls -td $PWD/*/*/ | head -%s" % args.repeat).read()
        print("\nFollowing paths will be parsed:\n%s" % out_path)
        out_path = out_path.splitlines()
        df_tot = pd.DataFrame()
        tot_time = []
        for path in out_path:
            path = path.strip('\n')
            print("\nPARSING OUTPUT PATH: " + path)
            os.chdir(path)
            os.system(
                "python -m rocpd.rocprofiler_import --ops_input_file hcc_ops_trace.txt --api_input_file hip_api_trace.txt --roctx_input_file roctx_trace.txt trace.rpd"
            )
            os.system(
                "python /tmp/rocmProfileData/rpd2tracing.py trace.rpd trace.json"
            )
            os.chdir(curr)
            df, total_time = parse(path + "trace.json")
            tot_time.append(total_time)
            df_tot = pd.merge(df_tot,
                              df,
                              how='outer',
                              left_index=True,
                              right_index=True)
            if (args.debug):
                print("JSON FILE PATH: " + path + "trace.json")

        if (args.debug):
            print(df_tot)

        tmp_sum = df_tot.loc[:, df_tot.columns.str.contains('SUM')].astype(int)
        tmp_min = df_tot.loc[:, df_tot.columns.str.contains('MIN')].astype(int)
        tmp_max = df_tot.loc[:, df_tot.columns.str.match("^MAX_.$")].astype(
            int)
        tmp_count = df_tot.loc[:, df_tot.columns.str.match("COUNT")].astype(
            int)

        tmp_sum['SUM_avg'] = tmp_sum.mean(axis=1).astype(int)
        tmp_min['MIN_avg'] = tmp_min.mean(axis=1).astype(int)
        tmp_max['MAX_avg'] = tmp_max.mean(axis=1).astype(int)

        df2 = tmp_sum['SUM_avg'].copy()
        df2 = pd.merge(df2,
                       tmp_min['MIN_avg'],
                       how='outer',
                       left_index=True,
                       right_index=True)
        df2 = pd.merge(df2,
                       tmp_max['MAX_avg'],
                       how='outer',
                       left_index=True,
                       right_index=True)
        df2 = pd.merge(df2,
                       tmp_count['COUNT_x'],
                       how='outer',
                       left_index=True,
                       right_index=True)
        df2.rename(columns={'COUNT_x': 'COUNT'}, inplace=True)
        df2.sort_values(by=['SUM_avg'], inplace=True, ascending=False)

        if (args.debug):
            pd.set_option('display.max_columns', None)
            print(df_tot)
        print("\n*** RESULTS ***")
        print(df2)
        out_time = sum(tot_time) / len(tot_time)
        print("\nAVG TOTAL TIME: %s us\n" % int(out_time))
        df2.to_csv(filename, mode='a')
        with open(filename, 'a') as f:
            f.write("AVG TOTAL TIME: %s us\n" % int(out_time))
        print("OUTPUT CSV FILE: %s" % filename)

    if (args.parse):
        if not (file):
            raise Exception("JSON path is not provided for parsing.")
        parse(file)


if __name__ == "__main__":
    main()
