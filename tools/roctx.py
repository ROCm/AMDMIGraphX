import json
import argparse
import os
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parser for MIGraphX ROCTX Markers")
    parser.add_argument('--json_path',
                        type=str,
                        metavar='json_path',
                        help='path to json file')
    parser.add_argument('--migraphx_args',
                        type=str,
                        metavar='migraphx_args',
                        help='args to pass to migraphx-driver')
    parser.add_argument('--out',
                        type=str,
                        metavar='out',
                        help='output directory for run')
    parser.add_argument('--parse', default=False, action='store_true')
    parser.add_argument('--run', default=False, action='store_true')
    parser.add_argument('--onnx_file', type=str)

    args = parser.parse_args()
    return args


def parse(file):
    with open(file, "r") as read_file:
        data = json.load(read_file)

    #Get marker names
    list_names = []
    for i in data:
        if (i):
            if ("Marker start:" in i['name']) and (i['name']
                                                   not in list_names):
                list_names.append(i['name'])

    # Get timing information for each marker name
    # print(list_names)
    list_times_per_names = []
    for name in list_names:
        print(name)
        temp_list = []
        for entry in data:
            if (entry) and (
                    name == entry['name']
            ):  # name can match on gpu or cpu side, for gpu, we need data from gpu markers.
                if (("gpu::" in name)
                        and ("UserMarker frame:"
                             in entry['args']['desc'])):  #gpu side information
                    print(entry)
                    temp_list.append(int(entry.get('dur')))
                elif (("gpu::" not in name)
                      and ("Marker start:"
                           in entry['args']['desc'])):  #cpu side information
                    print(entry)
                    temp_list.append(int(entry.get('dur')))
        list_times_per_names.append(temp_list)

    print(list_names)
    print(list_times_per_names)

    # Sum duration for each entry for a given name
    sum_per_name = []
    for list in list_times_per_names:
        sum_per_name.append(sum(list))

    max_per_name = []
    for list in list_times_per_names:
        try:
            max_per_name.append(max(list))
        except:
            max_per_name.append("ERR")

    min_per_name = []
    for list in list_times_per_names:
        try:
            min_per_name.append(min(list))
        except:
            min_per_name.append("ERR")
    
    max_index_per_name = []
    for list in list_times_per_names:
        try:
            max_index_per_name.append(list.index(max(list)))
        except:
            max_index_per_name.append("ERR")
    
    print("SUM: %s" % sum_per_name)
    print("MAX: %s" % max_per_name)
    print("MIN: %s" % min_per_name)
    print("MAX_INDEX: %s" % max_index_per_name)

    total_time = sum(sum_per_name)

    d = {'SUM': sum_per_name, 'MIN': min_per_name, 'MAX': max_per_name, 'MAX_INDEX': max_index_per_name}
    df2 = pd.DataFrame(d)
    df2.index = list_names
    df2.sort_values(by=['SUM'], inplace=True, ascending=False)

    print(df2)
    print("\nTOTAL TIME: %s us\n" % total_time)


def run():
    args = parse_args()
    onnx_path = args.onnx_file
    migraphx_args = args.migraphx_args
    if not (onnx_path):
        raise Exception("No ONNX file is provided to run.")
    onnx_rpath = os.path.realpath(onnx_path)
    print(onnx_rpath)
    #configurations
    configs = '--hip-trace --roctx-trace --flush-rate 10ms --timestamp on'
    output_dir = '-d %s' % args.out
    executable = '/opt/rocm/bin/migraphx-driver roctx %s %s' % (onnx_rpath,
                                                                migraphx_args)
    process_args = configs + ' ' + output_dir + ' ' + executable
    os.system('rocprof ' + process_args)
    print("RUN COMPLETE.")


def main():

    args = parse_args()
    print(args)
    file = args.json_path

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
        out_path = os.popen("ls -td $PWD/*/*/ | head -1").read()
        out_path = out_path.strip('\n')
        print("OUTPUT PATH: " + out_path)
        os.chdir(out_path)
        os.system(
            "python -m rocpd.rocprofiler_import --ops_input_file hcc_ops_trace.txt --api_input_file hip_api_trace.txt --roctx_input_file roctx_trace.txt trace.rpd"
        )
        os.system(
            "python /tmp/rocmProfileData/rpd2tracing.py trace.rpd trace.json")
        os.chdir(curr)
        parse(out_path + "trace.json")
        print("JSON FILE PATH: " + out_path + "trace.json")

    if (args.parse):
        if not (file):
            raise Exception("JSON path is not provided for parsing.")
        parse(file)


if __name__ == "__main__":
    main()
