import os, json, subprocess, tempfile, sys, argparse, contextlib

ck_function = -1


@contextlib.contextmanager
def tmp_file(dump=None):
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            tmp_name = f.name
            if dump:
                dump(f)
        yield tmp_name
    finally:
        os.unlink(tmp_name)


def pretty_print(obj):
    print(json.dumps(obj, indent=2))


def run_driver(b):
    print(b)
    with tmp_file(lambda tf: json.dump(b, tf)) as tf:
        cp = subprocess.run('./bin/gpu-driver {}'.format(tf),
                            capture_output=True,
                            check=True,
                            shell=True)
        for line in cp.stdout.decode().split("\n"):
            s = line.strip()
            if not s:
                continue
            if not ']: ' in s:
                continue
            yield s.split(']: ')[1].strip()


def convert_to_float(s):
    return s[:-2]


def get_device_time(s):
    fields = s.split(',')
    return convert_to_float(fields[-1].strip())


def benchmark_ck(config, name, tuning):
    try:
        b = {
            'settings': {
                'iterations': 100
            },
            'compile_op': {
                'name': name,
                'check': True,
                'tuning_val': tuning,
                'inputs': config
            }
        }
        for line in run_driver(b):
            dtime = get_device_time(line)
            print(dtime)
            return float(dtime)
    except:
        return sys.float_info.max


def benchmark(config, name, size):
    times = [benchmark_ck(config, name, i) for i in range(size)]
    return times.index(min(times))


def parse_log(f):
    for line in open(f).readlines():
        line = line.strip()
        global ck_function
        if line.startswith('ck_gemm:'):
            line = line[len('ck_gemm:'):].strip()
            config = json.loads(line)
            yield (config, 'ck_gemm')
        if line.startswith('ck_gemm_softmax_gemm:'):
            line = line[len('ck_gemm_softmax_gemm:'):].strip()
            config = json.loads(line)
            ck_function = 1
            yield (config, 'ck_gemm_softmax_gemm')


def benchmark_log(f, n):
    result = []
    for config, name in parse_log(f):
        tuned = benchmark(config, name, n)
        print("Tuned:", tuned)
        result.append([config, tuned])
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Simple tuner for CK gemms")
    parser.add_argument('--log',
                        '-l',
                        type=str,
                        metavar='file',
                        help='Path to logfile')
    parser.add_argument('--out',
                        '-o',
                        type=str,
                        metavar='file',
                        help='Output json file to save tunings')
    parser.add_argument('-n', type=int, help='Number of instances to tune')
    args = parser.parse_args()
    return args


def run(args):
    tuned = benchmark_log(args.log, args.n)
    json.dump(tuned, open(args.out, 'w+'))


def tune(log, n, out):
    tuned = benchmark_log(log, n)
    json.dump(tuned, open(out, 'w+'))


if __name__ == '__main__':
    run(parse_args())
