import os, json, subprocess, tempfile, sys, argparse, contextlib

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

def benchmark_one(config, tuning):
    b = {
        'settings': {
            'iterations': 100
        },
        'compile_op': {
            'name': 'ck_gemm',
            'tuning_val': tuning,
            'inputs': config
        }
    }
    print(b)
    with tmp_file(lambda tf: json.dump(b, tf)) as tf:
        cp = subprocess.run('./bin/gpu-driver {}'.format(tf),
                            capture_output=True, shell=True)
        for line in cp.stdout.decode().split("\n"):
            s = line.strip()
            if not s:
                continue
            fields = s.split(',')
            dtime = fields[-1].strip()
            print(dtime)
            return float(dtime[:-2])
    return sys.float_info.max


def benchmark(config, size):
    times = [benchmark_one(config, i) for i in range(size)]
    return times.index(max(times))


def benchmark_log(f):
    result = []
    for line in open(f).readlines():
        line = line.strip()
        if not line.startswith('ck_gemm:'):
            continue
        line = line[len('ck_gemm:'):].strip()
        config = json.loads(line)
        tuned = benchmark(config, 13)
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
    args = parser.parse_args()
    return args


def run(args):
    tuned = benchmark_log(args.log)
    json.dump(tuned, open(args.out, 'w+'))


run(parse_args())
