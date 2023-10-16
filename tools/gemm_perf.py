import subprocess, csv, re, datetime, argparse, os, enum


def parse_args():
    parser = argparse.ArgumentParser(description="GEMM performance tools")
    parser.add_argument('--gemm',
                        action='store_true',
                        help='Run performance comparison on a range of GEMM problem sizes (fp16/int8)')
    parser.add_argument('--int8',
                        action='store_true',
                        help='Quantize GEMMs to int8 precision (not available for GEMM-Softmax-GEMM)')
    parser.add_argument('--gemm-softmax-gemm',
                        action='store_true',
                        help='Run performance comparison on a range of GEMM-Softmax-GEMM problem sizes (fp16)')
    parser.add_argument('--batch_sizes',
                        '-b',
                        nargs='+',
                        help='Batch sizes to run',
                        required=True)
    parser.add_argument('--exclude-lens',
                        '-e',
                        nargs='+',
                        help='Exclude lengths from [64, 256, 384, 512, 768, 1024, 1920, 2048, 2304, 3072, 4096] \
                              Lengths not excluded will be permuted as m, n, k, (o) inputs')
    parser.add_argument('--lens-from-file',
                        '-l',
                        help='Run a list of problem lens from file containing rows of \
                              m0 n0 k0 (o0) \
                              m1 n1 k1 (o1) \
                              ... \
                              (oi) only used for gemm-softmax-gemm')
    parser.add_argument('--timeout',
                        '-t',
                        type=int,
                        default=600,
                        help='Time in seconds before compilation timeout')
    args = parser.parse_args()

    return args


class CSVFile:
    def __init__(self, path="output.csv"):
        self.path = path

    def write_row(self, row=[]):
        row = [str(r) for r in row]
        with open(self.path, "a+") as f:
            cw = csv.writer(f)
            cw.writerow(row)


class GEMM_Provider(enum.Enum):
    CK = 1
    ROCBLAS = 2 
    MLIR = 3


def get_migraphx_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_device_name():
    out = subprocess.run("rocminfo",
                         capture_output=True,
                         check=True,
                         shell=True)
    matches = re.findall("gfx\d*[a-z]*", str(out.stdout))
    return matches[0]


def verify_format(file, single):
    if not file:
        return False
    format = r"^\d\s\d\s\d\s*$"  if single else r"^\d\s\d\s\d\s\d\s*$"
    with open(file, 'r') as f:
        return all([bool(re.match(format, line)) for line in f.readlines()])


def parse_lens(file):
    with open(file, 'r') as f:
        return [tuple(map(int, line.split())) for line in f.readlines()]


def get_total_time(output):
    summary = re.findall("Summary.*", output)[0].replace("\\n", "\n")
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    return float(total_time.replace("Total time: ", ""))


def verify_output(output, provider):
    summary = re.findall("Summary.*", output)[0].replace("\\n", "\n")
    if provider == GEMM_Provider.CK:
        return bool(re.search(".*ck_gemm.*", summary))
    if provider == GEMM_Provider.ROCBLAS:
        return bool(re.search(".*gpu::gemm.*", summary)) or bool(re.search(".*gpu::quant_gemm.*", summary))
    if provider == GEMM_Provider.MLIR:
        return bool(re.search(".*mlir_dot.*", summary)) or bool(re.search(".*mlir_quant_dot.*", summary))


def get_gemm_time(config, fp16, provider, timeout):
    root = get_migraphx_root()
    model = f"{root}/test/onnx/matmul_half.onnx" 
    b, m, n, k = config
    prec_str = "--fp16" if fp16 else "--int8"
    cmd = f"{root}/build/bin/driver perf {model} --input-dim @1 {b} {m} {k} @2 {b} {k} {n} {prec_str} --exhaustive-tune"
    use_CK = "1" if provider == GEMM_Provider.CK else "0"
    use_MLIR = "1" if provider == GEMM_Provider.MLIR else "0"

    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=timeout,
                             env=dict(os.environ, 
                                      MIGRAPHX_ENABLE_CK=use_CK,
                                      MIGRAPHX_ENABLE_MLIR=use_MLIR,
                                      MIGRAPHX_MLIR_USE_SPECIFIC_OPS="dot"))
    except Exception as e:
        print(f"{provider.name} encountered and exception {e}")
        return -100.0
    
    if verify_output(str(out.stdout), provider):
        total_time = get_total_time(str(out.stdout))
        print(f"{provider.name} total time: {total_time} ms")
        return total_time
    else:
        print(f"{provider.name} was not found in performance summary")
        return -100.0


def get_gemm_softmax_gemm_time(config, provider, timeout):
    root = get_migraphx_root()
    model = f"{root}/test/onnx/gemm_softmax_gemm_half.onnx" 
    b, m, n, k, o = config
    cmd = f"{root}/build/bin/driver perf {model} --input-dim @a {b} {m} {k} @b {b} {k} {n} @b1 {b} {n} {o} --fp16 --exhaustive-tune"
    use_CK = "1" if provider == GEMM_Provider.CK else "0"
    use_MLIR = "1" if provider == GEMM_Provider.MLIR else "0"

    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=timeout,
                             env=dict(os.environ, 
                                      MIGRAPHX_ENABLE_CK=use_CK,
                                      MIGRAPHX_ENABLE_MLIR=use_MLIR,
                                      MIGRAPHX_MLIR_USE_SPECIFIC_OPS="dot"))
    except Exception as e:
        print(f"{provider.name} encountered and exception {e}")
        return -100.0
    
    if verify_output(str(out.stdout), provider):
        total_time = get_total_time(str(out.stdout))
        print(f"{provider.name} total time: {total_time} ms")
        return total_time
    else:
        print(f"{provider.name} was not found in performance summary")
        return -100.0


def run_gemm_perf(batches, sizes, fp16, timeout):
    prec_str = "fp16" if fp16 else "int8"
    for b in batches:
        out = CSVFile(f"gemm_perf_{prec_str}_{b}.csv")
        out.write_row([get_device_name(), datetime.datetime.now()])
        out.write_row(["batch_size", "m", "n", "k", "CK Total Time (ms)", "rocBLAS Total Time (ms)", "MLIR Total Time (ms)"])
        for shape in sizes:
            config = (b,) + shape
            print("Running {prec} gemm with config: {0}, {1}, {2}, {3}".format(prec=prec_str, *config))
            ck_time = get_gemm_time(config, fp16, GEMM_Provider.CK, timeout)
            rb_time = get_gemm_time(config, fp16, GEMM_Provider.ROCBLAS, timeout)
            mlir_time = get_gemm_time(config, fp16, GEMM_Provider.MLIR, timeout)
            out.write_row(list(config) + [ck_time, rb_time, mlir_time])


def run_gemm_softmax_gemm_perf(batches, sizes, timeout):
    for b in batches:
        out = CSVFile(f"gemm_softmax_gemm_perf_fp16_{b}.csv")
        out.write_row([get_device_name(), datetime.datetime.now()])
        out.write_row(["batch_size", "m", "n", "k", "o", "CK Total Time (ms)", "rocBLAS Total Time (ms)", "MLIR Total Time (ms)"])
        for shape in sizes:
            config = (b,) + shape
            print("Running fp16 gemm-softmax-gemm with config: {0}, {1}, {2}, {3}, {4}".format(*config))
            ck_time = get_gemm_softmax_gemm_time(config, GEMM_Provider.CK, timeout)
            rb_time = get_gemm_softmax_gemm_time(config, GEMM_Provider.ROCBLAS, timeout)
            mlir_time = get_gemm_softmax_gemm_time(config, GEMM_Provider.MLIR, timeout)
            out.write_row(list(config) + [ck_time, rb_time, mlir_time])


if __name__ == "__main__":
    args = parse_args()
    exclude = args.exclude_lens
    exclude = [int(x) for x in exclude]
    sizes = [64, 256, 384, 512, 768, 1024, 1920, 2048, 2304, 3072, 4096]
    sizes = [x for x in sizes if x not in exclude]
    fp16 = not args.int8
    timeout = int(args.timeout)
    if args.gemm:
        gemm_sizes = [(m, n, k) for m in sizes for n in sizes for k in sizes]
        if verify_format(args.lens_from_file, True):
            gemm_sizes = parse_lens(args.lens_from_file)
        run_gemm_perf(args.batch_sizes, gemm_sizes, fp16, timeout)
    if args.gemm_softmax_gemm:
        gemm_softmax_gemm_sizes = [(m, n, k, o) for m in sizes for n in sizes for k in sizes for o in sizes]
        if verify_format(args.lens_from_file, False):
            gemm_softmax_gemm_sizes = parse_lens(args.lens_from_file)
        run_gemm_softmax_gemm_perf(args.batch_sizes, gemm_softmax_gemm_sizes, timeout)
