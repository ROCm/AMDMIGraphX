#%matplotlib
import subprocess, csv, re, datetime, argparse, os, enum
from subprocess import STDOUT


class GEMM_Provider(enum.Enum):
    CK = 1
    ROCBLAS = 2 
    MLIR = 3


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
                        help='Exclude lengths from [64, 256, 384, 512, 768, 1024, 1920, 2048, 2304, 3072, 4096]')
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


def get_device_name():
    out = subprocess.run("rocminfo",
                         capture_output=True,
                         check=True,
                         shell=True)
    matches = re.findall("gfx\d*[a-z]*", str(out.stdout))
    return matches[0]


def get_gemm_time(config, fp16, provider, timeout):
    model = "../test/onnx/matmul_half.onnx" 
    b, m, n, k = config
    prec_str = "--fp16" if fp16 else "--int8"
    cmd = f"../build/bin/driver perf {model} --input-dim @1 {b} {m} {k} @2 {b} {k} {n} {prec_str}"
    use_CK = "1" if provider == GEMM_Provider.CK else "0"
    use_MLIR = "1" if provider == GEMM_Provider.MLIR else "0"

    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=timeout,
                             env=dict(os.environ, 
                                      MIGRAPHX_ENABLE_CK=use_CK,
                                      MIGRAPHX_ENABLE_MLIR=use_MLIR))
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        print(f"{provider.name} GEMM {b}, {m}, {n}, {k}:", end=" ")
        print("-100.0")
        return -100.0
    
    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")

    return float(total_time)


def get_gemm_softmax_gemm_time(config, provider, timeout):
    model = "../test/onnx/gemm_softmax_gemm_half.onnx" 
    b, m, n, k, o = config
    cmd = f"../build/bin/driver perf {model} --input-dim @a {b} {m} {k} @b {b} {k} {n} @b1 {b} {n} {o} --fp16"
    use_CK = "1" if provider == GEMM_Provider.CK else "0"
    use_MLIR = "1" if provider == GEMM_Provider.MLIR else "0"

    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=timeout,
                             env=dict(os.environ, 
                                      MIGRAPHX_ENABLE_CK=use_CK,
                                      MIGRAPHX_ENABLE_MLIR=use_MLIR))
    except Exception as e:
        return -100.0
    
    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")

    return float(total_time)


def run_gemm_perf(batches, sizes, fp16):
    prec_str = "half" if fp16 else "int"
    for b in batches:
        out = CSVFile(f"gemm_perf_{prec_str}_{b}.csv")
        out.write_row([get_device_name(), datetime.datetime.now()])
        out.write_row(["batch_size", "m", "n", "k", "CK Total Time (ms)", "rocBLAS Total Time (ms)", "MLIR Total Time (ms)"])
        for shape in [(m, n, k) for m in sizes for n in sizes for k in sizes]:
            config = (b,) + shape
            ck_time = get_gemm_time(config, fp16, GEMM_Provider.CK)
            rb_time = get_gemm_time(config, fp16, GEMM_Provider.ROCBLAS)
            mlir_time = get_gemm_time(config, fp16, GEMM_Provider.MLIR)
            out.write_row(list(config) + [ck_time, rb_time, mlir_time])


def run_gemm_softmax_gemm_perf(batches, sizes):
    for b in batches:
        out = CSVFile(f"gemm_softmax_gemm_perf_fp16_{b}.csv")
        out.write_row([get_device_name(), datetime.datetime.now()])
        out.write_row(["batch_size", "m", "n", "k", "o", "CK Total Time (ms)", "rocBLAS Total Time (ms)", "MLIR Total Time (ms)"])
        for shape in [(m, n, k, o) for m in sizes for n in sizes for k in sizes for o in sizes]:
            config = (b,) + shape
            ck_time = get_gemm_time(config, GEMM_Provider.CK)
            rb_time = get_gemm_time(config, GEMM_Provider.ROCBLAS)
            mlir_time = get_gemm_time(config, GEMM_Provider.MLIR)
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
        run_gemm_perf(args.batches, gemm_sizes, fp16, timeout)
    if args.gemm_softmax_gemm:
        gemm_softmax_gemm_sizes = [(m, n, k, o) for m in sizes for n in sizes for k in sizes for o in sizes]
        run_gemm_softmax_gemm_perf(args.batches, gemm_sizes, timeout)
