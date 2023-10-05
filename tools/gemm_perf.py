#%matplotlib
import subprocess, csv, re, datetime, argparse, os
from subprocess import STDOUT


def parse_args():
    parser = argparse.ArgumentParser(description="GEMM performance tools")
    parser.add_argument(
        '--gemm',
        action='store_true',
        help='Run performance comparison on a range of GEMM problem sizes')
    args = parser.parse_args()

    return args


class CSVFile:

    def __init__(self, path="output.csv"):
        self.path = path

    def write_row(self, row=[]):
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


def run_perf(model,
             batch_size,
             int8=False,
             use_ck=False,
             use_large_k=False,
             disable_fusion=False):
    env_vars = ""
    if use_ck:
        env_vars += "MIGRAPHX_ENABLE_CK=1 "
        if use_large_k:
            env_vars += "MIGRAPHX_USE_LARGE_K=1 "
        if disable_fusion:
            env_vars += "MIGRAPHX_DISABLE_CK_FUSION=1 "
    int8_str = "--int8" if int8 else ""
    cmd = f"{env_vars} ../build/bin/driver perf {model} --fill1 input_ids --input-dim @input_ids {batch_size} 384 --batch {batch_size} --fp16 {int8_str}  --exhaustive-tune"
    out = subprocess.run(cmd, capture_output=True, check=True, shell=True)

    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")

    ck_gemm_time = re.findall("ck_gemm_kernel: \d+\.\d*", summary)
    if ck_gemm_time:
        ck_gemm_time = re.findall("\d+\.\d*", ck_gemm_time[0])[0]
    else:
        ck_gemm_time = "0.0"

    rb_gemm_time = re.findall("gpu::quant_gemm: \d+\.\d*|gpu::gemm: \d+\.\d*",
                              summary)
    if rb_gemm_time:
        rb_gemm_time = re.findall("\d+\.\d*", rb_gemm_time[0])[0]
    else:
        rb_gemm_time = "0.0"

    gemm_pack_time = re.findall("gpu::int8_gemm_pack_a: \d+\.\d*", summary)
    if gemm_pack_time:
        gemm_pack_time = re.findall("\d+\.\d*", gemm_pack_time[0])[0]
    else:
        gemm_pack_time = "0.0"

    gemm_times = [ck_gemm_time, rb_gemm_time, gemm_pack_time]
    total_gemm_time = [str(sum(map(float, gemm_times)))]
    gemm_times.extend(total_gemm_time)

    print(cmd)
    print(total_time + "ms")
    with open("perf_summaries.txt", "a+") as f:
        f.write(cmd + "\n")
        f.write(summary + "\n\n")

    return [total_time] + gemm_times


def run_ck_perf(model, batch_size, int8=False, use_large_k=False):
    # CK with fusions
    total_time = run_perf(model, batch_size, int8, True, use_large_k, False)[0]
    # CK without fusions
    gemm_times = run_perf(model, batch_size, int8, True, use_large_k, True)

    return [total_time] + gemm_times[1:]


def gemm_perf(b, m, n, k, fp16):
    model = "../test/onnx/matmul_half.onnx" 
    prec_str = "--fp16" if fp16 else "--int8"
    #rocBLAS run
    cmd = f"../build/bin/driver perf {model} --input-dim @1 {b} {m} {k} @2 {b} {k} {n} {prec_str}"
    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=600,
                             env=dict(os.environ, MIGRAPHX_ENABLE_CK="0"))
        # print(out.stderr)
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        print(f"{b}, {m}, {n}, {k}:", end=" ")
        print("-100.0, -100.0")
        return -100.0, -100.0
    
    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    # print(summary)
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")
    rb_time = total_time
    rb_gemm_time = re.findall("gpu::quant_gemm: \d+\.\d*|gpu::gemm: \d+\.\d*",
                              summary)
    rb_gemm_time = re.findall("\d+\.\d*", rb_gemm_time[0])[0]

    cmd = f"../build/bin/driver perf {model} --input-dim @1 {b} {m} {k} @2 {b} {k} {n} {prec_str} --exhaustive-tune"
    try:
        out = subprocess.run(cmd.split(),
                             capture_output=True,
                             check=True,
                             timeout=600,
                             env=dict(os.environ, 
                                      MIGRAPHX_ENABLE_CK="1",
                                      MIGRAPHX_USE_LARGE_K="1",
                                      MIGRAPHX_DISABLE_CK_FUSION="1"))
        # print(out.stderr)
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        print(f"{b}, {m}, {n}, {k}:", end=" ")
        print("100.0, 100.0")
        return 100.0, 100.0

    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    # print(summary)
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")
    ck_time = total_time
    ck_gemm_time = re.findall("ck_gemm_kernel: \d+\.\d*", summary)
    ck_gemm_time = re.findall("\d+\.\d*", ck_gemm_time[0])[0]

    total_diff = float(ck_time) - float(rb_time)
    kernel_diff = float(ck_gemm_time) - float(rb_gemm_time)
    print(f"{b}, {m}, {n}, {k}:", end=" ")
    print(f"{total_diff}, {kernel_diff}")
    with open(f"gemm_log.txt", "a+") as f:
        f.write(f"{b},{m},{n},{k},{total_diff},{kernel_diff}\n")
    return total_diff, kernel_diff


def run_gemm_perf():
    batches = [128]
    sizes = [64, 256, 384, 768, 1024, 2048, 2304, 3072]
    # batches = [768]
    # sizes = [64, 384, 768, 2304]
    fp16 = True
    prec_str = "half" if fp16 else "int"
    # problems = [(1, 384, 384, 64),
    #             (64, 384, 384, 64),
    #             (1, 384, 64, 384),
    #             (64, 384, 64, 384)]
    # for p in problems:
    #     with open(f"gemm_log.txt", "a+") as f:
    #         timestamp = str(datetime.datetime.now())
    #         f.write(f"{timestamp}\n")
    #     b, m, n, k = p
    #     gemm_perf(b*12, m, n, k, fp16)
    for b in batches:
        with open(f"gemm_log.txt", "a+") as f:
            timestamp = str(datetime.datetime.now())
            f.write(f"{timestamp}\n")
        results = [(b, m, n, k, gemm_perf(b, m, n, k, fp16)) for m in sizes for n in sizes for k in sizes]
        with open(f"gemm_results_{b}_{prec_str}.txt", "w+") as f:
            for r in results:
                f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4][0]},{r[4][1]}\n")

    # fp16 = True
    # prec_str = "half" if fp16 else "int"
    # for b in batches:
    #     with open(f"gemm_log.txt", "a+") as f:
    #         timestamp = str(datetime.datetime.now())
    #         f.write(f"{timestamp}\n")
    #     results = [(b, m, n, k, gemm_perf(b, m, n, k, fp16)) for m in sizes for n in sizes for k in sizes]
    #     with open(f"gemm_results_{b}_{prec_str}.txt", "w+") as f:
    #         for r in results:
    #             f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4][0]},{r[4][1]}\n")

if __name__ == "__main__":
    args = parse_args()
    if args.gemm:
        run_gemm_perf()
