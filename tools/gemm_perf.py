import subprocess, csv, re, datetime


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


if __name__ == "__main__":
    device_id = get_device_name()
    model = "/code/bert_base_cased_1_fp16_gpu.onnx"
    cf = CSVFile()
    cf.write_row([str(datetime.datetime.now())])
    cf.write_row([device_id])
    cf.write_row([model])
    headers = [
        "", "Total Time (ms)", "CK GEMM Time (ms)", "RB GEMM Time (ms)",
        "GEMM Pack Time (ms)", "Total GEMM Time (ms)"
    ]

    batch_size = "1"
    # int8:
    quantize = True
    label = f"Int8 / BatchSize: {batch_size}" if quantize else f"FP16 / BatchSize: {batch_size}"
    cf.write_row([label])
    cf.write_row(headers)
    # CK Only
    cf.write_row(["CK"] + run_ck_perf(model, batch_size, quantize, True))
    # CK + rocBLAS (k>2048)
    cf.write_row(["CK + rocBLAS(k>2048)"] +
                 run_ck_perf(model, batch_size, quantize, False))
    # rocBLAS Only
    cf.write_row(["rocBLAS"] + run_perf(model, batch_size, quantize))
    cf.write_row()

    # fp16:
    quantize = False
    label = f"Int8 / BatchSize: {batch_size}" if quantize else f"FP16 / BatchSize: {batch_size}"
    cf.write_row([label])
    cf.write_row(headers)
    # CK Only
    cf.write_row(["CK"] + run_ck_perf(model, batch_size, quantize, True))
    # CK + rocBLAS (k>2048)
    cf.write_row(["CK + rocBLAS(k>2048)"] +
                 run_ck_perf(model, batch_size, quantize, False))
    # rocBLAS Only
    cf.write_row(["rocBLAS"] + run_perf(model, batch_size, quantize))
    cf.write_row()

    batch_size = "64"
    # int8:
    quantize = True
    label = f"Int8 / BatchSize: {batch_size}" if quantize else f"FP16 / BatchSize: {batch_size}"
    cf.write_row([label])
    cf.write_row(headers)
    # CK Only
    cf.write_row(["CK"] + run_ck_perf(model, batch_size, quantize, True))
    # CK + rocBLAS (k>2048)
    cf.write_row(["CK + rocBLAS(k>2048)"] +
                 run_ck_perf(model, batch_size, quantize, False))
    # rocBLAS Only
    cf.write_row(["rocBLAS"] + run_perf(model, batch_size, quantize))
    cf.write_row()

    # fp16:
    quantize = False
    label = f"Int8 / BatchSize: {batch_size}" if quantize else f"FP16 / BatchSize: {batch_size}"
    cf.write_row([label])
    cf.write_row(headers)
    # CK Only
    cf.write_row(["CK"] + run_ck_perf(model, batch_size, quantize, True))
    # CK + rocBLAS (k>2048)
    cf.write_row(["CK + rocBLAS(k>2048)"] +
                 run_ck_perf(model, batch_size, quantize, False))
    # rocBLAS Only
    cf.write_row(["rocBLAS"] + run_perf(model, batch_size, quantize))
    cf.write_row()
