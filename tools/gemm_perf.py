import subprocess, csv, re

def get_device_name():
    out = subprocess.run("rocminfo",
                         capture_output=True,
                         check=True,
                         shell=True)
    matches = re.findall("gfx\d*[a-z]*", str(out.stdout))
    return matches[0]

def run_perf(model, batch_size, int8=False, use_ck=False, use_large_k=False, disable_fusion=False):
    env_vars = ""
    if use_ck:
        env_vars += "MIGRAPHX_ENABLE_CK=1 " 
        if use_large_k:
            env_vars += "MIGRAPHX_USE_LARGE_K=1 " 
        if disable_fusion:
            env_vars += "MIGRAPHX_DISABLE_CK_FUSION=1 "
    int8_str = "--int8" if int8 else ""
    cmd = "{env_vars} ../build/bin/driver perf {model} --fill1 input_ids --input-dim @input_ids {batch_size} 384 --batch {batch_size} --fp16 {int8}  --exhaustive-tune".format(
            env_vars=env_vars,
            model=model,
            batch_size=str(batch_size),
            int8=int8_str
    )
    out = subprocess.run(cmd,
                         capture_output=True,
                         check=True,
                         shell=True)
    summary = re.findall("Summary.*", str(out.stdout))[0].replace("\\n", "\n")
    total_time = re.findall("Total time: \d+\.\d*", summary)[0]
    total_time = total_time.replace("Total time: ", "")

    print(summary)
    print(total_time)
    with open("summaries.txt", "w+") as f:
        f.write(cmd + "\n")
        f.write(summary + "\n\n")


# run model with:
#    RocBlas 
#        Get gemm info
#    CK
#        With fusions
#        Without fusions



if __name__ == "__main__":
    device_id = get_device_name()
    model = "/code/bert_base_cased_1_fp16_gpu.onnx"
    run_perf(model, 1, True, True, True, True)