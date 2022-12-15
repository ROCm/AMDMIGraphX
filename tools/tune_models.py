import os, subprocess, argparse, time, json, difflib
import tune_ck as tc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune CK GEMMs for one or more ONNX models")
    parser.add_argument('--models',
                        '-m',
                        nargs='+',
                        help='ONNX models to be tuned',
                        required=True)
    parser.add_argument('--batch_sizes',
                        '-b',
                        nargs='+',
                        help='Batch sizes to tune',
                        required=True)
    parser.add_argument('--sequence_length',
                        '-s',
                        type=int,
                        default=384,
                        help='Sequence length for transformer models')
    parser.add_argument('-n',
                        type=int,
                        default=18,
                        help='Number of instances to tune')
    parser.add_argument('--update',
                        '-u',
                        type=str,
                        help='Existing tuning JSON. Configs already present will not be re-tuned.')
    args = parser.parse_args()
    return args


def tune_models(models, batch_sizes, seq_len, n, existing):
    time_stamp = time.strftime("%Y_%m_%d_%H_%M")
    log_file = "ck_tuning_{}.log".format(time_stamp)
    json_file = "ck_tuning_{}.json".format(time_stamp)
    for model in models:
        for batch in batch_sizes:
            out = subprocess.run(
                'MIGRAPHX_LOG_CK_GEMM=1 ../build/bin/driver run {} -g --fill1 input_ids --input-dim @input_ids {} {}  | grep \'ck_gemm.*: \[{{\' | sort -u >> {}'
                .format(model, batch, seq_len, log_file),
                capture_output=True,
                check=True,
                shell=True)

    if (existing is not None):
        f = open(existing)
        configs = json.load(f)
        configs = [str(s).replace(" ", "") for l in configs for s in l]
        update_logs = []
        with open(log_file, "r") as lf:
            logs = [line for line in lf]
            stripped_logs = [line.replace("ck_gemm: ", "").replace("ck_gemm_softmax_gemm: ", "").replace("\"", "'").replace("\n", "") for line in logs]
            
            for i in range(len(stripped_logs)):
                if (stripped_logs[i] not in configs):
                    update_logs.append(logs[i])
        
        with open(log_file, "w") as lf:
            for line in update_logs:
                lf.write(line)
        
        f.close()

    tc.tune(log_file, n, json_file)

    if (existing is not None):
        f_old = open(existing, "r")
        f_new = open(json_file, "r")
        old = json.load(f_old)
        new = json.load(f_new)
        new = old + new
        f_old.close()
        f_new.close()
        json.dump(new, open(json_file, "w"))

    tuning_path = os.path.abspath(json_file)
    os.environ["MIGRAPHX_CK_TUNING"] = tuning_path
    print("\nTuning results have been saved to:\n{}\n".format(json_file))


def run(args):
    tune_models(args.models, args.batch_sizes, args.sequence_length, args.n, args.update)


run(parse_args())
