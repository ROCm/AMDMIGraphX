import os, json, subprocess, tempfile, sys, argparse, contextlib, time
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
                        default=16,
                        help='Number of instances to tune')
    args = parser.parse_args()
    return args


def tune_models(models, batch_sizes, seq_len, n):
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

    tc.tune(log_file, n, json_file)
    print("\nTuning results have been saved to:\n{}\n".format(json_file))


def run(args):
    tune_models(args.models, args.batch_sizes, args.sequence_length, args.n)


run(parse_args())
