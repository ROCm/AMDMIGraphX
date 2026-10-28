#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
import api, argparse, os, runpy, subprocess, sys, te
from pathlib import Path

clang_format_path = Path('clang-format.exe' if os.name ==
                         'nt' else '/opt/rocm/llvm/bin/clang-format')
work_dir = Path().cwd()
src_dir = (work_dir / '../src').absolute()
migraphx_py_path = src_dir / 'api/migraphx.py'

# Parts of the C API that a build can turn off. migraphx.py checks for these
# names in its globals to decide whether to emit the corresponding handles and
# functions.
optional_components = ['enable_onnx', 'enable_tensorflow']


def clang_format(buffer, **kwargs):
    return subprocess.run(f'{clang_format_path} -style=file',
                          capture_output=True,
                          shell=True,
                          check=True,
                          input=buffer.encode('utf-8'),
                          cwd=work_dir,
                          **kwargs).stdout.decode('utf-8')


def maybe_format(buffer, do_format=True):
    return clang_format(buffer) if do_format else buffer


def api_generate(input_path: Path, output_path: Path, do_format=True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(maybe_format(api.run(input_path), do_format))


def te_generate(input_path: Path, output_path: Path, do_format=True):
    with open(output_path, 'w') as f:
        f.write(maybe_format(te.run(input_path), do_format))


def generate_api(output_dir: Path, defines=None, do_format=True):
    runpy.run_path(str(migraphx_py_path), init_globals=defines)
    header_path = output_dir / 'include/migraphx/migraphx.h'
    source_path = output_dir / 'api.cpp'
    api_generate(work_dir / 'api/migraphx.h', header_path, do_format)
    print(f'Finished generating header {header_path}')
    api_generate(work_dir / 'api/api.cpp', source_path, do_format)
    print(f'Finished generating source {source_path}')


def generate_all(defines=None, do_format=True):
    include_dir = Path('include').absolute()
    for f in [f for f in include_dir.iterdir() if f.is_file()]:
        te_generate(f, src_dir / f'include/migraphx/{f.name}', do_format)
    # Backends under include/gpu/ generate into the gpu target tree.
    gpu_include_dir = include_dir / 'gpu'
    if gpu_include_dir.is_dir():
        for f in [f for f in gpu_include_dir.iterdir() if f.is_file()]:
            te_generate(f,
                        src_dir / f'targets/gpu/include/migraphx/gpu/{f.name}',
                        do_format)
    generate_api(src_dir / 'api', defines, do_format)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--clang-format', type=Path)
    api.add_define_argument(parser)
    parser.add_argument('--api-only',
                        action='store_true',
                        help='Only generate the C API files (migraphx.h and '
                        'api.cpp) under the directory given by '
                        '--api-output-dir instead of writing into the source '
                        'tree')
    parser.add_argument('--api-output-dir',
                        type=Path,
                        help='Base output directory for the generated C API '
                        'files: migraphx.h is written under include/migraphx/ '
                        'and api.cpp at the top level')
    args = parser.parse_args()

    if args.api_only and not args.api_output_dir:
        parser.error('--api-only requires --api-output-dir')

    global clang_format_path
    if args.clang_format:
        clang_format_path = args.clang_format

    try:
        if args.api_only:
            # These files are only consumed by the compiler, so skip
            # clang-format; only `make generate` formats them for review.
            generate_api(args.api_output_dir,
                         api.parse_defines(args.define),
                         do_format=False)
        else:
            if not clang_format_path.is_file():
                print(f"{clang_format_path}: invalid path or not installed",
                      file=sys.stderr)
                return
            # The source-tree copy is the reviewable reference, so it always
            # covers every optional component regardless of -D.
            generate_all(dict.fromkeys(optional_components, ''))
    except subprocess.CalledProcessError as ex:
        if ex.stdout:
            print(ex.stdout.decode('utf-8'))
        if ex.stderr:
            print(ex.stdout.decode('utf-8'))
        print(f"Command '{ex.cmd}' returned {ex.returncode}")
        raise


if __name__ == "__main__":
    main()
