#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
import os, shutil, argparse, subprocess

CLANG_FORMAT_PATH = '/opt/rocm/llvm/bin'


def run(cmd, **kwargs):
    print(cmd)
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def eval(cmd, **kwargs):
    return subprocess.run(cmd,
                          capture_output=True,
                          shell=True,
                          check=True,
                          **kwargs).stdout.decode('utf-8').strip()


def get_top():
    return eval("git rev-parse --show-toplevel")


def get_head():
    return eval("git rev-parse --abbrev-ref HEAD")


def get_merge_base(branch):
    head = get_head()
    return eval(f"git merge-base {branch} {head}")


def clang_format(against, apply=False, path=CLANG_FORMAT_PATH):
    base = get_merge_base(against)
    clang_format = os.path.join(path, 'clang-format')
    if not os.path.exists(clang_format):
        print(f"{clang_format} not installed. Skipping format.")
        return
    git_clang_format = os.path.join(path, 'git-clang-format')
    if not os.path.exists(git_clang_format):
        print(f"{git_clang_format} not installed. Skipping format.")
        return
    diff_flag = "" if apply else "--diff"
    run(f"{git_clang_format} --binary {clang_format} {diff_flag} {base}")


def get_files_changed(against, ext=('py')):
    files = eval(f"git diff-index --cached --name-only {against}",
                 cwd=get_top()).splitlines()
    return (f for f in files if f.endswith(ext))


def yapf_format(against, apply=False):
    if not shutil.which('yapf'):
        print("yapf not installed. Skipping format.")
        return
    diff_flag = "--in-place" if apply else "--diff"
    files = ' '.join(get_files_changed(against))
    if files:
        run(f"yapf {diff_flag} -p {files}")
    else:
        print("No modified python files to format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('against', default='develop', nargs='?')
    parser.add_argument('-i', '--in-place', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    try:
        clang_format(args.against, apply=args.in_place)
        yapf_format(args.against, apply=args.in_place)
    except subprocess.CalledProcessError as ex:
        if ex.stdout:
            print(ex.stdout.decode('utf-8'))
        if ex.stderr:
            print(ex.stderr.decode('utf-8'))
        if not args.quiet:
            print(f"Command '{ex.cmd}' returned {ex.returncode}")
            raise
            # sys.exit(ex.returncode)


if __name__ == "__main__":
    main()
