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
import os
import shutil
import argparse
import subprocess
from git_tools import get_all_files, get_changed_files, get_merge_base, get_top, run

DEFAULT_CLANG_FORMAT = 'clang-format'

EXCLUDE_FILES = ['requirements.in', 'onnx.proto']

CLANG_EXTENSIONS = ('.c', '.cpp', '.hpp', '.h', '.cl', '.hip', '.in')
YAPF_EXTENSIONS = ('.py')


def is_excluded(f):
    base = os.path.basename(f)
    return base in EXCLUDE_FILES


def clang_format(against,
                 apply=False,
                 all_files=False,
                 clang_format=DEFAULT_CLANG_FORMAT):
    if not shutil.which(clang_format):
        print(f"{clang_format} not installed. Skipping format.")
        return
    files = get_all_files() if all_files else get_changed_files(against)
    files = [
        f for f in files if f.endswith(CLANG_EXTENSIONS) and not is_excluded(f)
    ]
    if not files:
        print("No modified cpp files to format")
        return
    if all_files:
        mode_flags = ['-i'] if apply else ['--dry-run', '-Werror']
        run([clang_format] + mode_flags + files, cwd=get_top(), verbose=True)
        return
    git_clang_format = os.path.join(os.path.dirname(clang_format),
                                    'git-clang-format')
    if not shutil.which(git_clang_format):
        print(f"{git_clang_format} not installed. Skipping format.")
        return
    diff_flag = [] if apply else ["--diff"]
    base = get_merge_base(against)
    run([git_clang_format, '--binary', clang_format] + diff_flag + [base] +
        files,
        cwd=get_top(),
        verbose=True)


def yapf_format(against, apply=False, all_files=False, yapf='yapf'):
    if not shutil.which(yapf):
        print(f"{yapf} not installed. Skipping format.")
        return
    diff_flag = "--in-place" if apply else "--diff"
    files = get_all_files() if all_files else get_changed_files(against)
    files = [
        f for f in files if f.endswith(YAPF_EXTENSIONS) and not is_excluded(f)
    ]
    if files:
        run([yapf, diff_flag, '-p'] + files, cwd=get_top(), verbose=True)
    else:
        print("No modified python files to format")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('against', default='develop', nargs='?')
    parser.add_argument('-i', '--in-place', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    parser.add_argument('-a',
                        '--all',
                        action='store_true',
                        help='Format all tracked files, not just changed ones')
    parser.add_argument('--clang-format-path',
                        default=DEFAULT_CLANG_FORMAT,
                        help='Path to the clang-format executable')
    parser.add_argument('--yapf-path',
                        default='yapf',
                        help='Path to the yapf executable')
    parser.add_argument('--exit-zero',
                        action='store_true',
                        help='Exit 0 even when formatting differs')
    args = parser.parse_args()
    try:
        # clang_format(args.against,
        #              apply=args.in_place,
        #              all_files=args.all,
        #              clang_format=args.clang_format_path)
        yapf_format(args.against,
                    apply=args.in_place,
                    all_files=args.all,
                    yapf=args.yapf_path)
    except subprocess.CalledProcessError as ex:
        if ex.stdout:
            print(ex.stdout)
        if ex.stderr:
            print(ex.stderr)
        if args.exit_zero:
            return
        if not args.quiet:
            print(f"Command '{ex.cmd}' returned {ex.returncode}")
            raise
            # sys.exit(ex.returncode)


if __name__ == "__main__":
    main()
