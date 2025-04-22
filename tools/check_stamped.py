#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.
#####################################################################################

import subprocess, sys, datetime, argparse, os
from stamp_status import StampStatus, stampCheck, currentYear
from git_tools import getChangedFiles

SUPPORTED_FILE_TYPES = (".cpp", ".hpp", ".h", ".ipynb", ".py", ".txt", ".sh",
                        ".bsh", "LICENSE", ".cmake")
IGNORED_FILES = ("digits.txt", "Dockerfile", "Jenkinsfile",
                 "imagenet_classes.txt", '')


def checkFile(file, debug=False):
    try:
        with open(file, 'r') as f:
            content = f.read()
    except (OSError, UnicodeDecodeError) as e:
        if debug: print(f"{file}: Skipping ({e})")
        return StampStatus.ERROR_READING

    year = currentYear()
    return stampCheck(file, year, content, debug=debug)


def print_status(status):
    files = status.files
    if files:
        print(
            f"\n{'Error' if status.error else 'Warning'}: {len(files)} {status.label} files:\n{files}"
        )
        return status.error
    return False


def main(branch, debug):
    files = getChangedFiles(branch)
    if debug: print(f"Changed files vs {branch}: {files}")

    for file in files:
        filename = os.path.basename(file)

        # Assign appropriate StampStatus based on filename or content
        if filename in IGNORED_FILES:
            status = StampStatus.IGNORED
        elif not filename.endswith(SUPPORTED_FILE_TYPES):
            status = StampStatus.UNSUPPORTED
        else:
            status = checkFile(file, debug)

        status.files.append(file)

    has_errors = any([
        print_status(StampStatus.UNSUPPORTED),
        print_status(StampStatus.IGNORED),
        print_status(StampStatus.NOT_STAMPED),
        print_status(StampStatus.WRONG_YEAR)
    ])

    if has_errors:
        sys.exit(1)
    else:
        print("Success: All files properly stamped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("branch", help="Branch to compare against")
    parser.add_argument("-d",
                        "--debug",
                        action="store_true",
                        help="Enable debug output")
    args = parser.parse_args()
    main(args.branch, args.debug)
