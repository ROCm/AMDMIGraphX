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
import subprocess
import shlex
import datetime


def run(cmd, cwd=None, verbose=False, text=True, **kwargs):
    """
    Run a shell command with optional output capture and verbosity.

    Args:
        cmd (str | list): The command to run.
        capture (bool): If True, capture and return the stdout as a string.
        verbose (bool): If True, print the command being run.
        text (bool): If True, decode stdout as UTF-8 (default: True).
        **kwargs: Extra arguments passed to subprocess.run().

    Returns:
        str | None: If capture is True, returns the command's stdout as a string; otherwise None.
    """
    if verbose:
        print(cmd if isinstance(cmd, str) else shlex.join(cmd))

    result = subprocess.run(cmd,
                            shell=isinstance(cmd, str),
                            check=True,
                            capture_output=True,
                            text=text,
                            cwd=cwd,
                            **kwargs).stdout.strip()

    if verbose:
        print(result)

    return result


def getTop():
    """
    Return the absolute path to the root directory of the Git repository.
    Useful as the base path for running Git commands or locating files.
    """
    return run("git rev-parse --show-toplevel")


def getMergeBase(branch):
    """
    Get the common ancestor (merge base) commit hash between the current branch
    and the given branch. This is the commit where they diverged, used to compare changes.
    """
    head = run("git rev-parse --abbrev-ref HEAD")  # Get current branch name
    return run(f"git merge-base {branch} {head}"
               )  # Get merge base SHA between current and target branch


def getChangedFiles(branch):
    """
    Return a list of files that are staged for commit and have changed
    compared to the merge base with the given branch.
    Only includes modified or added files (excludes deleted files).
    """
    base = getMergeBase(branch)
    return run(f"git diff --cached --name-only --diff-filter=d {base}",
               cwd=getTop()).splitlines()


def getAllFiles():
    """
    Return a list of all tracked files in the Git repository.
    """
    return run("git ls-files", cwd=getTop()).splitlines()


def getYearOfLatestCommit(file):
    """
    Get year of latest commit given file 
    """
    date_str = run(f"git log -1 --format=%cd --date=short {file}",
                   cwd=getTop())
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').year
