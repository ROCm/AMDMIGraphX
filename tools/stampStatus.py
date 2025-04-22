from enum import Enum, auto
import subprocess, datetime, re

class StampStatus(Enum):
    """
    License Stamp Status
    """
    OK = auto(), "Stamped correctly", False
    WRONG_YEAR = auto(), "Wrong year in license", True
    NOT_STAMPED = auto(), "Missing license stamp", True
    IGNORED = auto(), "Intentionally ignored", False
    UNSUPPORTED = auto(), "Unsupported file type", False
    ERROR_READING = auto(), "File read error", False
    OTHER_LICENSE = auto(), "Other software license", False

    def __init__(self, *_args):
        self._value_ = _args[0]
        self.label = _args[1]
        self.error = _args[2]
        self.files = []


def stampCheck(file, year, content, debug=False):
    """
    Determine license stamp status for a given file's content.

    Returns:
        StampStatus: OK, WRONG_YEAR, NOT_STAMPED, or OTHER_LICENSE.
    """

    if "Advanced Micro Devices, Inc. All rights reserved" in content:
        if f"2015-{year} Advanced Micro Devices" not in content:
            if debug: print(f"{file}: Wrong year")
            return StampStatus.WRONG_YEAR
        return StampStatus.OK
    elif "Software License" in content:
        return StampStatus.OTHER_LICENSE
    else:
        return StampStatus.NOT_STAMPED

def run(cmd, cwd=None):
    return subprocess.run(cmd, capture_output=True, shell=isinstance(cmd, str), check=True, cwd=cwd).stdout.decode().strip()

def getChangedFiles(branch):
    """
    Return a list of files changed (staged) compared to the given Git branch.
    Includes only files not deleted.
    """
    head = run("git rev-parse --abbrev-ref HEAD")
    top = run("git rev-parse --show-toplevel")
    base = run(f"git merge-base {branch} {head}")
    return run(f"git diff --cached --name-only --diff-filter=d {base}", cwd=top).splitlines()

def getAllFiles(against=None):
    """
    Return a list of all tracked files in the Git repository.
    """
    top = run("git rev-parse --show-toplevel")
    return run("git ls-files", cwd=top).splitlines()

def getYearOfLatestCommit(rfile: str) -> int:
    """
    Get year of latest commit given file 
    """
    top = run("git rev-parse --show-toplevel")
    date_str = run(f"git log -1 --format=%cd --date=short {rfile}", cwd=top)
    return datetime.datetime.strptime(date_str, '%Y-%m-%d').year

def currentYear() -> int:
    """
    Get the current year based on the system clock.

    Returns:
        int: Current year in YYYY format.
    """
    return datetime.datetime.now().date().year

def updateYear(filename: str, year: int) -> None:
    """
    Update the license year in the specified file to match the latest Git commit year.

    Args:
        filename (str): File whose license year needs to be updated.
        year (int): Year to update the license to.
    """
    with open(filename, 'r') as f:
        newfileContent = re.sub(
            "2015-\d+ Advanced Micro Devices",
            f'2015-{year} Advanced Micro Devices', f.read())

    with open(filename, 'w') as f:
        f.write(newfileContent)