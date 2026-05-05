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

from enum import Enum, auto
import datetime
import re


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


def stamp_check(file, year, content, debug=False):
    """
    Determine license stamp status for a given file's content.

    Returns:
        StampStatus: OK, WRONG_YEAR, NOT_STAMPED, or OTHER_LICENSE.
    """

    if "Advanced Micro Devices, Inc. All rights reserved" in content:
        year_pattern = re.compile(
            rf"Copyright \(c\) (?:\d{{4}}\s*-\s*)?{year} Advanced Micro Devices"
        )
        if year_pattern.search(content):
            return StampStatus.OK
        if debug:
            print(f"{file}: Wrong year")
        return StampStatus.WRONG_YEAR
    if "Software License" in content:
        return StampStatus.OTHER_LICENSE

    return StampStatus.NOT_STAMPED


def current_year():
    """
    Get the current year based on the system clock.

    Returns:
        int: Current year in YYYY format.
    """
    return datetime.datetime.now().date().year


def update_year(filename: str, year: int):
    """
    Update the license year in the specified file to match the latest Git commit year.

    Args:
        filename (str): File whose license year needs to be updated.
        year (int): Year to update the license to.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    def replacer(match):
        start_year = match.group(1)
        return f"Copyright (c) {start_year}-{year} Advanced Micro Devices"

    new_content = re.sub(
        r"Copyright \(c\) (\d{4})(?:\s*-\s*\d{4})? Advanced Micro Devices",
        replacer, content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_content)
