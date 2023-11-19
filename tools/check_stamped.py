#!/usr/bin/env python3
#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
import subprocess, sys
from license_stamper import getYearOfLatestCommit

debug = True
# The filetypes we want to check for that are stamped
# LICENSE is included here as it SHOULD have a license in it otherwise flag it as unstamped
supported_file_types = (".cpp", ".hpp", ".h", ".ipynb", ".py", ".txt", ".sh",
                        ".bsh", "LICENSE", ".cmake")

# add general stuff we shouldn't stamp and any exceptions here
unsupported_file_types = [
    ".onnx", ".pb", ".rst", ".jpg", ".jpeg", ".proto", ".md", ".clang",
    ".weight", ".ini", ".json", ".docker", ".git", ".rules", ".yml"
]

specificIgnores = ("digits.txt", "Dockerfile", "Jenkinsfile", "")

unsupportedFiles = []
unstampedFiles = []
stampedFilesWithBadYear = []
unknownFiles = []


def hasKeySequence(inputfile: str, key_message: str) -> bool:
    if key_message in inputfile:
        return True
    return False


# Simple just open and write stuff to each file with the license stamp
def needStampCheck(filename: str) -> bool:
    # open save old contents and append things here
    if debug: print("Open", filename, end=' ')

    try:
        file = open(filename, 'r')
    except OSError as e:
        if debug: print(str(e) + "....Open Error: Skipping  file ")
        file.close()
        return False
    else:
        with file as contents:
            try:
                save = contents.read()

                # Check if we have a license stamp already
                if hasKeySequence(
                        save,
                        "Advanced Micro Devices, Inc. All rights reserved"):
                    yearOfLastCommit = getYearOfLatestCommit(filename)
                    if not hasKeySequence(
                            save,
                            f"2015-{yearOfLastCommit[0]} Advanced Micro Devices"
                    ) and yearOfLastCommit[0] > 2022:
                        if debug: print("....Already Stamped but wrong year")
                        stampedFilesWithBadYear.append(
                            [filename, yearOfLastCommit[0],yearOfLastCommit[1]])

                    elif debug:
                        print("....Already Stamped: Skipping  file ")

                    contents.close()
                    return False

            except UnicodeDecodeError as eu:
                if debug: print(f"{str(eu)}...Skipping binary file ")
                contents.close()
                return False

    return True


# Check if any element in fileTuple is in filename
def check_filename(filename: str, fileTuple: tuple or list) -> bool:
    if any([x in filename for x in fileTuple]):
        return True
    return False


def main() -> None:
    unsupported_file_types.extend(specificIgnores)

    # Get a list of all the tracked files in our git repo
    proc = subprocess.run("git ls-files --exclude-standard",
                          shell=True,
                          stdout=subprocess.PIPE)
    fileList = proc.stdout.decode().split('\n')

    if debug: print("Target file list:\n" + str(fileList))

    for file in fileList:
        if check_filename(file, supported_file_types):
            if needStampCheck(file):
                unstampedFiles.append(file)

        elif check_filename(file, unsupported_file_types):
            unsupportedFiles.append(file)
        else:
            unknownFiles.append(file)

    # Do a bunch of checks based on our file lists
    if len(unstampedFiles) > 0:
        print("\nError: The following " + str(len(unstampedFiles)) +
              " files are currently without a license:")
        print(str(unstampedFiles))
        sys.exit(1)

    if len(stampedFilesWithBadYear) > 0:
        print("\nError: The following " + str(len(stampedFilesWithBadYear)) +
              " files licenses do not match the year of commit:")
        print(str(stampedFilesWithBadYear))
        sys.exit(1)

    if len(unknownFiles) > 0:
        print("\nError: The following " + str(len(unknownFiles)) +
              " files not handled:")
        print(str(unknownFiles))
        sys.exit(2)


if __name__ == "__main__":
    main()
