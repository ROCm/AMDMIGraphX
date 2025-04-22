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
#################################### GUIDE ##########################################
#####################################################################################
# This license_stamper script is to be triggered manually via users to update the license stamps for all the
# files in the current local branch. It works by generating a list of all the files in the current active
# local branch via GIT LS-FILES and then compares each files license stamp year to the latest
# commit year (which is found via GIT LOG). It then updates the year if needed. If a license stamp is not found, then
# it will add a stamp at the begenning of the file with the year set to the current year.
#####################################################################################
import subprocess, os, datetime, re, argparse
from stamp_status import StampStatus, stampCheck, updateYear, currentYear
from git_tools import getAllFiles, getChangedFiles, getYearOfLatestCommit, getTop

delimiters = {
    ".cpp": "*",
    ".hpp": "*",
    ".h": "*",
    ".ipynb": "//",
    ".py": "#",
    ".txt": "#",
    ".bsh": "#",
    ".sh": "#",
    ".cmake": "#"
}
extensions = delimiters.keys()


# Get the delimiter from the file type based on what we care about to tag with our licence
# file. Otherwise return None for the delimiter and skip the file
def getDelimiter(filename):
    delimiter = None
    for extension in extensions:
        if extension in filename:
            delimiter = delimiters[extension]
            break
    return delimiter


def getipynb_markdownBlockAsList():
    markdownBlock = [
        '\t{\n'
        '\t\t"cell_type": "code",\n', '\t\t"execution_count": null,\n',
        '\t\t"metadata": {},\n', '\t\t"outputs": [],\n', '\t\t"source": [\n',
        '\t\t\t\"#  The MIT License (MIT)\\n\",\n', '\t\t\t\"#\\n\",\n',
        f'\t\t\t\"#  Copyright (c) 2015-{currentYear()} Advanced Micro Devices, Inc. All rights reserved.\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  Permission is hereby granted, free of charge, to any person obtaining a copy\\n\",\n',
        '\t\t\t\"#  of this software and associated documentation files (the \'Software\'), to deal\\n\",\n',
        '\t\t\t\"#  in the Software without restriction, including without limitation the rights\\n\",\n',
        '\t\t\t\"#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\\n\",\n',
        '\t\t\t\"#  copies of the Software, and to permit persons to whom the Software is\\n\",\n',
        '\t\t\t\"#  furnished to do so, subject to the following conditions:\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  The above copyright notice and this permission notice shall be included in\\n\",\n',
        '\t\t\t\"#  all copies or substantial portions of the Software.\\n\",\n',
        '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  THE SOFTWARE IS PROVIDED \'AS IS\', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\\n\",\n',
        '\t\t\t\"#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\\n\",\n',
        '\t\t\t\"#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE\\n\",\n',
        '\t\t\t\"#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\\n\",\n',
        '\t\t\t\"#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\\n\",\n',
        '\t\t\t\"#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\\n\",\n',
        '\t\t\t\"#  THE SOFTWARE.\\n\"\n', '\t\t]\n', '\t},'
    ]
    return markdownBlock


# Header and footer of the comment block
# modify these if we want some different style
def topHeader(commentChar):
    delim = None

    #Early return
    if "//" in commentChar:
        delim = getipynb_markdownBlockAsList()
        delim.append("\n")
        return ''.join(str(x) for x in delim)

    if "*" in commentChar:
        delim = "/*\n"
    if "#" in commentChar:
        delim = "#####################################################################################\n"
    return delim


def bottomFooter(commentChar):
    delim = None
    #Early return - no footer handled by
    if "//" in commentChar:
        return delim

    if "*" in commentChar:
        delim = "*/\n"
    if "#" in commentChar:
        delim = "#####################################################################################\n"
    return delim


# Simple just open and write stuff to each file with the license stamp
def openAndWriteFile(rfile,
                     message,
                     commentChar,
                     useLastCommitYear=False,
                     debug=False):

    filename = os.path.join(getTop(), rfile)
    add_shebang = False
    #markdown file stamping for .ipynb
    save_markdown_lines = []
    modify_markdown = False

    #open save old contents and append things here
    if debug is True: print("Open", filename)

    try:
        with open(filename, 'r') as contents:
            if commentChar != "//":
                saved_shebang = contents.readline()
                if "#!" in saved_shebang:
                    add_shebang = True
                else:
                    contents.seek(0)

            if commentChar == "//":
                save_markdown_lines.append(contents.readline())
                save_markdown_lines.append(contents.readline())
                modify_markdown = True

            save = contents.read()
    except (OSError, UnicodeDecodeError) as e:
        if debug:
            print(f"{e}....Skipping file")
        return

    year = getYearOfLatestCommit(rfile) if useLastCommitYear else currentYear()
    status = stampCheck(filename, year, save, debug=debug)

    if status in (StampStatus.OK, StampStatus.OTHER_LICENSE,
                  StampStatus.WRONG_YEAR):
        if status == StampStatus.WRONG_YEAR:
            updateYear(filename, year)
        return

    if debug: print("Writing header")

    with open(filename, 'w') as contents:
        if add_shebang:
            contents.write(saved_shebang + "\n")
        if modify_markdown:
            contents.write(''.join(save_markdown_lines))
        delim = topHeader(commentChar)
        if delim:
            contents.write(delim)
        if not modify_markdown:
            for line in message:
                contents.write(
                    f"{commentChar} {line}\n" if line else f"{commentChar}\n")
        delim = bottomFooter(commentChar)
        if delim:
            contents.write(delim)
        contents.write(save)

    if debug is True: print("Done")


def main(args):
    message = open(os.path.join(getTop(), 'LICENSE')).read().split('\n')

    fileList = None
    if args.all:
        fileList = getAllFiles()
    else:
        fileList = getChangedFiles(args.against)

    if args.debug is True:
        print("Target file list:\n" + '\n'.join(fileList))
        print("Output Message:\n" + '\n'.join(message))

    for rfile in fileList:
        commentDelim = getDelimiter(rfile)
        if commentDelim is not None:
            print(f"Updating file: {rfile}")
            openAndWriteFile(rfile,
                             message,
                             commentDelim,
                             useLastCommitYear=args.all,
                             debug=args.debug)
        else:
            print(f"No valid delimeter for file: {rfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('against', default='origin/develop', nargs='?')
    parser.add_argument('-a',
                        '--all',
                        action='store_true',
                        help='Update all files')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug output')
    args = parser.parse_args()
    main(args)
