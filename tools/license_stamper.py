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
import subprocess, os

#Debug flag
debug = False

__repo_dir__ = os.path.normpath(
    os.path.join(os.path.realpath(__file__), '..', '..'))


# Markdown code blob we should use to insert into notebook files
def getipynb_markdownBlockAsList():
    markdownBlock = [
        '\t{\n'
        '\t\t"cell_type": "code",\n', '\t\t"execution_count": null,\n',
        '\t\t"metadata": {},\n', '\t\t"outputs": [],\n', '\t\t"source": [\n',
        '\t\t\t\"#  The MIT License (MIT)\\n\",\n', '\t\t\t\"#\\n\",\n',
        '\t\t\t\"#  Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.\\n\",\n',
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


def hasKeySequence(inputfile, key_message):
    result = False
    if key_message in inputfile:
        result = True

    return result


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


#Simple just open and write stuff to each file with the license stamp
def openAndWriteFile(filename, message, commentChar):
    add_shebang = False
    #markdown file stamping for .ipynb
    save_markdown_lines = []
    modify_markdown = False

    #open save old contents and append things here
    if debug is True:
        print("Open", filename, end='')

    #with open(filename, 'r') as contents:
    #    save = contents.read()
    try:
        file = open(filename, 'r')
    except OSError as e:
        if debug is True:
            print(str(e) + "....Open Error: Skipping  file ")
        file.close()
        return
    else:
        with file as contents:
            try:
                if commentChar != "//":
                    saved_shebang = contents.readline()
                    add_shebang = hasKeySequence(saved_shebang, "#!")

                    # No shebang so start at beginning line
                    if add_shebang is False:
                        contents.seek(0)

                # Get the first tags in notebook before we insert license into a cell as a comment block
                if commentChar == "//":
                    save_markdown_lines.extend(contents.readline())  # { tag
                    save_markdown_lines.extend(
                        contents.readline())  # "cells": [ tag
                    modify_markdown = True

                #read remaining lines in the original file
                save = contents.read()

                hasAmdLic = hasKeySequence(
                    save, "Advanced Micro Devices, Inc. All rights reserved")
                hasOtherLic = hasKeySequence(save, "Software License")

                #Check if we have a licence stamp already
                if hasAmdLic or hasOtherLic is True:
                    if debug is True:
                        print("....Already Stamped: Skipping  file ")

                    contents.close()
                    return

            except UnicodeDecodeError as eu:
                if debug is True:
                    print(str(eu) + "...Skipping binary file ")
                contents.close()
                return

    if debug is True:
        print("...Writing header", end='')

    with open(filename, 'w') as contents:
        #append the licence to the top of the file

        #Append shebang before license
        if add_shebang is True:
            contents.write(saved_shebang + "\n")

        #Append markdown hooks before license
        if modify_markdown is True:
            contents.write(''.join(str(x) for x in save_markdown_lines))

        delim = topHeader(commentChar)
        if delim is not None:
            contents.write(delim)
            #print(delim)

        if modify_markdown is False:
            for line in message:
                if line != '':
                    contents.write(commentChar + " " + line + "\n")
                else:
                    contents.write(commentChar + "\n")

        delim = bottomFooter(commentChar)
        if delim is not None:
            contents.write(delim)

        #write remaining contents
        contents.write(save)
    if debug is True:
        print("...done")


# Get the file type based on what we care about to tag with our licence
# file. Otherwise return None for the delimiter and skip the file


def getDelimiter(filename):

    delimiterDict = {
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
    listOfKeys = delimiterDict.keys()
    delimiter = None

    for extension in listOfKeys:
        if extension in filename:
            delimiter = delimiterDict[extension]
            break

    return delimiter


def main():

    message = open(os.path.join(__repo_dir__, 'LICENSE')).read()

    #Get a list of all the files in our git repo
    #bashCommand = "git ls-files --exclude-standard"
    #print (bashCommand.split())
    proc = subprocess.run("git ls-files --exclude-standard",
                          shell=True,
                          stdout=subprocess.PIPE,
                          cwd=__repo_dir__)
    fileList = proc.stdout.decode().split('\n')
    message = message.split('\n')

    if debug is True:
        print("Target file list:\n" + str(fileList))
        print("Output Message:\n" + str(message))

    for rfile in fileList:
        file = os.path.join(__repo_dir__, rfile)
        #print(file)
        commentDelim = getDelimiter(file)
        if commentDelim is not None:
            openAndWriteFile(file, message, commentDelim)


if __name__ == "__main__":
    main()
