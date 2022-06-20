#!/usr/bin/env python3
#####################################################################################
#  The MIT License (MIT)
#
#  Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#Debug flag
debug = True


def hasKeySequence(inputfile, key_message):
    result = False
    if key_message in inputfile:
        result = True

    return result


# Header and footer of the comment block
# modify these if we want some different style
def topHeader(commentChar):
    delim = None
    if "*" in commentChar:
        delim = "/*"
    if "#" in commentChar:
        delim = "#####################################################################################"
    return delim


def bottomFooter(commentChar):
    delim = None
    if "*" in commentChar:
        delim = "*/"
    if "#" in commentChar:
        delim = "#####################################################################################"
    return delim


#Simple just open and write stuff to each file with the license stamp
def openAndWriteFile(filename, message, commentChar):
    add_shebang = False

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
                saved_shebang = contents.readline()
                add_shebang = hasKeySequence(saved_shebang, "#!")

                # No shebang so start at beginning line
                if add_shebang is False:
                    contents.seek(0)

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

        if add_shebang is True:
            contents.write(saved_shebang + "\n")

        delim = topHeader(commentChar)
        if delim is not None:
            contents.write(delim + "\n")

        for line in message:
            if line is not '':
                contents.write(commentChar + " " + line + "\n")
            else:
                contents.write(commentChar + "\n")

        delim = bottomFooter(commentChar)
        if delim is not None:
            contents.write(delim + "\n")

        #write remaining contents
        contents.write(save)

    print("...done")


# Get the file type based on what we care about to tag with our licence
# file. Otherwise return None for the delimiter and skip the file


def getDelimiter(filename):

    delimiter = None
    if ".cpp" in filename:
        delimiter = "*"
    if ".hpp" in filename:
        delimiter = "*"
    if ".py" in filename:
        delimiter = "#"
    if ".txt" in filename:
        delimiter = "#"

    return delimiter


def main():

    message = open('LICENSE').read()

    #Get a list of all the files in our git repo
    #bashCommand = "git ls-files --exclude-standard"
    #print (bashCommand.split())
    proc = subprocess.run("git ls-files --exclude-standard",
                          shell=True,
                          stdout=subprocess.PIPE)
    fileList = proc.stdout.decode().split('\n')
    message = message.split('\n')

    if debug is True:
        print("Target file list:\n" + str(fileList))
        print("Output Message:\n" + str(message))

    for file in fileList:
        #print(file)
        commentDelim = getDelimiter(file)
        if commentDelim is not None:
            openAndWriteFile(file, message, commentDelim)


if __name__ == "__main__":
    main()
