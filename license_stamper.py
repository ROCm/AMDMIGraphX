from fileinput import filename
from operator import contains
import sys
import subprocess

# Header and footer of the comment block 
# modify these if we want some different style
def topHeader(commentChar):
    delim = None
    if "*" in commentChar:
        delim = "/*"
    if  "#" in commentChar:
        delim = "###########################################"
    return delim

def bottomFooter(commentChar):
    delim = None
    if "*" in commentChar:
        delim = "*/"
    if  "#" in commentChar:
        delim = "###########################################"
    return delim

#Simple just open and write stuff to each file with the license stamp
def openAndWriteFile(filename, message, commentChar):
    print("Writing header to ", filename)

    #open save old contents and append things here
    with open(filename, 'r') as contents:
        save = contents.read()
    with open(filename, 'w') as contents:
        #append the licence to the top of the file

        delim = topHeader(commentChar)
        if delim is not None:
            contents.write(delim + "\n")

        for line in message:
            contents.write(commentChar + " " + line + "\n")

        delim = bottomFooter(commentChar)
        if delim is not None:
            contents.write(delim + "\n")

        #write remaining contents    
        contents.write(save)

    print("done")

# Get the file type based on what we care about to tag with our licence
# file. Otherwise return None for the delimiter and skip the file

def getDelimiter(filename):

    delimiter = None
    if ".cpp" in filename:
        delimiter = "* " 
    if ".hpp" in filename:
        delimiter = "* "
    if ".py" in filename:
        delimiter = "# "
    if ".txt" in filename:
        delimiter = "# "

    return delimiter 

def main():
    #figure out what file we want to read in our message from
#    liscence_file = input("Enter License file (default is LICENSE)")

    #if liscence_file is None:
#        licence_file = "LICENSE"

    bashCommand = "cat LICENSE"
    process = subprocess.Popen(bashCommand.split(), stdout= subprocess.PIPE)
    message, error = process.communicate()

    if error == None:
        #Get a list of all the files in our git repo
        bashCommand = "git ls-files --exclude-standard"
        process = subprocess.Popen(bashCommand.split(), stdout= subprocess.PIPE)
        output, error = process.communicate()
        if error is None:
            fileList = output.splitlines()
            message  = message.splitlines()
            #print(message)
            for file in fileList:
                commentDelim = getDelimiter(file)
                if commentDelim is not None:
                    openAndWriteFile(file, message, commentDelim)

        else:
            print ("error parsing in list of files")
    else:
        print ("error parsing in license file")

if __name__ == "__main__":
    main()