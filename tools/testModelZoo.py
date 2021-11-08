#!/usr/bin/env python3

import os
import git
from git import RemoteProgress
import subprocess
from time import sleep
import csv

cwd = os.getcwd()

class CloneProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        if message:
            print(message)

def cloneRepo(repoPath,cwd):
    print('Cloning into %s' % repoPath)
    git.Repo.clone_from(repoPath, cwd, progress=CloneProgress())

def isGitRepo(path):
    try:
        _ = git.Repo(path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        return False

try:
    cloneRepo("https://github.com/onnx/models.git", "models")
except:
    print("Git clone failed. Checking if path exists..")
    if os.path.exists("%s/models"%cwd):
        print("Path exists.")
        print("Checking if it is Git repository")
        if isGitRepo("%s/models"%cwd):
            print("It is a Git repo. Continuing...")
        else:
            print("It is not a Git repo. Stopping")
            os._exit_(1)
    else:
        print("You should not be here.")    

### First find the .onnx models ###
textFolderPath = "%s/models/text"%cwd
visionFolderPath = "%s/models/vision"%cwd

onnxFiles = [] #has full path of every onnx file in the zoo.

### Make sure git-lfs is installed ###
proc = subprocess.Popen('apt-get update', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=None, executable="/bin/bash")
proc = subprocess.Popen('apt-get install git-lfs', shell=True, stdin=None, stdout=open(os.devnull,"wb"), stderr=None, executable="/bin/bash")
proc.wait()

### Fetch LFS data ###
os.chdir("%s/models"%cwd)
print(os.getcwd())
os.system("git lfs pull --include=\"*\" --exclude=\"\" ") 

### Get paths for onnx files ###
for root, dirs, files in os.walk(textFolderPath):
    for file in files:
        if file.endswith(".onnx"):
             onnxFiles.append(os.path.join(root, file))

for root, dirs, files in os.walk(visionFolderPath):
    for file in files:
        if file.endswith(".onnx"):
             onnxFiles.append(os.path.join(root, file))

# migraphx-driver as binary
binaryPath = "/opt/rocm/bin/migraphx-driver" #global binary in correct rocm installatioon

### Test each ONNX model using migraphx-driver ###
with open("/tmp/test_model_zoo_output.txt", 'w') as f:
    for onnxpath in onnxFiles:
        print("TEST: %s compile %s --onnx --gpu"%(binaryPath,onnxpath))
        out = subprocess.Popen(" %s compile %s --onnx --gpu"%(binaryPath,onnxpath), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _, stderr = out.communicate()
        if(stderr):
            print("\t ERROR DETECTED")
            f.write("FAILED: %s compile %s --onnx --gpu\n"%(binaryPath,onnxpath))
            f.write("%s\n\n"%stderr.decode('utf-8'))
        else:
            f.write("PASSED: %s compile %s --onnx --gpu\n"%(binaryPath,onnxpath))
            print("\t PASSED")

print("OUTPUT FILE: /tmp/test_model_zoo_output.txt")

os._exit(0)

with open("/tmp/test_model_zoo_output.txt",'r') as f:
    content = f.readlines()
content = [x.strip() for x in content] 
for item in content:
    print(item)
os._exit(0)

summary = []
for i in range(0,len(content)-1):
    temp = []
    try:
        print(content[i])
        if ".onnx" in content[i]:
            temp.append(content[i])
            if "FAILED" in content[i]:
                while("terminate called" not in content[i+1]):
                    i = i + 1
                temp_string = ''
                while("PASSED" or "FAILED" not in content[i+1]):
                    temp_string += "%s\n"%content[i]
                    i = i + 1
                temp.append(temp_string)
            else:
                temp.append("OK")
            summary.append(temp)
    except Exception as e:
        print("Parsing exception: " + str(e))

for item in summary:
    print(item)
os._exit(0)


with open('output.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    for item in summary:
        print(item)
        wr.writerow(item)