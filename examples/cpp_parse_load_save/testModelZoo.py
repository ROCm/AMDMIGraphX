#!/usr/bin/env python3

# Need git lfs installed

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

# First find the .onnx models in /text/

textFolderPath = "%s/models/text"%cwd
visionFolderPath = "%s/models/vision"%cwd

onnxFiles = []

for root, dirs, files in os.walk(textFolderPath):
    for file in files:
        if file.endswith(".onnx"):
             onnxFiles.append(os.path.join(root, file))

for root, dirs, files in os.walk(visionFolderPath):
    for file in files:
        if file.endswith(".onnx"):
             onnxFiles.append(os.path.join(root, file))

# 	2. git lfs pull --include="/vision/classification/resnet/model/resnet101-v1-7.onnx" --exclude=""

#os.chdir("%s/models"%cwd)
#print(os.getcwd())
#os.system("git lfs pull --include=\"*\" --exclude=\"\" ") 

binaryPath = textFolderPath = "%s/build/parse_load_save"%cwd
# with open('out.txt', 'w') as f:
#     print('Filename:', file=f)  # Python 3.x
# for onnxpath in onnxFiles:
#     print("%s %s --parse onnx"%(binaryPath,onnxpath))
#     out = os.popen("%s %s --parse onnx"%(binaryPath,onnxpath)).read()
#     #print(out)

# exit

with open("%s/output.txt"%cwd) as f:
    content = f.readlines()
content = [x.strip() for x in content] 

summary = []

for i in range(0,len(content)):
    print(content[i])
    temp = []
    try:
        if ".onnx" in content[i]:
            temp.append(content[i])
            if "MIGraphX Error" in content[i+1]:
                temp.append(content[i+1])
            else:
                temp.append("OK")
            summary.append(temp)
    except:
        print(i)

print(summary)

with open('output.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    for item in summary:
        print(item)
        wr.writerow(item)