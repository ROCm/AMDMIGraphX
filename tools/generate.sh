#####################################################################################
# The MIT License (MIT)
#
# Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#####################################################################################
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLANG_FORMAT=/opt/rocm/llvm/bin/clang-format
SRC_DIR=$DIR/../src
PYTHON=python3
if type -p python3.6 > /dev/null ; then
    PYTHON=python3.6
fi
if type -p python3.8 > /dev/null ; then
    PYTHON=python3.8
fi
ls -1 $DIR/include/ | xargs -n 1 -P $(nproc) -I{} -t bash -c "$PYTHON $DIR/te.py $DIR/include/{} | $CLANG_FORMAT -style=file > $SRC_DIR/include/migraphx/{}"

function api {
    $PYTHON $DIR/api.py $SRC_DIR/api/migraphx.py $1 | $CLANG_FORMAT -style=file > $2
}

api $DIR/api/migraphx.h $SRC_DIR/api/include/migraphx/migraphx.h
echo "Finished generating header migraphx.h"
api $DIR/api/api.cpp $SRC_DIR/api/api.cpp
echo "Finished generating source api.cpp "
