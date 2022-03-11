DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR=$DIR/../src
PYTHON=python3
if type -p python3.6 > /dev/null ; then
    PYTHON=python3.6
fi
if type -p python3.8 > /dev/null ; then
    PYTHON=python3.8
fi
ls -1 $DIR/include/ | xargs -n 1 -P $(nproc) -I{} -t bash -c "$PYTHON $DIR/te.py $DIR/include/{} | clang-format-10 -style=file > $SRC_DIR/include/migraphx/{}"

function api {
    $PYTHON $DIR/api.py $SRC_DIR/api/migraphx.py $1 | clang-format-10 -style=file > $2
}

api $DIR/api/migraphx.h $SRC_DIR/api/include/migraphx/migraphx.h
echo "Finished generating header migraphx.h"
api $DIR/api/api.cpp $SRC_DIR/api/api.cpp
echo "Finished generating source api.cpp "
