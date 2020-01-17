DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SRC_DIR=$DIR/../src
ls -1 $DIR/include/ | xargs -n 1 -P $(nproc) -I{} -t bash -c "python3.6 $DIR/te.py $DIR/include/{} | clang-format-5.0 -style=file > $SRC_DIR/include/migraphx/{}"

function api {
    python3.6 $DIR/api.py $SRC_DIR/api/migraphx.py $1 | clang-format-5.0 -style=file > $2
}

api $SRC_DIR/api/template/migraphx.h $SRC_DIR/api/include/migraphx/migraphx.h
api $SRC_DIR/api/template/api.cpp $SRC_DIR/api/api.cpp
