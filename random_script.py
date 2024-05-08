import sys
mgx_lib_path = "/workspace/halilcevic/AMDMIGraphX/build/lib/"
if mgx_lib_path not in sys.path:
    sys.path.append(mgx_lib_path)
import migraphx as mgx
from cuda import cuda
from cuda import cudart
from cuda import nvrtc
from hip import hip

import ctypes
from typing import Optional, List, Union

import numpy as np

def cuda_call(call):
    err, res = call[0], call[1:]
    if len(res) == 1:
        res = res[0]
    return res

if __name__ == "__main__":
    dtype = np.dtype(np.uint8)
    nbytes = 10 * dtype.itemsize
    host_mem = int(cuda_call(cudart.cudaMallocHost(nbytes, 0)))
    print(type(host_mem))
    pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

    host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (10,))
    print(host.dtype)

    src = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.uint8)
    np.copyto(host, src)

    print(host)

    dev_mem = cuda_call(cudart.cudaMalloc(nbytes))
    print(type(dev_mem))

