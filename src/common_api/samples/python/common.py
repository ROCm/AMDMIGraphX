import os
from typing import Optional, Union, List
import argparse
import ctypes
import numpy as np

from hip import hip
import migraphx.common_api as trt


# NOTE: Defined here until an __init__.py is created for the migraphx module
def nptype(trt_type):

    mapping = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int8: np.int8,
        trt.int32: np.int32,
        trt.int64: np.int64,
        trt.bool: np.bool_,
        trt.uint8: np.uint8,
    }
    if trt_type in mapping:
        return mapping[trt_type]
    raise TypeError("Could not resolve TensorRT datatype to an equivalent numpy datatype.")


def locate_files(data_paths, filenames, err_msg=""):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError(
                "Could not find {:}. Searched in data paths: {:}\n{:}".format(
                    filename, data_paths, err_msg
                )
            )
    return found_files

def find_sample_data(
    description="Runs a TensorRT Python sample", subfolder="", find_files=[], err_msg=""
):
    """
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    """

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "code", "AMDMIGraphX", "src", "common_api", "samples_data")
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-d",
        "--datadir",
        help="Location of the TensorRT sample data directory, and any additional data directories.",
        action="append",
        default=[kDEFAULT_DATA_ROOT],
    )
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            if data_dir != kDEFAULT_DATA_ROOT:
                print(
                    "WARNING: "
                    + data_path
                    + " does not exist. Trying "
                    + data_dir
                    + " instead."
                )
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)) and data_dir != kDEFAULT_DATA_ROOT:
            print(
                "WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(
                    data_path
                )
            )
        return data_path

    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    return data_paths, locate_files(data_paths, find_files, err_msg)

def check_cuda_err(err):
    if isinstance(err, hip.hipError_t):
        if err != hip.hipError_t.hipSuccess:
            raise RuntimeError("HIP Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))

def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: Optional[np.dtype] = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        # NOTE: The cast to int is required to circumvent a difference in the hip and cuda python packages
        #       Cuda returns an int, which trt expects, while hip returns a hip._util.types.Pointer
        host_mem = int(cuda_call(hip.hipMallocHost(nbytes)))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(hip.hipMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: Union[np.ndarray, bytes]):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[:data.size], data.flat, casting='safe')
        else:
            assert self.host.dtype == np.uint8
            self.host[:self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(hip.hipFree(self.device))
        cuda_call(hip.hipFreeHost(self.host.ctypes.data))

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine, profile_idx: Optional[int] = None):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda_call(hip.hipStreamCreate())
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for binding in tensor_names:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        shape = engine.get_tensor_shape(binding) #if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(f"Binding {binding} has dynamic shape, " +\
                "but no profile was specified.")
        size = trt.volume(shape)
        trt_type = engine.get_tensor_dtype(binding)

        # Allocate host and device buffers
        # TODO add nptype to common_api python api
        if nptype(trt_type):
            dtype = np.dtype(nptype(trt_type))
            bindingMemory = HostDeviceMem(size, dtype)
        else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            size = int(size * trt_type.itemsize)
            bindingMemory = HostDeviceMem(size)

        # Append the device buffer to device bindings.
        bindings.append(int(bindingMemory.device))

        # Append to the appropriate list.
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            inputs.append(bindingMemory)
        else:
            outputs.append(bindingMemory)
    return inputs, outputs, bindings, stream

def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: hip.hipStream_t):
    for mem in inputs + outputs:
        mem.free()
    cuda_call(hip.hipStreamDestroy(stream))

def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = hip.hipMemcpyKind.hipMemcpyHostToDevice
    [cuda_call(hip.hipMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)) for inp in inputs]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = hip.hipMemcpyKind.hipMemcpyDeviceToHost
    [cuda_call(hip.hipMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)) for out in outputs]
    # Synchronize the stream
    cuda_call(hip.hipStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)
    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)