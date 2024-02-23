import os
import onnxruntime as ort

from utils import get_model_io, numpy_to_pb


def generate_test_dataset(model_path, dataset_it, transform_fn, limit=None):
    folder_name_prefix = f"{os.path.dirname(model_path)}/test_data_set"
    inputs, outputs = get_model_io(model_path)
    input_pb_name = "input_{}.pb"
    output_pb_name = "output_{}.pb"

    for idx, data in enumerate(dataset_it):
        folder_name = f"{folder_name_prefix}_{idx}"
        print(f"Creating {folder_name}...")
        input_data_map = transform_fn(inputs, data)

        os.makedirs(folder_name, exist_ok=True)
        for input_idx, (input_name,
                        input_data) in enumerate(input_data_map.items()):
            numpy_to_pb(input_name, input_data,
                        f"{folder_name}/{input_pb_name.format(input_idx)}")

        sess = ort.InferenceSession(model_path)
        ort_result = sess.run(outputs, input_data_map)

        for output_idx, (output_name,
                         result_data) in enumerate(zip(outputs, ort_result)):
            numpy_to_pb(output_name, result_data,
                        f"{folder_name}/{output_pb_name.format(output_idx)}")

        if limit and limit < idx:
            break
