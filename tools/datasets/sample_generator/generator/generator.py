import os
import onnxruntime as ort

from ..utils import get_model_io, numpy_to_pb


def generate_test_dataset(model,
                          dataset,
                          output_path=None,
                          sample_limit=None,
                          decode_limit=None):
    if not output_path:
        output_path = f"{dataset.name}/{model.name}"
    folder_name_prefix = f"{output_path}/test_data_set"
    input_pb_name = "input_{}.pb"
    output_pb_name = "output_{}.pb"

    # Model
    model_path = model.download(output_path)
    inputs, outputs = get_model_io(model_path)

    sess = ort.InferenceSession(model_path)
    print(f"Creating {folder_name_prefix}s...")
    test_idx = 0
    for idx, data in enumerate(dataset):
        input_data_map = dataset.transform(inputs, data, model.preprocess)
        is_eos, decode_idx = False, 0
        while not is_eos and not (decode_limit and decode_limit < decode_idx):
            folder_name = f"{folder_name_prefix}_{test_idx}"
            os.makedirs(folder_name, exist_ok=True)
            for input_idx, (input_name,
                            input_data) in enumerate(input_data_map.items()):
                numpy_to_pb(
                    input_name, input_data,
                    f"{folder_name}/{input_pb_name.format(input_idx)}")

            ort_result = sess.run(outputs, input_data_map)
            output_data_map = {
                output_name: result_data
                for (output_name, result_data) in zip(outputs, ort_result)
            }
            for output_idx, (output_name, result_data) in enumerate(
                    output_data_map.items()):
                numpy_to_pb(
                    output_name, result_data,
                    f"{folder_name}/{output_pb_name.format(output_idx)}")

            is_eos = not model.is_decoder() or model.decode_step(
                input_data_map, output_data_map)
            test_idx += 1
            decode_idx += 1

        if sample_limit and sample_limit < idx:
            break
