from utils import download

# TODO
# Class
#   download
#   get IO?

RESNET_MODEL_NAME = "resnet50_v1.onnx"
RESNET50_MODEL_URL = f"https://zenodo.org/record/2592612/files/{RESNET_MODEL_NAME}"


def download_resnet_model(output_filepath):
    print(f"Download model from {RESNET50_MODEL_URL}")
    download(RESNET50_MODEL_URL, output_filepath)
