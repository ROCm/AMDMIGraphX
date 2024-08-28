
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/model_separable_rnnt.py
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/rnn.py
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/preprocessing.py
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/helpers.py
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/metrics.py

mkdir parts
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/parts/features.py
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/parts/segment.py

mv features.py parts/
mv segment.py parts/


mkdir configs
wget https://raw.githubusercontent.com/mlcommons/inference/c7db1c3f9a0ae0623200e05580d77a8759644812/retired_benchmarks/speech_recognition/rnnt/pytorch/configs/rnnt.toml

mv rnnt.toml configs/