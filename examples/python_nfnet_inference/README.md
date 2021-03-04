# NFNet Inference with MIGraphX

## NFNet
NFNet: Normalizer-Free Nets. An image recognition model that can be trained without batch normalization layers. They instead use gradient clipping algorithm to provide same affects of BatchNorm, such as training stability.

<ins>**Summary:**</ins>
- SOTA on ImageNet (86.5% top-1 w/o extra data)
- Up to 8.7x faster to train than EfficientNets to a given accuracy
- Normalizer-free (no BatchNorm!)

**Paper**: https://arxiv.org/pdf/2102.06171.pdf

**Colab noteook**: https://github.com/deepmind/deepmind-research/tree/master/nfnets

### Why not batch norm?

Batch normalization has three significant practical disadvantages,

1. It is a surprisingly expensive computational primitive, which incurs memory overhead, and significantly increases the time required to evaluate the gradient in some networks.
2. It introduces a discrepancy between the behavior of the model during training and at inference time, introducing hidden hyper-parameters that have to be tuned.
3. Last and most important point, batch normalization breaks the independence between training examples in the minibatch (batch size matters with batch norm, distributed training becomes extremely cumbersome)

Instead:

- They propose Adaptive Gradient Clipping (AGC), which clips gradients based on the unit-wise ratio of gradient norms to parameter norms, and they demonstrate that AGC allows them to train normalizer-free networks with larger batch sizes and stronger data augmentations.
- They design a family of Normalizer-Free ResNets, called NFNets, which set new state-of-the-art validation accuracies on ImageNet for a range of training latencies. Their NFNet-F1 model achieves similar accuracy to EfficientNet-B7 while being 8.7× faster to train, and their largest model sets a new overall state of the art without extra data of 86.5% top-1 accuracy.
- They show that NFNets achieve substantially higher validation accuracies than batch-normalized networks when fine-tuning on ImageNet after pre-training on a large private dataset of 300 million labelled images. Their best model achieves 89.2% top-1 after fine-tuning.

## Inference with MIGraphX

There is no ONNX model released for NFNet, however PyTorch model is available at:
https://github.com/rwightman/pytorch-image-models. There is a fork of this repository https://github.com/cagery/pytorch-image-models and it provides:
* NFNet PyTorch to ONNX conversion script: https://github.com/cagery/pytorch-image-models/blob/master/pt_to_onnx.py
* Modification to model where it uses torch.std_mean(), splitting the function to two seperate calls std() and mean(), as ONNX does not support std_mean() operator:
```
→ RuntimeError: Exporting the operator std_mean to ONNX opset version 11 is not supported. Please open a bug to request ONNX export support for the missing operator.
```

### Creating ONNX file
```
git clone https://github.com/cagery/pytorch-image-models.git
python3 pytorch-image-models/pt_to_onnx.py
```

### Inferencing a sample image
```<write here>```
