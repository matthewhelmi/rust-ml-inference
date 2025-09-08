# Simple ImageNet1K Classification with ResNet50

This project demonstrates a simple ImageNet1K classification pipeline using an off-the-shelf ResNet50 model.

---

## Install Rust
You'll need Rust to run the last bit of this demo. I assume you have Python installed.
```bash
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Download ImageNet labels
You'll need the ImageNet labels for testing the models.
```bash
$ curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

## Run Python
The following lines will download a ResNet50 from torch models with pretrained weights (ImageNet1K classification). We will then proceed to convert the PyTorch model to ONNX and then quantise and test the quantised results.
```bash
$ uv sync
$ source .venv/bin/activate
(rust-resnet50-inference) $ python save_resnet50_onnx.py
```

We now have an ONNX model at src/resnet50.onnx. Now let's prepare it for quantisation. ONNX Runtime does this by first optimising the model.
Read more here: https://github.com/microsoft/onnxruntime-inference-examples/tree/main/quantization/image_classification/cpu.
```bash
(rust-resnet50-inference) $ python -m onnxruntime.quantization.preprocess --input ./src/resnet50.onnx --output ./src/resnet50_infer.onnx
```

The following lines will quantise the model into FP16, Dynamic UInt8, and Static Int8. Feel free to test any of them.
```bash
(rust-resnet50-inference) $ python quantise_resnet50_onnx.py
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision fp32
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision fp16
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision stat-quant
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision dyn-quant
```
There is a known issue with dynamic Int8 Conv layers: https://github.com/microsoft/onnxruntime/issues/15888?

Static quantisation requires a calibration dataset, which must replicate real-life as much as possible. Weights and activations are quantised based on ranges computed with the calibration dataset. A poor calibration set could introduce what I refer to as quantisation-bias.

Dynamic quantisation avoids this by only quantising the weights and leaving the activation quantisation to be done during inference. This way a lot of the quantisation is already achieved and at the cost of a slight overhead, we avoid the "quantisation-bias" and need for a calibration set.

Overall we are aiming to achieve a reduced model size to optimise storage and reduce inference time, whilst maintaining performance. Below we can see the reduced model sizes.
| FP32 [mb] | FP16 [mb] | Static Quant. (Int8) [mb]  | Dynamic Quant. (UInt8) [mb] |
| ------------- | ------------- | ------------- | ------------- |
| 97  | 49  | 25  | 25  |

Inference times become hardware dependent.

Example Output from test_resnet50_onnx.py:
```burn
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision fp32                                 

tabby 0.653938889503479
Egyptian cat 0.27055826783180237
tiger cat 0.06971198320388794
lynx 0.002341167302802205
tiger 0.0008536005043424666
ONNX Runtime Inference time = 29.53 ms

```
```burn
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision dyn-quant 

tabby 0.6432048678398132
Egyptian cat 0.27739453315734863
tiger cat 0.074055977165699
lynx 0.0020694576669484377
tiger 0.0006779290270060301
ONNX Runtime Inference time = 45.26 ms
```

## Run Rust
Only run this after doing the above. It requires src/resnet50.onnx is created by save_resnet50_onnx.py. Feel free to compare the execution time and output to the Python scripts above.
```bash
$ cargo build
$ cargo run --release
```

```burn
(rust-resnet50-inference) $ cargo run --release 

Example Output:
[Tract] inference: 298.522ms
Tract top-5:
 281: tabby                                                         0.6464
 285: Egyptian cat                                                  0.2831
 282: tiger cat                                                     0.0646
 287: lynx                                                          0.0025
 292: tiger                                                         0.0008

[ORT] inference: 145.104ms
ORT top-5:
 281: tabby                                                         0.6464
 285: Egyptian cat                                                  0.2831
 282: tiger cat                                                     0.0646
 287: lynx                                                          0.0025
 292: tiger                                                         0.0008

[Burn] inference: 308.928ms
Burn top-5:
 281: tabby                                                         0.6464
 285: Egyptian cat                                                  0.2831
 282: tiger cat                                                     0.0646
 287: lynx                                                          0.0025
 292: tiger                                                         0.0008
```
