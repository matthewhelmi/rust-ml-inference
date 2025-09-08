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
(rust-resnet50-inference) $ python quantise_resnet50_onnx.py
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision fp32
(rust-resnet50-inference) $ python test_resnet50_onnx.py --precision fp16
```

## Run Rust
Only run this after doing the above. It requires src/resnet50.onnx is created by save_resnet50_onnx.py. Feel free to compare the execution time and output to the Python scripts above.
```bash
$ cargo build
$ cargo run --release
```
