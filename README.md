# Simple ImageNet1K Classification with ResNet50

This project demonstrates a simple ImageNet1K classification pipeline using an off-the-shelf ResNet50 model.

---
## Download ImageNet labels
```bash
$ curl -o imagenet_classes.txt https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

## Run Python
```bash
$ uv sync
$ source .venv/bin/activate
(rust-resnet50-inference) $ python save_resnet50_onnx.py
(rust-resnet50-inference) $ python test_resnet50_onnx.py
```

## Run Rust
```bash
$ cargo build
$ cargo run --release
```
