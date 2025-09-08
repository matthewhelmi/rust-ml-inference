import argparse
import time
import onnxruntime
import numpy as np
from torchvision import transforms as T
from PIL import Image
import os

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e_x = np.exp(x, dtype=np.float64)
    return (e_x / e_x.sum()).astype(np.float32)

latency = []

def run_sample(session, image_file, categories, inputs):
    # Get actual input name from the ONNX model
    input_name = session.get_inputs()[0].name

    # Warm up
    input_arr = inputs.cpu().detach().numpy()
    ort_outputs = session.run([], {input_name: input_arr})[0]

    start = time.time()
    ort_outputs = session.run([], {input_name: input_arr})[0]
    latency.append(time.time() - start)

    output = ort_outputs.flatten()
    output = softmax(output)  # optional but usually nice for readability
    top5_catid = np.argsort(-output)[:5]
    for catid in top5_catid:
        print(categories[catid], float(output[catid]))
    return ort_outputs

def main():
    parser = argparse.ArgumentParser(description="Run ResNet50 ONNX with selectable precision.")
    parser.add_argument("--precision", choices=["fp32", "fp16", "dyn-quant", "stat-quant"], default="fp32",
                        help="Inference precision & model variant to use.")
    parser.add_argument("--image", default="src/cat.jpg", help="Path to input image.")
    parser.add_argument("--classes", default="imagenet_classes.txt", help="Path to ImageNet classes file.")
    parser.add_argument("--providers", nargs="*", default=["CPUExecutionProvider"],
                        help="ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider")
    parser.add_argument("--model-fp32", default="src/resnet50.onnx",
                        help="Path to FP32 model.")
    parser.add_argument("--model-fp16", default="src/resnet50_fp16.onnx",
                        help="Path to FP16 model.")
    parser.add_argument("--model-dyn", default="src/resnet50_dynamic.onnx",
                        help="Path to Dynamic Quantised model.")
    parser.add_argument("--model-stat", default="src/resnet50_stat.quant.onnx",
                        help="Path to Static Quantised model.")
    args = parser.parse_args()

    # Pick model path based on precision
    if args.precision == "fp16":
        model_path = args.model_fp16
    elif args.precision == "fp32":
        model_path = args.model_fp32
    elif args.precision == "dyn-quant":
        model_path = args.model_dyn
    elif args.precision == "stat-quant":
        model_path = args.model_stat
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    # Create session
    session = onnxruntime.InferenceSession(model_path, providers=args.providers)

    # Read categories
    with open(args.classes, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Load & preprocess image
    input_image = Image.open(args.image).convert("RGB")
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    # Cast to chosen precision
    if args.precision == "fp16":
        input_batch = input_tensor.unsqueeze(0).half()
    else:
        input_batch = input_tensor.unsqueeze(0).float()

    # Run inference
    ort_output = run_sample(session, args.image, categories, input_batch)
    print("ONNX Runtime Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.2f')))

if __name__ == "__main__":
    main()
