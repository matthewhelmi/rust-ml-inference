import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

# Helper class for loading and preprocessing images
import resnet50_data_reader

## --- FP16 quantisation i.e. half precision --- ##
fp32_model_path = "src/resnet50.onnx"
model = onnx.load(fp32_model_path)
model_fp16 = float16.convert_float_to_float16(model)
onnx.save(model_fp16, "src/resnet50_fp16.onnx")

##  --- Dynamic quantisation recommended for RNNs and Transformers --- ##
# Quantises weights statically but quantises activations
# dynamically based on a zero point and scale.
input_model = "src/resnet50_infer.onnx"
model_quant = 'src/resnet50_dynamic.onnx'
quantized_model = quantize_dynamic(
                        model_input=input_model,
                        model_output=model_quant,
                        weight_type=QuantType.QUInt8, # UInt8 quantisation - I ran into inference issues with Int8 
                        # Conv layers were not supported in Int8 precision
                        # https://github.com/microsoft/onnxruntime/issues/15888?
#                        op_types_to_quantize=["MatMul"],  # don't touch Conv, but this results in very minimal quantisation
#                        extra_options={"EnableSubgraph": True},
                        )

## --- Static quantisation recommended for CNNs --- ##
# Model weights and activations are statically quantised
# using a calibration dataset. The calibration dataset is used
# to compute the scale and zero-point i.e. range of the values
# to determine the quantisation.
# This calibration dataset becomes very important because it defines
# the expected range of values for every weight and activation. It must
# represent the expect real-life as closely as possible.
input_model = "src/resnet50_infer.onnx"
model_quant = 'src/resnet50_stat.quant.onnx'

# This is the input data reader used to compute the weights and activation
# ranges. It is very important the the preprocessing is identical to the
# preprocessing expected at inference.
dr = resnet50_data_reader.ResNet50DataReader(
        './test_images', "src/resnet50_infer.onnx"
    )
quantized_model = quantize_static(
                        input_model,
                        model_quant,
                        dr,
                        quant_format=QuantFormat.QDQ,
                        per_channel=False,
                        weight_type=QuantType.QInt8 # Int8 quantisation
                        )
