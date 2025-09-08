import onnxruntime
import os
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from torchvision import transforms as T
from PIL import Image

def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names
    unconcatenated_batch_data = []
    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name
        input_image = Image.open(image_filepath).convert("RGB")
        preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_data = preprocess(input_image).numpy()
        nchw_data = np.expand_dims(input_data, axis=0)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = np.concatenate(
        np.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nchw_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )

        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nchw_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nchw_data} for nchw_data in self.nchw_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
