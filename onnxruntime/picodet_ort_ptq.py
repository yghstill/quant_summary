import os
import numpy as np
import cv2
import argparse
from paddle.vision.transforms import Compose, Resize, CenterCrop, Normalize
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, quantize_static, QuantType, CalibrationMethod
from onnxruntime import InferenceSession, get_available_providers


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_path',
        type=str,
        default='picodet_s_416_npu.onnx',
        help="path of fp32 onnx model.")
    parser.add_argument(
        '--calibration_image_dir',
        type=str,
        default='/paddle/dataset/coco/val2017',
        help="Calibration image dir.")
    parser.add_argument(
        '--save_path',
        type=str,
        default='picodet_s_416_npu_ort_ptq.onnx',
        help="path of save quant onnx model.")

    return parser


class ImageDataset(object):
    def __init__(self, image_dir=None, img_size=[416, 416]):
        self.image_dir = image_dir
        self.img_size = img_size
        self.image_list = os.listdir(self.image_dir)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.image_list[idx]))
        img, _ = self.image_preprocess(img, self.img_size)
        return np.expand_dims(img, axis=0)

    def __len__(self):
        return len(self.image_list)

    def image_preprocess(self, img, target_shape):
        # Resize image
        resize_h, resize_w = target_shape
        origin_shape = img.shape[:2]
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
        img = cv2.resize(
            img,
            None,
            None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img / 255, [2, 0, 1])
        scale_factor = np.array([im_scale_y, im_scale_x])
        return img.astype(np.float32), scale_factor


class DataReader(CalibrationDataReader):
    def __init__(self, datas, batch_size=1, batch_num=64):
        self.batch_num = batch_num
        self.datas = self._batch_reader(datas, batch_size)
        self.batch_idx = 0

    def get_next(self):
        self.batch_idx += 1
        if self.batch_idx >= self.batch_num:
            return None
        else:
            return next(self.datas, None)

    def _batch_reader(self, dataset, batch_size):
        _datas = []
        length = len(dataset)
        for i, data in enumerate(dataset):
            if batch_size == 1:
                yield {'image': data}
            elif (i + 1) % batch_size == 0:
                _datas.append(data)
                yield {'image': np.concatenate(_datas, 0)}
                _datas = []
            elif i < length - 1:
                _datas.append(data)
            else:
                _datas.append(data)
                yield {'image': np.concatenate(_datas, 0)}


def quant(model_fp32, model_quant_static):
    dataset = ImageDataset(
        image_dir=FLAGS.calibration_image_dir, img_size=[416, 416])

    data_reader = DataReader(dataset, 1)

    quantize_static(
        model_input=model_fp32,
        model_output=model_quant_static,
        # op_types_to_quantize=['Conv'],
        calibration_data_reader=data_reader,
        quant_format=QuantFormat.QDQ,  # QDQ / QOperator
        activation_type=QuantType.QInt8,  # Int8 / UInt8
        weight_type=QuantType.QInt8,  # Int8 / UInt8
        calibrate_method=CalibrationMethod.
        MinMax,  # MinMax / Entropy / Percentile
        optimize_model=True)


if __name__ == '__main__':
    parser = argsparser()
    FLAGS = parser.parse_args()
    quant(FLAGS.model_path, FLAGS.save_path)
