import cv2
import torch
import numpy as np

from models.ssd.utils import dboxes300_coco, Encoder
from common_parameters import CommonParameters


def padding_by_zeros(image: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    max_size = max(w, h)
    image_pad = np.zeros((max_size, max_size, c), dtype=np.uint8)
    image_pad[:h, :w, :] = image
    return image_pad


def preprocess_input(image: np.ndarray, parameters: CommonParameters) -> torch.Tensor:
    model_input = cv2.resize(image, (parameters.target_width, parameters.target_height))
    model_input = (model_input - parameters.norm_mean) / parameters.norm_std
    model_input = np.rollaxis(model_input, 2, 0)

    model_input = torch.from_numpy(np.array([model_input]))
    model_input = model_input.float()
    if parameters.use_fp16_mode:
        model_input = model_input.half()
    model_input = model_input.to(parameters.device)
    return model_input


def decode_results(predictions):
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    ploc, plabel = [val.float() for val in predictions]
    results = encoder.decode_batch(ploc, plabel, criteria=0.5, max_output=20)
    return [[pred.detach().cpu().numpy() for pred in detections] for detections in results]


def pick_best(detections, threshold=0.3):
    bboxes, classes, confidences = detections
    best = np.argwhere(confidences > threshold)[:, 0]
    return [pred[best] for pred in detections]


def get_coco_object_dictionary():
    class_names = open("category_names.txt").readlines()    # TODO: temporal path
    class_names = [c.strip() for c in class_names]
    return class_names
