import cv2
import numpy as np
from typing import List, Dict


def draw_bboxes(image: np.ndarray, best_results_per_input: List, image_idx: int,
                target_width: int, target_height: int, ratio_w: float, ratio_h: float,
                classes_to_labels: List) -> None:
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, w = [int(val * target_width * ratio_w) for val in [left, right - left]]
        y, h = [int(val * target_height * ratio_h) for val in [bot, top - bot]]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        cv2.putText(image,
                    '{} {:.0f}%'.format(classes_to_labels[classes[idx] - 1], confidences[idx] * 100),
                    (x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(255, 255, 255),
                    thickness=2)


def draw_fps(image: np.ndarray, fps_logger: Dict) -> None:
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2

    x = int(image.shape[1] * 0.7)
    y = int(image.shape[0] * 0.05)
    dy = cv2.getTextSize('text',
                         fontFace=font_face,
                         fontScale=font_scale,
                         thickness=thickness)[0][1]
    dy = int(dy * 1.5)

    for log_name, fps in fps_logger.items():
        text = '{}: {} fps'.format(log_name, int(fps))
        cv2.putText(image, text, (x, y),
                    fontFace=font_face,
                    fontScale=font_scale,
                    color=(0, 255, 0),
                    thickness=thickness)
        y += dy
