import cv2
import numpy as np
from typing import List, Dict, Tuple


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


def perspective_transform_coordinates(x: float, y: float, transform_matrix: np.ndarray) -> Tuple[int, int]:
    denominator = transform_matrix[2, 0] * x + transform_matrix[2, 1] * y + transform_matrix[2, 2]
    return (int((transform_matrix[0, 0] * x + transform_matrix[0, 1] * y + transform_matrix[0, 2]) / denominator),
            int((transform_matrix[1, 0] * x + transform_matrix[1, 1] * y + transform_matrix[1, 2]) / denominator))


def get_image_for_demo_app(frame_for_demo, best_results_per_input,
                           perspective_projection_matrix, perspective_image_width, perspective_image_height,
                           parameters, ratio_w, ratio_h):
    k = 0.8
    up = int(k * perspective_image_height)
    down = int(perspective_image_height)

    frame_for_demo = cv2.warpPerspective(frame_for_demo, perspective_projection_matrix,
                                         (perspective_image_width, perspective_image_height))
    zone_image = frame_for_demo.copy()
    cv2.rectangle(zone_image, (0, up), (perspective_image_width, down), (56, 253, 254), thickness=-1)
    frame_for_demo = cv2.addWeighted(frame_for_demo, 0.8, zone_image, 0.2, 0)

    bboxes, _, _ = best_results_per_input[0]
    for left, _, right, top in bboxes:
        x, y = (left + right) / 2. * parameters.target_width * ratio_w, top * parameters.target_height * ratio_h
        x, y = perspective_transform_coordinates(x, y, perspective_projection_matrix)
        color = (0, 255, 0) if y < up else (0, 0, 255)
        cv2.circle(frame_for_demo, (x, y), 5, color, -1)

    return frame_for_demo
