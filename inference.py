import cv2
import torch
import torch2trt
import numpy as np

from typing import List, Dict

from profiler import LogDuration


def preprocess_input(image: np.ndarray, target_width: int, target_height: int) -> torch.Tensor:
    model_input = cv2.resize(image, (target_width, target_height))
    model_input = np.rollaxis(model_input, 2, 0)
    model_input = torch.from_numpy(np.array([model_input / 255.]))
    model_input = model_input.cuda().half()
    return model_input


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


def main():
    fps_logger = {}
    print_log = False

    target_width = 300
    target_height = 300

    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    classes_to_labels = utils.get_coco_object_dictionary()

    # video_path = '/home/agladyshev/Downloads/201202_01_Oxford Shoppers_4k_051.mp4'
    video_path = '/home/agladyshev/Downloads/ChampsElysees_150610_03_Videvo.mov'

    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
    ssd_model = torch2trt.TRTModule()
    ssd_model.load_state_dict(torch.load('/home/agladyshev/Documents/UNN/DL/huawei/weights/ssd_trt_300.pth'))

    ssd_model.to('cuda')
    ssd_model.eval().half()

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with LogDuration('preprocess_input', logger=fps_logger, print_log=print_log):
            model_input = preprocess_input(frame, target_width, target_height)

        with torch.no_grad(), LogDuration('model_inference', logger=fps_logger, print_log=print_log):
            detections_batch = ssd_model(model_input)

        with LogDuration('decode_results', logger=fps_logger, print_log=print_log):
            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

        draw_bboxes(frame,
                    best_results_per_input=best_results_per_input, image_idx=0,
                    target_width=target_width, target_height=target_height,
                    ratio_w=frame.shape[1] / target_width, ratio_h=frame.shape[0] / target_height,
                    classes_to_labels=classes_to_labels)

        draw_fps(frame, fps_logger)

        resize_coefficient = 0.5
        frame = cv2.resize(frame, (int(frame.shape[1] * resize_coefficient), int(frame.shape[0] * resize_coefficient)))
        cv2.imshow('Detections', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
