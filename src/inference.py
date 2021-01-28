import cv2
import yaml
import torch
import numpy as np

from models.ssd.processing_utils import padding_by_zeros, preprocess_input,\
                                        decode_results, pick_best, get_coco_object_dictionary

from utils.profiler import LogDuration
from utils.utils_common import generate_model
from utils.utils_visual import draw_bboxes, draw_fps, get_image_for_demo_app
from common_parameters import CommonParameters


def main():
    parameters = CommonParameters()
    parameters.load_parameters('config.yaml')
    fps_logger = {}
    print_log = False
    if parameters.demo_mode:
        with open('data/perspective_transforms.yaml', 'r') as stream:  # TODO: hardcode
            try:
                transforms = yaml.safe_load(stream)
                perspective_projection_matrix = np.array(transforms['perspective_transform'])
                perspective_image_width = int(transforms['perspective_image_width'])
                perspective_image_height = int(transforms['perspective_image_height'])
            except yaml.YAMLError as exc:
                print(exc)

    classes_to_labels = get_coco_object_dictionary('data/category_names.txt')    # TODO: hardcode

    # Load model
    ssd_model = generate_model(parameters)

    # Run demo
    cap = cv2.VideoCapture(parameters.video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if parameters.demo_mode:
            frame_for_demo = frame.copy()

        with LogDuration('preprocess_input', logger=fps_logger, print_log=print_log):
            model_input = padding_by_zeros(frame)
            padding_image_width, padding_image_height = model_input.shape[1], model_input.shape[0]
            model_input = preprocess_input(model_input, parameters)

        with torch.no_grad(), LogDuration('model_inference', logger=fps_logger, print_log=print_log):
            detections_batch = ssd_model(model_input)

        with LogDuration('decode_results', logger=fps_logger, print_log=print_log):
            results_per_input = decode_results(detections_batch)
            best_results_per_input = [pick_best(results, parameters.net_confidence) for results in results_per_input]

        ratio_w = padding_image_width / parameters.target_width
        ratio_h = padding_image_height / parameters.target_height
        draw_bboxes(frame,
                    best_results_per_input=best_results_per_input, image_idx=0,
                    target_width=parameters.target_width, target_height=parameters.target_height,
                    ratio_w=ratio_w, ratio_h=ratio_h,
                    classes_to_labels=classes_to_labels)

        draw_fps(frame, fps_logger)

        resize_coefficient = 0.5
        frame = cv2.resize(frame, (int(frame.shape[1] * resize_coefficient), int(frame.shape[0] * resize_coefficient)))
        cv2.imshow('Detections', frame)

        if parameters.demo_mode:
            frame_for_demo = get_image_for_demo_app(frame_for_demo, best_results_per_input,
                                                    perspective_projection_matrix, perspective_image_width, perspective_image_height,
                                                    parameters, ratio_w, ratio_h)
            cv2.imshow('Demo app', frame_for_demo)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
