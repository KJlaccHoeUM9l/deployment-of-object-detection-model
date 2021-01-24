import cv2
import torch

from models.ssd.processing_utils import padding_by_zeros, preprocess_input,\
                                        decode_results, pick_best, get_coco_object_dictionary

from profiler import LogDuration
from common_parameters import CommonParameters
from utils_common import generate_model
from utils_visual import draw_bboxes, draw_fps


def main():
    parameters = CommonParameters()
    parameters.load_parameters()
    fps_logger = {}
    print_log = False

    classes_to_labels = get_coco_object_dictionary()

    # Load model
    ssd_model = generate_model(parameters)

    # Run demo
    cap = cv2.VideoCapture(parameters.video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        with LogDuration('preprocess_input', logger=fps_logger, print_log=print_log):
            model_input = padding_by_zeros(frame)
            padding_image_width, padding_image_height = model_input.shape[1], model_input.shape[0]
            model_input = preprocess_input(model_input, parameters)

        with torch.no_grad(), LogDuration('model_inference', logger=fps_logger, print_log=print_log):
            detections_batch = ssd_model(model_input)

        with LogDuration('decode_results', logger=fps_logger, print_log=print_log):
            results_per_input = decode_results(detections_batch)
            best_results_per_input = [pick_best(results, parameters.net_confidence) for results in results_per_input]

        draw_bboxes(frame,
                    best_results_per_input=best_results_per_input, image_idx=0,
                    target_width=parameters.target_width, target_height=parameters.target_height,
                    ratio_w=padding_image_width / parameters.target_width, ratio_h=padding_image_height / parameters.target_height,
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
