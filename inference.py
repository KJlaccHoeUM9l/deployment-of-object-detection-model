import cv2
import torch
import torch2trt
import numpy as np

from profiler import LogDuration
from common_parameters import CommonParameters
from visual_utils import draw_bboxes, draw_fps


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
    if parameters.use_fp16_mode:
        model_input = model_input.half()
    model_input = model_input.to(parameters.device)
    return model_input


def main():
    parameters = CommonParameters()
    parameters.load_parameters()
    fps_logger = {}
    print_log = False

    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    classes_to_labels = utils.get_coco_object_dictionary()

    # Load model
    if not parameters.use_tensorrt:
        ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
    else:
        ssd_model = torch2trt.TRTModule()
        ssd_model.load_state_dict(torch.load(parameters.weights_path))
    if parameters.use_fp16_mode:
        ssd_model.half()
    if parameters.use_eval_mode:
        ssd_model.eval()
    ssd_model.to(parameters.device)

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
            results_per_input = utils.decode_results(detections_batch)
            best_results_per_input = [utils.pick_best(results, parameters.net_confidence) for results in
                                      results_per_input]

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
