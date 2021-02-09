import torch
import argparse
from torch2trt import torch2trt

from models.ssd.model import SSD300
from utils.utils_conversion import ImageFolderCalibDataset


def arguments_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_weights_path',
                        type=str,
                        help='Path to source weights',
                        required=True)
    parser.add_argument('--save_weights_path',
                        type=str,
                        help='Name for save weights',
                        required=True)
    parser.add_argument('--calib_dataset_path',
                        type=str,
                        help='Path to calibration dataset for int8 mode',
                        required=False)
    parser.add_argument('--use_fp16_mode',
                        type=bool,
                        default=False,
                        help='Use FP16 mode',
                        required=False)
    parser.add_argument('--use_int8_mode',
                        type=bool,
                        default=False,
                        help='Use INT8 mode',
                        required=False)
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=64,
                        help='Max size of batch',
                        required=False)
    parser.add_argument('--input_channels',
                        type=int,
                        default=3,
                        help='Number of channels for input image',
                        required=False)
    parser.add_argument('--input_height',
                        type=int,
                        default=300,
                        help='Input image height',
                        required=False)
    parser.add_argument('--input_weight',
                        type=int,
                        default=300,
                        help='Input image width',
                        required=False)
    return parser.parse_args()


def main():
    args = arguments_parser()
    load_weights_path = args.load_weights_path
    save_weights_path = args.save_weights_path
    calib_dataset_path = args.calib_dataset_path
    use_fp16_mode = args.use_fp16_mode
    use_int8_mode = args.use_int8_mode
    max_batch_size = args.use_int8_mode
    input_channels = args.input_channels
    input_height = args.input_height
    input_weight = args.input_weight

    model = SSD300()
    model.load_state_dict(torch.load(load_weights_path))
    model = model.cuda().eval()
    if use_fp16_mode:
        model = model.half()

    x = torch.ones((max_batch_size, input_channels, input_height, input_weight)).float().cuda()
    if use_fp16_mode:
        x = x.half()

    dataset = None if not use_int8_mode else ImageFolderCalibDataset(calib_dataset_path)
    model_trt = torch2trt(model, [x],
                          max_batch_size=max_batch_size,
                          fp16_mode=use_fp16_mode,
                          int8_mode=use_int8_mode,
                          int8_calib_dataset=dataset,
                          int8_calib_batch_size=16
                          )

    torch.save(model_trt.state_dict(), save_weights_path)
    print('Conversion done')


if __name__ == '__main__':
    main()
