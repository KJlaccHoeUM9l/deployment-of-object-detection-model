import torch
from torch2trt import torch2trt

from models.ssd.model import SSD300
from utils_conversion import ImageFolderCalibDataset


def main():
    use_fp16_mode = False
    use_int8_mode = True
    max_batch_size = 32
    # TODO: temporal paths
    load_weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp32.pth'
    save_weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_int8_calib_trt_batch_{}.pth'.format(max_batch_size)
    calib_dataset_path = '/home/agladyshev/Desktop/coco_calib_data'

    model = SSD300()
    model.load_state_dict(torch.load(load_weights_path))
    model = model.cuda().eval()
    if use_fp16_mode:
        model = model.half()

    x = torch.ones((max_batch_size, 3, 300, 300)).float().cuda()
    if use_fp16_mode:
        x = x.half()

    dataset = None if not use_int8_mode else ImageFolderCalibDataset(calib_dataset_path)
    model_trt = torch2trt(model, [x],
                          max_batch_size=max_batch_size,
                          fp16_mode=use_fp16_mode,
                          int8_mode=use_int8_mode,
                          int8_calib_dataset=dataset,
                          int8_calib_batch_size=32  # TODO: hardcode calib batch size
                          )

    torch.save(model_trt.state_dict(), save_weights_path)
    print('Conversion done')


if __name__ == '__main__':
    main()
