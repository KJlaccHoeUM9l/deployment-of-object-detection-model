import torch
from torch2trt import torch2trt

from models.ssd.model import SSD300


def main():
    use_fp16_mode = False
    load_weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp32.pth'
    save_weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp32_trt.pth'

    model = SSD300()
    model.load_state_dict(torch.load(load_weights_path))
    model = model.cuda().eval()
    if use_fp16_mode:
        model = model.half()

    x = torch.ones((1, 3, 300, 300)).float().cuda()
    if use_fp16_mode:
        x = x.half()

    model_trt = torch2trt(model, [x], fp16_mode=use_fp16_mode)

    torch.save(model_trt.state_dict(), save_weights_path)
    print('Conversion done')


if __name__ == '__main__':
    main()
