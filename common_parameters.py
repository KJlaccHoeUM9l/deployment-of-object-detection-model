import torch


class CommonParameters:
    weights_path = None
    video_path = None

    target_width = None
    target_height = None
    norm_mean = None
    norm_std = None

    use_fp16_mode = None
    use_tensorrt = None
    use_eval_mode = None
    device = None

    net_confidence = None

    def load_parameters(self):
        self.weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp32_trt.pth'  # TODO: temporal path
        # self.weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp32.pth'  # TODO: temporal path
        self.video_path = '/home/agladyshev/Downloads/ChampsElysees_150610_03_Videvo.mov'   # TODO: temporal path

        self.target_width = 300
        self.target_height = 300
        self.norm_mean = [123.675, 116.28, 103.53]
        self.norm_std = [58.395, 57.12, 57.375]

        self.use_fp16_mode = False
        self.use_tensorrt = True
        self.use_eval_mode = True
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.net_confidence = 0.40
