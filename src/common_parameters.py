import torch


class CommonParameters:
    demo_mode = None

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

    # Benchmarking
    coco_data_path = None
    eval_batch_size = None
    num_workers = None

    def load_parameters(self):
        self.demo_mode = True

        self.weights_path = '/home/agladyshev/Documents/UNN/DL/ssd_weights/ssd_fp16.pth'  # TODO: temporal path
        self.video_path = '/home/agladyshev/Downloads/ChampsElysees_150610_03_Videvo.mov'   # TODO: temporal path

        self.target_width = 300
        self.target_height = 300
        self.norm_mean = [123.675, 116.28, 103.53]
        self.norm_std = [58.395, 57.12, 57.375]

        self.use_fp16_mode = True
        self.use_tensorrt = False
        self.use_eval_mode = True
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.net_confidence = 0.40

        self.coco_data_path = '/home/agladyshev/Documents/UNN/DL/Datasets/COCO'     # TODO: temporal path
        self.eval_batch_size = 64
        self.num_workers = 4
