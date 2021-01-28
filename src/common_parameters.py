import yaml
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

    def load_parameters(self, config_path):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)

                self.demo_mode = config['demo_mode']
                self.weights_path = config['weights_path']
                self.video_path = config['video_path']
                self.target_width = config['target_width']
                self.target_height = config['target_height']
                self.norm_mean = config['norm_mean']
                self.norm_std = config['norm_std']
                self.use_fp16_mode = config['use_fp16_mode']
                self.use_tensorrt = config['use_tensorrt']
                self.use_eval_mode = config['use_eval_mode']
                self.net_confidence = config['net_confidence']
                self.coco_data_path = config['coco_data_path']
                self.eval_batch_size = config['eval_batch_size']
                self.num_workers = config['num_workers']
            except yaml.YAMLError as exc:
                print(exc)
