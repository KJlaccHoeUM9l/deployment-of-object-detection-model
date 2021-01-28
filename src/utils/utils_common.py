import torch

from models.ssd.model import SSD300


def generate_model(parameters):
    if not parameters.use_tensorrt:
        ssd_model = SSD300()
    else:
        import torch2trt
        ssd_model = torch2trt.TRTModule()
    ssd_model.load_state_dict(torch.load(parameters.weights_path))
    if parameters.use_fp16_mode:
        ssd_model.half()
    if parameters.use_eval_mode:
        ssd_model.eval()
    ssd_model.to(parameters.device)

    return ssd_model
