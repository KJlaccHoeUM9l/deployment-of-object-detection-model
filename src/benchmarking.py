from argparse import ArgumentParser

from models.ssd.utils import dboxes300_coco, Encoder
from models.ssd.evaluate import evaluate
from models.ssd.data import get_val_dataset, get_val_dataloader, get_coco_ground_truth

from common_parameters import CommonParameters
from utils.utils_common import generate_model


def make_parser():
    parser = ArgumentParser(description="Train Single Shot MultiBox Detector"
                                        " on COCO")
    parser.add_argument('--data', '-d', type=str, default='/coco', required=False,
                        help='path to test and training data files')
    parser.add_argument('--eval-batch-size', '--ebs', type=int, default=32,
                        help='number of examples for each evaluation iteration')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to model checkpoint file')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--amp', action='store_true',
                        help='Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.')

    return parser


def bench(args, parameters):
    # Setup data, defaults
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes)
    cocoGt = get_coco_ground_truth(args)

    val_dataset = get_val_dataset(args)
    val_dataloader = get_val_dataloader(val_dataset, args)

    ssd300 = generate_model(parameters)

    inv_map = {v: k for k, v in val_dataset.label_map.items()}

    acc = evaluate(ssd300, val_dataloader, cocoGt, encoder, inv_map, args)
    print('Model precision {} mAP'.format(acc))


def main():
    parser = make_parser()
    args = parser.parse_args()
    common_parameters = CommonParameters()
    common_parameters.load_parameters('conf/config.yaml')

    args.data = common_parameters.coco_data_path
    args.checkpoint = common_parameters.weights_path
    args.amp = common_parameters.use_fp16_mode
    args.no_cuda = False
    args.eval_batch_size = common_parameters.eval_batch_size
    args.num_workers = common_parameters.num_workers

    bench(args, common_parameters)


if __name__ == "__main__":
    main()
