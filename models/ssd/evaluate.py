# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import time
import numpy as np

from pycocotools.cocoeval import COCOeval
from tqdm import tqdm


def evaluate(model, coco, cocoGt, encoder, inv_map, args):
    model.eval()
    if not args.no_cuda:
        model.cuda()
    ret = []
    model_eval_times = []
    start = time.time()

    for nbatch, (img, img_id, img_size, _, _) in enumerate(tqdm(coco)):
        with torch.no_grad():
            inp = img.cuda()
            if args.amp:
                inp = inp.half()

            # Get predictions
            model_eval_start = time.time()
            ploc, plabel = model(inp)
            model_eval_times.append((time.time() - model_eval_start) / args.eval_batch_size)

            ploc, plabel = ploc.float(), plabel.float()

            # Handle the batch of predictions produced
            # This is slow, but consistent with old implementation.
            for idx in range(ploc.shape[0]):
                # ease-of-use for specific predictions
                ploc_i = ploc[idx, :, :].unsqueeze(0)
                plabel_i = plabel[idx, :, :].unsqueeze(0)

                try:
                    result = encoder.decode_batch(ploc_i, plabel_i, 0.50, 200)[0]
                except:
                    # raise
                    print("")
                    print("No object detected in idx: {}".format(idx))
                    continue

                htot, wtot = img_size[0][idx].item(), img_size[1][idx].item()
                loc, label, prob = [r.cpu().numpy() for r in result]
                for loc_, label_, prob_ in zip(loc, label, prob):
                    ret.append([img_id[idx], loc_[0] * wtot, \
                                loc_[1] * htot,
                                (loc_[2] - loc_[0]) * wtot,
                                (loc_[3] - loc_[1]) * htot,
                                prob_,
                                inv_map[label_]])

    # Now we have all predictions from this rank, gather them all together
    # if necessary
    ret = np.array(ret).astype(np.float32)

    final_results = ret

    # if args.local_rank == 0:
    print("")
    print("Predicting Ended, total time: {:.2f} s".format(time.time() - start))

    cocoDt = cocoGt.loadRes(final_results)

    E = COCOeval(cocoGt, cocoDt, iouType='bbox')
    E.evaluate()
    E.accumulate()
    E.summarize()
    print("Current AP: {:.5f}".format(E.stats[0]))

    mean_performance_sec = np.mean(model_eval_times)
    print('*************************************')
    print('Model performance per sample:\n{:.5f} sec, {:.5f} fps'.format(mean_performance_sec,
                                                                         1. / mean_performance_sec))
    print('*************************************')

    # put your model in training mode back on
    model.bench()

    return E.stats[0]  # Average Precision  (AP) @[ IoU=050:0.95 | area=   all | maxDets=100 ]
