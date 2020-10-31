from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map

import numpy as np
import os
from PIL import Image
import shutil
import time
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__ == "__main__":
    START_TIME = time.time()
    LIMIT_TIME = 400 - 20
    PAST_TIME = 0
    NO_TTA_TIME = 0
    TTA_TIME = 0

    args = Options().parse()

    torch.backends.cudnn.benchmark = True
    
    print(torch.cuda.is_available())
    testset = ChangeDetection(root='/val', mode="test")
    testloader = DataLoader(testset, batch_size=8, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model1 = get_model('pspnet', 'hrnet_w40', False, len(testset.CLASSES) - 1, True)
    model1.load_state_dict(torch.load('./outdir/models/change_detection/pspnet_hrnet_w40_39.37.pth'), strict=True)
    model2 = get_model('pspnet', 'hrnet_w18', False, len(testset.CLASSES) - 1, True)
    model2.load_state_dict(torch.load('./outdir/models/change_detection/pspnet_hrnet_w18_38.74.pth'), strict=True)

    models = [model1, model2]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    cmap = color_map()

    tbar = tqdm(testloader)
    TOTAL_ITER = len(testloader)
    CHECK_ITER = TOTAL_ITER // 5
    NO_TTA_ITER = TOTAL_ITER

    if not os.path.exists('/output/im1'):
        os.mkdir('/output/im1')
    if not os.path.exists('/output/im2'):
        os.mkdir('/output/im2')

    with torch.no_grad():
        for k, (img1, img2, id) in enumerate(tbar):
            if k == CHECK_ITER - 1:
                iter_start_time = time.time()
            if k == CHECK_ITER + 1:
                PAST_TIME = time.time() - START_TIME
                NO_TTA_ITER = (LIMIT_TIME - PAST_TIME - NO_TTA_TIME * TOTAL_ITER +
                               (CHECK_ITER + 1) * TTA_TIME) / (TTA_TIME - NO_TTA_TIME)

            img1, img2 = img1.cuda(non_blocking=True), img2.cuda(non_blocking=True)

            out1_list, out2_list, out_bin_list = [], [], []

            if k < CHECK_ITER:
                for model in models:
                    out1, out2, out_bin = model(img1, img2, True)
                    out1 = torch.softmax(out1, dim=1)
                    out2 = torch.softmax(out2, dim=1)

                    out1_list.append(out1)
                    out2_list.append(out2)
                    out_bin_list.append(out_bin)

            elif k == CHECK_ITER:
                start = time.time()
                for model in models:
                    out1, out2, out_bin = model(img1, img2, False)
                    out1 = torch.softmax(out1, dim=1)
                    out2 = torch.softmax(out2, dim=1)

                    out1_list.append(out1)
                    out2_list.append(out2)
                    out_bin_list.append(out_bin)
                end = time.time()
                NO_TTA_TIME = end - start

                start = time.time()
                for model in models:
                    out1, out2, out_bin = model(img1, img2, True)
                    out1 = torch.softmax(out1, dim=1)
                    out2 = torch.softmax(out2, dim=1)

                    out1_list.append(out1)
                    out2_list.append(out2)
                    out_bin_list.append(out_bin)
                end = time.time()
                TTA_TIME = end - start

                NO_TTA_TIME = PER_ITER_TIME - TTA_TIME + NO_TTA_TIME
                TTA_TIME = PER_ITER_TIME

            else:
                if k < NO_TTA_ITER:
                    use_tta = True
                else:
                    use_tta = False
                for model in models:
                    out1, out2, out_bin = model(img1, img2, use_tta)
                    out1 = torch.softmax(out1, dim=1)
                    out2 = torch.softmax(out2, dim=1)

                    out1_list.append(out1)
                    out2_list.append(out2)
                    out_bin_list.append(out_bin)

            out1 = torch.stack(out1_list, dim=0)
            out1 = torch.sum(out1, dim=0) / len(models)
            out2 = torch.stack(out2_list, dim=0)
            out2 = torch.sum(out2, dim=0) / len(models)
            out_bin = torch.stack(out_bin_list, dim=0)
            out_bin = torch.sum(out_bin, dim=0) / len(models)

            out1 = torch.argmax(out1, dim=1) + 1
            out2 = torch.argmax(out2, dim=1) + 1
            out_bin = (out_bin > 0.5)
            out1[out_bin == 1] = 0
            out2[out_bin == 1] = 0
            out1 = out1.cpu().numpy()
            out2 = out2.cpu().numpy()

            for i in range(out1.shape[0]):
                mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save("/output/im1/" + id[i])

                mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save("/output/im2/" + id[i])

            if k == CHECK_ITER - 1:
                iter_end_time = time.time()
                PER_ITER_TIME = iter_end_time - iter_start_time

    END_TIME = time.time()
    print("Inference Time: %.1fs" % (END_TIME - START_TIME))
