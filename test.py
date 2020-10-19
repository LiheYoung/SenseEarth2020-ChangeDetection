from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map

import numpy as np
import os
from PIL import Image
import shutil
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__ == "__main__":
    args = Options().parse()

    torch.backends.cudnn.benchmark = True

    testset = ChangeDetection(root=args.data_root, mode="test")
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model1 = get_model('pspnet', 'hrnet_w40', False, len(testset.CLASSES) - 1, True)
    model1.load_state_dict(torch.load('outdir/models/change_detection/pspnet_hrnet_w40_39.37.pth'), strict=True)
    model2 = get_model('pspnet', 'hrnet_w18', False, len(testset.CLASSES) - 1, True)
    model2.load_state_dict(torch.load('outdir/models/change_detection/pspnet_hrnet_w18_38.74.pth'), strict=True)

    models = [model1, model2]
    for i in range(len(models)):
        models[i] = models[i].cuda()
        models[i].eval()

    cmap = color_map()

    tbar = tqdm(testloader)

    with torch.no_grad():
        for img1, img2, id in tbar:
            img1, img2 = img1.cuda(non_blocking=True), img2.cuda(non_blocking=True)

            out1_list, out2_list, out_bin_list = [], [], []
            for model in models:
                out1, out2, out_bin = model(img1, img2, True)
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
                mask.save("outdir/masks/test/im1/" + id[i])

                mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
                mask.putpalette(cmap)
                mask.save("outdir/masks/test/im2/" + id[i])
