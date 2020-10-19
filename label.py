from datasets.change_detection import ChangeDetection
from models.model_zoo import get_model
from utils.options import Options
from utils.palette import color_map

import numpy as np
from PIL import Image
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

if __name__ == '__main__':
    args = Options().parse()

    dataset = ChangeDetection(root=args.data_root, mode='pseudo_labeling')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=4, drop_last=False)

    model1 = get_model('fcn', 'hrnet_w40', False, len(dataset.CLASSES) - 1, True)
    model1.load_state_dict(
        torch.load('outdir/models/change_detection/base_models_iter1/fcn_hrnet_w40_37.37.pth'), strict=True)

    model2 = get_model('fcn', 'hrnet_w48', False, len(dataset.CLASSES) - 1, True)
    model2.load_state_dict(
        torch.load('outdir/models/change_detection/base_models_iter1/fcn_hrnet_w48_37.46.pth'), strict=True)

    model3 = get_model('pspnet', 'hrnet_w40', False, len(dataset.CLASSES) - 1, True)
    model3.load_state_dict(
        torch.load('outdir/models/change_detection/base_models_iter1/pspnet_hrnet_w40_37.69.pth'), strict=True)

    model4 = get_model('pspnet', 'hrnet_w48', False, len(dataset.CLASSES) - 1, True)
    model4.load_state_dict(
        torch.load('outdir/models/change_detection/base_models_iter1/pspnet_hrnet_w48_37.63.pth'), strict=True)

    model5 = get_model('pspnet', 'hrnet_w64', False, len(dataset.CLASSES) - 1, True)
    model5.load_state_dict(
        torch.load('outdir/models/change_detection/base_models_iter1/pspnet_hrnet_w64_37.89.pth'), strict=True)

    models = [model1, model2, model3, model4, model5]
    for i in range(len(models)):
        models[i] = DataParallel(models[i]).cuda()
        models[i].eval()

    tbar = tqdm(dataloader)
    cmap = color_map()
    for img1, img2, mask1, mask2, id in tbar:
        img1, img2 = img1.cuda(), img2.cuda()

        pseudo_mask1_list, pseudo_mask2_list = [], []
        mask1 = mask1.numpy()
        mask2 = mask2.numpy()
        for model in models:
            with torch.no_grad():
                out1, out2, out_bin = model(img1, img2, True)

            out1 = torch.argmax(out1, dim=1).cpu().numpy() + 1
            out2 = torch.argmax(out2, dim=1).cpu().numpy() + 1

            pseudo_mask1 = np.zeros_like(out1)
            pseudo_mask2 = np.zeros_like(out2)

            pseudo_mask1[mask1 != 0] = mask1[mask1 != 0]
            pseudo_mask2[mask2 != 0] = mask2[mask2 != 0]
            pseudo_mask1[(mask1 == 0) & (out1 == out2)] = out1[(mask1 == 0) & (out1 == out2)]
            pseudo_mask2[(mask2 == 0) & (out1 == out2)] = out2[(mask2 == 0) & (out1 == out2)]

            pseudo_mask1_list.append(np.arange(7) == pseudo_mask1[..., None])
            pseudo_mask2_list.append(np.arange(7) == pseudo_mask2[..., None])

        pseudo_mask1 = np.stack(pseudo_mask1_list, axis=0)
        pseudo_mask1 = np.sum(pseudo_mask1, axis=0).astype(np.float)
        pseudo_mask2 = np.stack(pseudo_mask2_list, axis=0)
        pseudo_mask2 = np.sum(pseudo_mask2, axis=0).astype(np.float)

        out1 = np.argmax(pseudo_mask1, axis=3)
        out2 = np.argmax(pseudo_mask2, axis=3)

        for i in range(out1.shape[0]):
            mask = Image.fromarray(out1[i].astype(np.uint8), mode="P")
            mask.putpalette(cmap)
            mask.save("outdir/masks/train/im1/" + id[i])

            mask = Image.fromarray(out2[i].astype(np.uint8), mode="P")
            mask.putpalette(cmap)
            mask.save("outdir/masks/train/im2/" + id[i])
