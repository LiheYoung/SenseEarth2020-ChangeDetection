from PIL import Image
import random


class RandomFlipOrRotate(object):
    def __call__(self, sample):
        img1, img2, mask1, mask2, mask_bin = \
            sample['img1'], sample['img2'], sample['mask1'], sample['mask2'], sample['mask_bin']

        rand = random.random()
        if rand < 1 / 6:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
            mask_bin = mask_bin.transpose(Image.FLIP_LEFT_RIGHT)

        elif rand < 2 / 6:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            mask1 = mask1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
            mask2 = mask2.transpose(Image.FLIP_TOP_BOTTOM)
            mask_bin = mask_bin.transpose(Image.FLIP_TOP_BOTTOM)

        elif rand < 3 / 6:
            img1 = img1.transpose(Image.ROTATE_90)
            mask1 = mask1.transpose(Image.ROTATE_90)
            img2 = img2.transpose(Image.ROTATE_90)
            mask2 = mask2.transpose(Image.ROTATE_90)
            mask_bin = mask_bin.transpose(Image.ROTATE_90)

        elif rand < 4 / 6:
            img1 = img1.transpose(Image.ROTATE_180)
            mask1 = mask1.transpose(Image.ROTATE_180)
            img2 = img2.transpose(Image.ROTATE_180)
            mask2 = mask2.transpose(Image.ROTATE_180)
            mask_bin = mask_bin.transpose(Image.ROTATE_180)

        elif rand < 5 / 6:
            img1 = img1.transpose(Image.ROTATE_270)
            mask1 = mask1.transpose(Image.ROTATE_270)
            img2 = img2.transpose(Image.ROTATE_270)
            mask2 = mask2.transpose(Image.ROTATE_270)
            mask_bin = mask_bin.transpose(Image.ROTATE_270)

        return {'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2, 'mask_bin': mask_bin}
