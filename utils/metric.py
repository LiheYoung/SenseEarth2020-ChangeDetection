import math
import numpy as np


def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        return score, miou, sek

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)
