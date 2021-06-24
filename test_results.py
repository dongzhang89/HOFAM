import numpy as np
import cv2 as cv
import os
import scipy.signal as signal
from glob import glob

def extract_image(filename):
    image = cv.imread(filename)
    return image

model_name = 'model_tmp_0721_3_stam'

cwd = '/data/tmpu1/WZQ/Test/test/test'+model_name
# cwd = '/data/tmpu1/WZQ/Test/test/LIMU/CameraParameter'

cwd = '/data/tmpu1/WZQ/Test/test/LASIESTA/O_MC'

image_files = glob('{}/*.png'.format(cwd))
def compute_confusion(gt, res):
    P = sum(sum(gt == 255))
    N = sum(sum(gt == 0))
    xx = 15
    TP = sum(sum(np.multiply((gt == 255), (res >= xx))))
    TN = sum(sum(np.multiply((gt == 0), (res < xx))))
    FP = sum(sum(np.multiply((gt == 0), (res >= xx))))
    FN = sum(sum(np.multiply((gt == 255), (res < xx))))
    if TP == 0 and FN == 0 and FP == 0:
        confusion = [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        dice = 2 * TP / (2 * TP + FN + FP)
        confusion = [P, N, TP, TN, FN, FP, dice, 1]

    if P == 1835008:
        confusion = [0, 0, 0, 0, 0, 0, 0, 0]
    return confusion

confusion = np.zeros(8)
#idx = 0
#imtmp = image_files[3300]
for idx in range(0, len(image_files)):
    try:
        image = extract_image(image_files[idx])
        img_gt = image[:, 256:512, 0]
        img_res = image[:, 256*3:, 0]
        #img_res = signal.medfilt2d(img_res, kernel_size=3)

        tmp = compute_confusion(img_gt, img_res)
    except:
        print(image_files[idx])
    confusion += tmp
    print("Processing id:", idx)
    idx += 1
mIOU = confusion[2] / (confusion[2] + confusion[4] + confusion[5])
Recall = confusion[2] / confusion[0]
Specificity = confusion[3] / (confusion[3] + confusion[5])
Accuracy = (confusion[2] + confusion[3]) / (confusion[0] + confusion[1])
False_alarm = confusion[5] / (confusion[2] + confusion[5])
FPR = confusion[5] / confusion[1]
FNR = confusion[4] /(confusion[4] + confusion[2])
Precision = confusion[2] / (confusion[2] + confusion[5])
FWC = 100 * (confusion[4] + confusion[5]) / (confusion[0] + confusion[1])
F_measure = 2 * Recall * Precision / (Precision + Recall)
dice = confusion[6] / confusion[7]
print("mIOU:", mIOU)

print("Specificity:", Specificity)
print("FPR:", FPR)
print("FNR:", FNR)

print("FWC:", FWC)
print("Recall:", Recall)
print("Precision:", Precision)
print("F-measure:", F_measure)
print("Dice:", dice)