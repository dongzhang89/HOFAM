# Training Only One Deep Model for Cross-Scene Foreground Segmentation

### Paper link will be given in a short time
### Author: Dong Liang, Zongqi Wei, Dong Zhang, Liyan Zhang, and Xiaoyang Tan

It is a challenging task by training a single model for cross-scene foreground segmentation, especially for large-scale video surveillance, as the off-the-peg models usually heavily rely on scene-specific information. Besides, the existing methods on optical flow mechanism can only represent instantaneous motion and are not robust to ambient changes in the open set. In this paper, we aim to achieve scene adaptation for foreground segmentation via fine-grained motion feature representations as well as interactions. To this end, we first design a new module, termed as hierarchical optical flows, to combine fine-grained motion features with an attention module. Based on this complementary mechanism, a cross-modal dynamic feature filter to realize the motion and appearance feature interaction can be constructed. Compared to the existing methods, our proposed module tends to learn more semantic information between the motion patterns of the foreground and the background area, such that better adaptability and robustness can be obtained. Moreover, since small objects are usually missed in the cross-scene foreground segmentation task due to the training bias, we further design a class-in scale focal loss function to balances the diversity of foreground instance sizes. The proposed modules can be plug-and-play into an arbitrary video surveillance recognition framework to improve the quality of cross-scene foreground segmentation masks. Experimental results on three benchmarks demonstrate that our model can significantly outperform the existing state-of-the-art methods by a large margin.

****
## Introduction
### This work is based on our conference work [STAM](https://www.mdpi.com/1424-8220/19/23/5142).
### Codes are based on Tensorflow 1.13 platform , CUDN 10.1.

![Video Frame / Ground True / Optical flow / Foreground segmentation Result](https://weizongqi.github.io/HOFAM/show/test_0055.png)

## Structure
The overall architecture of our proposed Hierarchical Optical Flow Attention Model (HOFAM).
![HOFAM](/show/hofam.png)

 Result comparisons to the baseline on DOTA dataset for oriented object detection with ResNet-101. The figures with blue boxes are the results of the baseline and pink boxes are the results of our proposed CG-Net.
![Attention module in HOFAM](/show/atten.png)

## Experiment

|Method|Mean Dice|Recall|Precision|F-measure|
|:---:|:---:|:---:|:---:|:---:|
|HOFAM|0.9466|0.9661|0.9893|0.9776|

You first need to download [checkpoint](https://drive.google.com/file/d/1RodI2WjeG7X28T1kSTRppGmvSX95CUO8/view?usp=sharing), and then place it in checkpoint/...


## Dataset prepare
Refer to [selflow](https://github.com/ppliuboy/SelFlow) to calculate different optical flows
```sh
Merge vidoe frame + hierarchical optical flow + ground truth like dataset/demo_data/test_000155.png
```
Prepare and Generate tfrecode file
```sh
change data path and run tfrecode.py
```

## Train and test
parameters setting
```sh
1. Change tfrecode file path in model.py line 137

2. Change train and test dataset in model.py  line 209 and 659

3. Change --phase(train or test) in main.py and run main.py

```
start train or test
```sh
$ run main.py
```

## Ablation results
Hierarchical optical flow (orange border) and foreground segmentation results.
![](/show/hop.png)

Visualization of Attention Module results.
![](/show/seg_atten.png)

Comparison results of foreground segmentation of
small objects with different losses.
![](/show/seg_loss.png)

## Cross-scene dataset results
Comparison results of different model on crossscene dataset LIMU. Each column has five images and there are video frame, segmented results of HOFAM, PSPNet,
DeepLabV3+ and STAM, from left to right. Green: False Positive, Red: False Negative.
![](/show/seg_limu.png)

Comparison results of different Model on cross-scene
dataset LASIESTA.
![](/show/seg_la.png)