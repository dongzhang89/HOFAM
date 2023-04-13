# CrossNet 

This repository contains the official PyTorch implementation of the following paper:

#### Cross-scene Background Subtraction Network via 3D Optical Flow

Dong Liang, Dong Zhang, Qiong Wang, Zongqi Wei, Liyan Zhang
MUAA, NJUST, SIAT 
https://ieeexplore.ieee.org/document/10100916

## Abstract 
<p align="justify">
This paper investigates an intriguing yet unsolved problem of cross-scene background subtraction for training only one deep model to process large-scale video streaming. We propose an end-to-end cross-scene background subtraction network via 3D optical flow, dubbed CrossNet. First, we design a new motion descriptor, hierarchical 3D optical flows (3D-HOP), to observe fine-grained motion. Then, we build a cross-modal dynamic feature filter (CmDFF) to enable the motion and appearance feature interaction. CrossNet exhibits better generalization since the proposed modules are encouraged to learn more discriminative semantic information between the foreground and the background. Furthermore, we design a loss function to balance the size diversity of foreground instances since small objects are usually missed due to training bias. Our whole background subtraction model is called Hierarchical Optical Flow Attention Model (HOFAM). Unlike most of the existing stochastic-process-based and CNN-based background subtraction models, HOFAM will avoid inaccurate online model updating, not heavily rely on scene-specific information, and well represent ambient motion in the open world. Experimental results on several well-known benchmarks demonstrate that it outperforms state-of-the-art by a large margin. The proposed framework can be flexibly integrated into arbitrary streaming media systems in a plug-and-play form.

****
## Introduction
### Codes are based on Tensorflow 1.13 platform , CUDN 10.1.

![Video Frame / Ground True / Optical flow / Foreground segmentation Result](https://weizongqi.github.io/HOFAM/show/test_0055.png)

## The overall architecture
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