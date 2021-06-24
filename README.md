# Training Only One Deep Model for Cross-Scene Foreground Segmentation

Training a single model for cross-scene foreground segmentation is a challenging task, especially for the large-scale video surveillance, as the off-the-peg models heavily rely on the scene-specific information, such that deploying the recognition system on new scenes usually requires additional annotations and workloads. Besides, the existing optical flow usage mechanism can only represent instantaneous motion and is not robust to ambient lighting changes in the open set. In this paper, we aim to achieve scene adaptation for foreground segmentation via fine-grained motion feature representation and interaction. To this end, we first design a complementary mechanism, termed as hierarchical optical flows, to generate fine-grained motion features and train an attention module. Based on this, a dynamic feature filter to realize the motion and appearance feature interaction can be constructed. Compared to the existing methods, the proposed scheme tends to learn more semantic-level information between the motion patterns of the foreground and the background area, such that a better adaptability and robustness can be obtained. Moreover, since small objects are usually missed in cross-scene foreground segmentation tasks due to the training bias, we then design a class-in scale focal loss function to balances the diversity of foreground sizes. The proposed modules can plug-and-play into a arbitrary video surveillance framework to implement cross-scene foreground segmentation. Experimental results declare that our model can significantly outperform the existing state-of-the-art methods in a large margin.

****
## Introduction
### Our work is based on our group accepeted work foreground segmentation model [STAM](https://www.mdpi.com/1424-8220/19/23/5142).
### Code uses Tensorflow 1.13, CUDN 10.1.

![Video Frame / Ground True / Optical flow / Foreground segmentation Result](https://weizongqi.github.io/HOFAM/show/test_0055.png)

## Structure
The overall structure of our proposed Hierarchical Optical Flow Attention Model (HOFAM).
![HOFAM](/show/hofam.png)

 Comparison to the baseline on DOTA for oriented object detection with ResNet-101. The figures with blue boxes are the results of the baseline and pink boxes are the results of our proposed CG-Net.
![Attention module in HOFAM](/show/atten.png)

## Experiment

|Method|Mean Dice|Recall|Precision|F-measure|
|:---:|:---:|:---:|:---:|:---:|
|HOFAM|0.9466|0.9661|0.9893|0.9776|

You first need to download [checkpoint](https://drive.google.com/file/d/1RodI2WjeG7X28T1kSTRppGmvSX95CUO8/view?usp=sharing), and then place it in checkpoint/(here)


## dataset prepare
Refer to [selflow](https://github.com/ppliuboy/SelFlow) to calculate different optical flows
```sh
Merge vidoe frame + hierarchical optical flow + ground truth like dataset/demo_data/test_000155.png
```
Prepare and Generate tfrecode file
```sh
change data path and run tfrecode.py
```

## train and test
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

## Ablation Results
Hierarchical optical flow (orange border) and foreground segmentation results.
![](/show/hop.png)

Visualization of Attention Module results.
![](/show/seg_atten.png)

Comparison results of foreground segmentation of
small objects with different losses.
![](/show/seg_loss.png)

## Cross-scene dataset Results
Comparison results of different model on crossscene dataset LIMU. Each column has five images and there are video frame, segmented results of HOFAM, PSPNet,
DeepLabV3+ and STAM, from left to right. Green: False Positive, Red: False Negative.
![](/show/seg_limu.png)

Comparison results of different Model on cross-scene
dataset LASIESTA.
![](/show/seg_la.png)