<div align="center">

  <div>&nbsp;</div>
  <div align="center">
    <h1>CCSNet:Cross-Stage Class-Specific Attention for Image
Semantic Segmentation</h1>


  </div>
  <div>&nbsp;</div>

<br />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmsegmentation)](https://pypi.org/project/mmsegmentation/)
[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/issues)

[📘Documentation](https://mmsegmentation.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmsegmentation.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmsegmentation.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmsegmentation/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>



This is the official code implementation of the paper written by my
colleague, called "Cross-Stage Class-Specific Attention for Image Semantic Segmentation".

## Introduction

Instead of simply incorporating features
from different stages, we propose a cross-stage class-specific attention
mainly for transformer-based backbones. Specifically, given a coarse prediction, we first employ the final stage features to aggregate a class center
within the whole image. Then high-resolution features from the earlier
stage are used as queries to absorb the semantics from class centers. To
eliminate the irrelevant classes within a local area, we build the context for each query position according to the classification score from
coarse prediction, and remove the redundant classes. So only relevant
classes provide keys and values in attention and participate the value
routing. We validate the proposed scheme on different datasets including
ADE20K, Pascal Context and COCO-Stuff, showing that the proposed
model improves the performance compared with other works.
It is a part of the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) project.

The master branch works with **PyTorch 1.5+**.

![demo image](resources/seg_demo.gif)

<details open>
<summary>Major features</summary>

- **Dynamic selection of class centers**

  To eliminate the irrelevant classes within a local area, we build the context for each query position according to the classification score from
coarse prediction, and remove the redundant classes.

- **Sharing keys and values**

  Different resolution receptive fields share keys and values.



</details>



Supported backbones:

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)
- [x] [PoolFormer (CVPR'2022)](configs/poolformer)


Supported datasets:

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#isaid)




## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{CCSNet2022,
    title={{CCSNet}: Cross-Stage Class-Specific Attention for Image Semantic Segmentation},
    author={CCSNet Contributors},
    howpublished = {\url{https://github.com/yaohusama/CCSNet}},
    year={2022}
}
```

## License

CCSNet is released under the Apache 2.0 license, while some specific features in this library are with other licenses. Please refer to [LICENSES.md](LICENSES.md) for the careful check, if you are using our code for commercial matters.

