# TDPNet: Single View Point Cloud Generation via Unified 3D Prototype

This repository contains the PyTorch implementation of the paper:

[Single View Point Cloud Generation via Unified 3D Prototype](). <br>
Yu Lin, Yigong Wang, Yifan Li, Yang Gao, Zhuoyi Wang, Latifur Khan <br>
In [AAAI 2021] (https://aaai.org/Conferences/AAAI-21/)

## Introduction
In this project, we are focusing on the point cloud reconstruction from a single image using prior 3D shape information, we called it 3D prototype. Previous methods usually consider 2D information only, or treat 2D information and 3D information equally. However, 3D information are more informative and should be utilized during the reconstruction process. Our solution is that we pre-compute a set of 3D prototype features from a point cloud dataset and infuse them with the incoming image features. We also designed a hierarchical point cloud decoder that treat each prototype separately. Empirically, we show that TDPNet achieves SOTA performance in point cloud single view reconstruction. We additionally found that a good quantitative results does not guarantee a good visual result.

## Intuition

<img src='imgs/intuition.png'>


