# SLucAM - A Comparison of FeaturesDescriptors on Tracking in Monocular Visual-SLAM
This repo contains the code of Luca Lobefaro's Master's thesis in Artificial Intelligence and Robotics for University of Roma "La Sapienza". \
In the following is briefly presented the thesis and it is descripted how to use it. \
Thesis: [A Comparison of Features Descriptors on Tracking in Monocular Visual-SLAM](doc/A_Comparison_of_Features_Descriptors_on_Tracking_in_Monocular_Visual_SLAM___Luca_Lobefaro.pdf)

## Abstract
The thesis has two purposes: the first is to propose a novel monocular visual-SLAM
system that is efficient and able to produce good results in trajectory prediction,
the second is to test how different image features affect the tracking. \
In particular, the SLAM system is developed from scratch, using different techniques
from literature, with great focus on initialization and tracking. It does not have the
aim to produce results that are better than existing systems, but instead to produce
a good baseline on which to determine what is the drift accumulated on various
sequences using different features. This while keeping the algorithm light and easy
to compute. \
The features explored are three: we have ORB as baseline and representative of
classical approaches and then Superpoint and LF-NET are used to understand how
good neural features can be on this kind of system. Actually, it is shown in the
results that neural features outperform ORB in almost any case. In particular,
the proposed system, in combination with Superpoint, is able to maintain a good
tracking in every sequence tested, even in those that are very difficult, because full
of noisy images and motion blur.

## Features Examples
An example of keypoints extracted with ORB:

![](doc/thesis%20images/orb_feats_fr1_desk.png)

An example of keypoints extracted with Superpoint: 

![](doc/thesis%20images/superpoint_feats_fr1_desk.png)

An example of keypoints extracted with Superpoint: 

![](doc/thesis%20images/lf_net_feats_fr1_desk.png)

## Tracking Examples
An example of tracking on the sequence fr1_desk using Superpoint features:

![](doc/thesis%20videos/fr1_desk_superpoint.gif)

# Usage

## Prerequisites
The system is tested on Ubuntu 22.04 LTS. You need to install the following libraries/frameworks/packages in order to be able to run the code:

- C++17 compiler
- OpenCV == 4.6.0
- Eigen == 3.4.0
- MATLAB == R2022a
- Python == 2.7.18
- Python == 3.10.4
- ffmpeg == 4.4.2

It cannot be confirmed if higher versions of them works without any problem.

## How to run the code
First, you need to build the code with the following command:

```
./build.sh
```

Then you can run the one of the examples present in "data/datasets/" by typing:

```
bin/SLucAM dataset_name features_name
```

where "dataset_name" is one of the examples in "data/datasets" and "features_name" is one of the following features: orb, superpoint, lf_net. \
Important: in order to use superpoint or lf_net you need the features extracted from them. \
To extract the features with superpoint:

```
python3 external/superpoint/extract_superpoint.py path_to_dataset/rgb/ path_to_dataset/superpoint/
```

(the implementation of this network is given by the repo: [Superpoint](https://github.com/rpautrat/SuperPoint.git), so all the rights are reserved to the authors that make it available for non-commercial use. Please refer to them for the license, that is also available at [superpoint-LICENSE](external/superpoint/LICENSE)). If the folder "path_to_dataset/superpoint/" is not present, please create it.

To extract the features with LF-NET:

```
python3 external/lf_net/extract_lf_net.py path_to_dataset/ path_to_lf_net_outputs/
```

where "path_to_lf_net_outputs/" is the outputs folder where the code of the authors ([LF-NET](https://github.com/vcg-uvic/lf-net-release.git)) extract the informations about the images. So please refer to such repo in order to extract the points first.

## How to evaluate results
To evaluate results please use the scripts present in the folder "evaluation/", where we have a script for each dataset proposed. The plots will be produced in the relative folders.

## How to plot results
To plot the results, please use the script "plots/main.m" in a MATLAB environment, by setting the directory of the dataset to plot. All the images, one for each frame, will be produced in the "plots/images/" folder (this will take a while). Important: ensure that the folder "plots/images/" is empty before to start the script. \
To produce a video from the images run: 

```
cd plots/
./generate_video.sh 10
```

This will generate a file "output.mp4" in the folder "plots/videos/".

# References
The repo containing the pre-trained code for superpoint used is at the following link: [Superpoint](https://github.com/rpautrat/SuperPoint.git) (all the rights are reserved to the authors, the code is available for research use as expressed in the file [superpoint LICENSE](external/superpoint/LICENSE), so please refer to them for the license). \
The repo containing the LF-NET code, the one proposed by the authors, can be downloaded at the following link: [LF-NET](https://github.com/vcg-uvic/lf-net-release.git).