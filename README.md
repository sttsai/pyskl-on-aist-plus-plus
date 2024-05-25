# PYSKL on AIST++ Dataset

This Repo is a copy of [PYSKL](https://github.com/kennymckormick/pyskl/tree/main), and we hope you first understand how PYSKL works. 

The following modifications have been added to this Repo:
1. In pyskl/datasets/pipelines/pose_related.py, we add SMPL format to class JointToBone.
2. In pyskl/utils/graph.py, we add SMPL format to class Graph.
3. In pyskl/datasets/pipelines/sampling.py, we add the class [ContinuousSampleFrames, ContinuousSample, ContinuousSampleDecode] and [RandomSampleFrames, RandomSample, RandomSampleDecode].
4. In direction ./data_segmentation/aist_keypoints3d/, we add tools to convert AIST++ Dataset to pyskl format

## Installation
```shell
git clone https://github.com/sttsai/pyskl-on-aist-plus-plus.git
cd pyskl-on-aist-plus-plus
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
```

## Data Preparation

1. Download the annotations from [AIST++ website](https://google.github.io/aistplusplus_dataset/factsfigures.html).
2. Use the following command to convert AIST++ annotations into keypoint3d
```shell
python data_segmentation/aist_keypoints3d/ConverSmplPoseToKeypoint3d.py
```
3. Use the following commands to segment the data according to experimental needs.  The configuration file is in INI format.  Some sample configuration files are in ./configs/segmentation
```shell
python data_segmentation/aist_keypoints3d/Segment_by_time.py configs/segmentation_smpl24/{config_name}
```

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
bash tools/dist_train.sh configs_smpl24_4A/aagcn/aagcn_aist++_smpl24_60/j.py 1 --validate --test-last --test-best

# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
bash tools/dist_test.sh configs_smpl24_4A/aagcn/aagcn_aist++_smpl24_60/j.py --eval top_k_accuracy mean_class_accuracy
```
For specific examples, please go to the README for each specific algorithm we supported.

## Citation


```BibTeX

```



