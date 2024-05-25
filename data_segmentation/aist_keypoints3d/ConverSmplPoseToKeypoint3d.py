import os
import glob

from absl import app
from absl import flags
from aist_plusplus.loader import AISTDataset
from aist_plusplus.visualizer import plot_on_video
from smplx import SMPL
import torch
import time
import pickle
import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from  tools.plot_smpl_tools import *

FLAGS = flags.FLAGS
flags.DEFINE_string('anno_dir',  '../../../AIST++/annotations', 'input local dictionary for AIST++ annotations.')
flags.DEFINE_string('smpl_dir',  '../../../AIST++/SMPL',        'input local dictionary that stores SMPL data.')
flags.DEFINE_string('output_dir','../../../AIST++/smpl_keypoints3d', 'output local dictionary for SMPL keypoints3d .')

def loadDataAndShowKeypoints3d(seq_name):
    file_path = os.path.join(FLAGS.output_dir, f'{seq_name}.pkl')
    assert os.path.exists(file_path), f'File {file_path} does not exist!'
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    smpl_keypoints3d = data['smpl_keypoints3d']
    plot_smpl_keypoints3d(smpl_keypoints3d)

def main(_):
    aist_dataset = AISTDataset(FLAGS.anno_dir)
    smplModel = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    motion_path = os.path.join(FLAGS.anno_dir, "motions" )
    motion_dir = Path(motion_path)

    paths = motion_dir.glob('*.pkl')
    print(f"load pkl files:{len((list(paths)))}")

    for path in tqdm(motion_dir.glob('*.pkl')):
        filename = path.stem
        converter(aist_dataset, smplModel, filename)


def converter(aist_dataset, smplModel, file_name):
    # Parsing data info.
    seq_name, view = AISTDataset.get_seq_name(file_name)
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(aist_dataset.motion_dir, seq_name)

    keypoints3d = smplModel.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(), # 這裡取[720,1], 即取每1個frame的第1個joint的x
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),      # 這裡取[720,71] ,在forward()中會將兩個再合在一起,即由提供每1幀 第0個joint.x軸向由global_orient決定
        transl=torch.from_numpy(smpl_trans).float(),
        scaling=torch.from_numpy(smpl_scaling.reshape(1, 1)).float(),
    ).joints.detach().numpy()

    nframes, njoints, _ = keypoints3d.shape

    saveData_keypoints3d(smpl_poses, smpl_trans, keypoints3d, seq_name)
    return

def saveData_keypoints3d(smpl_poses, smpl_trans, smpl_keypoints3d, seq_name):
    savePath = os.path.join( FLAGS.output_dir, f'{seq_name}.pkl'  )
    with open(savePath, 'wb') as pickle_file:
        pickle.dump({'smpl_poses': smpl_poses, 'smpl_trans':smpl_trans, 'smpl_keypoints3d': smpl_keypoints3d}, pickle_file)

if __name__ == '__main__':
    app.run(main)


