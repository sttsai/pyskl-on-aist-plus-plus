import sys
import types
import pickle
import configparser
from pathlib import Path
import numpy as np

def parse_args():
    config_file = sys.argv[1] if len(sys.argv) > 1 else ""
    if config_file == None or len(config_file) == 0:
        config_file = 'segment_by_time_60.ini'

    config = configparser.ConfigParser()
    config.read(config_file)

    # Path Argument
    args = types.SimpleNamespace()
    args.anno_dir           = Path(config['argument']['anno_dir'])
    args.ignore_file        = config['argument']['ignore_file']
    args.label_file         = config['argument']['label_file']
    args.output_file        = config['argument']['output_file']
    args.split_file_train   = config['argument']['split_file_train']
    args.split_file_val     = config['argument']['split_file_val']
    args.split_file_test    = config['argument']['split_file_test']

    # Other Argument
    args.num_frames = int(config['argument']['num_frames'])
    args.skeleton_format = config['argument']['skeleton_format']
    args.threshold = int(config['argument']['threshold'])
    args.segment_type = config['argument']['segment_type']

    print("args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    return args

def get_split_list(split_file_name, num_clip_dict):
    open_f = open(split_file_name, 'r')
    data = open_f.read()
    org_list = data.split('\n')
    new_list = []
    for id in org_list:
        if id not in num_clip_dict.keys():
            continue
        num_clip = num_clip_dict[id]

        for i in range(num_clip):
            new_list.append(id + f'_{i + 1:02d}')
    return new_list


def segment_fix(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list):
    num_clip = 1
    for i in range(0, total_len, args.num_frames):
        start_i = i if i + args.num_frames <= total_len else total_len - args.num_frames
        end_i = start_i + args.num_frames

        anno_dict = {}
        anno_dict['keypoint'] = np.expand_dims(keypoints[start_i:end_i], axis=0)
        anno_dict['frame_dir'] = pkl_filepath.stem + f'_{num_clip:02d}'
        anno_dict['total_frames'] = args.num_frames
        anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])

        anno_list.append(anno_dict)
        num_clip += 1
    num_clip_dict[pkl_filepath.stem] = num_clip - 1

def segment_random(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list):
    num_clips = total_len // args.num_frames
    clip_len = args.num_frames
    num_clip = 1
    for i in range(num_clips):
        start = np.random.randint(0, total_len-clip_len)
        end = start + clip_len
        assert end <  total_len

        anno_dict = {}
        anno_dict['keypoint'] = np.expand_dims(keypoints[start:end], axis=0)
        anno_dict['frame_dir'] = pkl_filepath.stem + f'_{num_clip:02d}'
        anno_dict['total_frames'] = args.num_frames
        anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])

        anno_list.append(anno_dict)
        num_clip += 1
    num_clip_dict[pkl_filepath.stem] = num_clip - 1

def segment_uniform_get_inds(num_frames, clip_len):
    allinds = []
    num_clips = num_frames // clip_len
    p_interval = (1, 1)

    for clip_idx in range(num_clips):
        old_num_frames = num_frames
        pi = p_interval
        ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
        num_frames = int(ratio * num_frames)
        off = np.random.randint(old_num_frames - num_frames + 1)

        if num_frames < clip_len:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + clip_len)
        elif clip_len <= num_frames < 2 * clip_len:
            basic = np.arange(clip_len)
            inds = np.random.choice(
                clip_len + 1, num_frames - clip_len, replace=False)
            offset = np.zeros(clip_len + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // clip_len for i in range(clip_len + 1)])
            bsize = np.diff(bids)
            bst = bids[:clip_len]
            offset = np.random.randint(bsize)
            inds = bst + offset

        inds = inds + off
        num_frames = old_num_frames

        allinds.append(inds)

    np.concatenate(allinds)

    return allinds


def segment_uniform(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list):
    inds = segment_uniform_get_inds(total_len, args.num_frames)
    #print(f"inds lens:{len(inds)} ")
    #print(f"{inds} ")

    num_clip = 1
    for index_array in inds:
        new_keypoints =  keypoints[index_array]

        #print(f"new:{new_keypoints[0][0]}  org{keypoints[index_array[0]][0]}");
        assert new_keypoints[0][0][0] == keypoints[index_array[0]][0][0]
        assert new_keypoints[1][0][0] == keypoints[index_array[1]][0][0]

        anno_dict = {}
        anno_dict['keypoint'] = np.expand_dims(new_keypoints, axis=0)
        anno_dict['frame_dir'] = pkl_filepath.stem + f'_{num_clip:02d}'
        anno_dict['total_frames'] = args.num_frames
        anno_dict['label'] = label_list.index(anno_dict['frame_dir'][1:3])

        anno_list.append(anno_dict)
        num_clip += 1
    num_clip_dict[pkl_filepath.stem] = num_clip - 1


def main(args):

    label_list = None
    with open(args.label_file, 'r') as label_f:
        data = label_f.read()
        label_list = data.split('\n')
    print(f'label:{label_list}')

    ignore_file_list = []
    with open(args.ignore_file, 'r') as ignore_f:
        data = ignore_f.read()
        ignore_file_list = data.split('\n')

    final_dict={}
    label_counter = {}
    anno_list = []
    num_clip_dict={}

    pklFiles = args.anno_dir.glob('*.pkl')
    print(f"load pkl files:{len((list(pklFiles)))}")

    for pkl_filepath in args.anno_dir.glob('*.pkl'):
        if pkl_filepath.stem in ignore_file_list:
            continue

        fileLabel = pkl_filepath.stem[1:3]
        if 'sBM' in pkl_filepath.stem:
            fileLabel += 'sBM'
        elif 'sFM' in pkl_filepath.stem:
            fileLabel += 'sFM'
            continue

        if args.threshold > 0:
            label_counter[fileLabel] = label_counter.get(fileLabel, 0) + 1
            if label_counter[fileLabel] > args.threshold:
                continue
            if label_counter[fileLabel] == args.threshold:
                print(f"Label {fileLabel} count exceeds the threshold.")

        pkl_f = open(pkl_filepath, 'rb')
        data = pickle.loads(pkl_f.read())

        keypoints = None
        if args.skeleton_format == 'smpl':
            keypoints = data['smpl_poses'].reshape(-1, 24, 3).astype(np.float64)
        elif args.skeleton_format == 'smpl24':
            smpl_keypoints3d = data['smpl_keypoints3d'].reshape(-1, 45, 3).astype(np.float64)
            keypoints = smpl_keypoints3d[:, :24, :]
        elif args.skeleton_format == 'smpl45':
            smpl_keypoints3d = data['smpl_keypoints3d'].reshape(-1, 45, 3).astype(np.float64)
            keypoints = smpl_keypoints3d
        assert (keypoints is not None), "Invalid skeleton format"

        new_keypoints = []
        for x in keypoints:
            if False in np.isfinite(x):
                continue
            new_keypoints.append(x)
        keypoints = np.array(new_keypoints)
        assert False not in np.isfinite(keypoints), "NAN or inf in the skeleton data"

        total_len = keypoints.shape[0]
        if args.num_frames > total_len:
            continue

        if args.segment_type == 'fix':
            segment_fix(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list)
        elif args.segment_type == 'random':
            segment_random(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list)
        elif args.segment_type == 'uniform':
            segment_uniform(anno_list, num_clip_dict, keypoints, total_len, pkl_filepath, label_list)

    final_dict['annotations'] = anno_list
    print(f"Dataset size:{len(final_dict['annotations'])}")

    ## Write split
    split_dict = {}

    split_dict = {}
    split_dict['train'] = get_split_list(args.split_file_train, num_clip_dict)
    split_dict['val']   = get_split_list(args.split_file_val,   num_clip_dict)
    split_dict['test']  = get_split_list(args.split_file_test,  num_clip_dict)
    final_dict['split'] = split_dict

    print(f" split_dict['train']:{len(split_dict['train'])}")  # 11683
    print(f" split_dict['val']:{len(split_dict['val'])}")  # 965
    print(f" split_dict['test']:{len(split_dict['test'])}")  # 5787

    # Dump into pickle file
    with open(args.output_file, 'wb') as out_f:
        pickle.dump(final_dict, out_f)

if __name__ == '__main__':
    args = parse_args()
    main(args)
