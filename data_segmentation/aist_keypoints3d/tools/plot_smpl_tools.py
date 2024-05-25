import matplotlib; matplotlib.use("TkAgg") # 可以顯示一個窗視出來
import matplotlib.pyplot as plt
import numpy as np

SMPL_JOINT_45 = [
    "root",                                 # 0
    "lhip", "rhip", "belly",                # 1-3
    "lknee", "rknee", "spine",              # 4-6
    "lankle", "rankle", "chest",            # 7-9
    "ltoes", "rtoes", "neck",               # 10-12
    "linshoulder", "rinshoulder",           # 13-14
    "head",  "lshoulder", "rshoulder",      # 15-17
    "lelbow", "relbow",                     # 18-19
    "lwrist", "rwrist",                     # 20-21
    "lhand", "rhand",                       # 22-23 SMPL到這裡, 以下為SMPL 1.1.0 ~ 45
    "nose",                                 # 24
    "R_Eye", "L_Eye",                       # 25-26
    'R_Ear', 'L_Ear',                       # 27-28
    'L_BigToe', 'L_SmallToe', 'L_Heel',     # 29-31
    'R_BigToe','R_SmallToe', 'R_Heel',      # 32-34
    'L_Hand_thumb', 'L_Hand_index', 'L_Hand_middle', 'L_Hand_ring', 'L_Hand_pinky', # 35-39
    'R_Hand_thumb', 'R_Hand_index','R_Hand_middle', 'R_Hand_ring', 'R_Hand_pinky',  # 40-44
]

BonePairs = [
    (23, 21), (21, 19), (19, 17),                             # Right Arm
    (21, 40), (21, 41), (21, 42), (21, 43),(21, 44),          # Right Hand(SMPLX)
    (22, 20), (20, 18), (18, 16),                             # Left Arm
    (20, 35), (20, 36), (20, 37), (20, 38),(20, 39),          # Left Hand(SMPLX)
    (2, 5), (5, 8), (8, 11),                                  # Right Leg
    (8, 32), (8, 33), (8, 34),                                # Right Foot(SMPLX)
    (1, 4), (4, 7), (7, 10),                                  # Left Leg
    (7, 29), (7, 30), (7, 31),                                # Left Foot(SMPLX)
    (24, 25), (25, 27),                                       # Right Face
    (24, 26), (26, 28),                                       # Left Face
    (15, 12), (12, 9), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), (17, 14), (14, 9), (16, 13), (13, 9)  # Body
]

BonePairs_Body = [
    (15, 12), (12, 9), (9, 6), (6, 3), (3, 0), (0, 2), (0, 1), (17, 14), (14, 9), (16, 13), (13, 9)  # Body
]

BonePairs_Left = [
    (22, 20), (20, 18), (18, 16),                             # Left Arm
    (20, 35), (20, 36), (20, 37), (20, 38),(20, 39),          # Left Hand(SMPLX)
    (1, 4), (4, 7), (7, 10),                                  # Left Leg
    (7, 29), (7, 30), (7, 31),                                # Left Foot(SMPLX)
    (24, 26), (26, 28),                                       # Left Face
]

BonePairs_Right = [
    (23, 21), (21, 19), (19, 17),                             # Right Arm
    (21, 40), (21, 41), (21, 42), (21, 43),(21, 44),          # Right Hand(SMPLX)
    (2, 5), (5, 8), (8, 11),                                  # Right Leg
    (8, 32), (8, 33), (8, 34),                                # Right Foot(SMPLX)
    (24, 25), (25, 27),                                       # Right Face
]

# 顯示全部
def plot_3d_keypoints(keypoints3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for frame in range(keypoints3d.shape[0]):
        x, y, z = keypoints3d[frame, :, 0], keypoints3d[frame, :, 1], keypoints3d[frame, :, 2]
        ax.scatter(x, y, z, marker='o', label=f'Frame {frame}')
        break #先只顯示一張
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.legend()
    plt.show()

# 畫出骨架
def plot_BonePairs(ax, keypoints3d, frame_index, bone_paris, color='gray'):
    # 按BoneParis畫出
    for pair in bone_paris:
        joint1 = keypoints3d[frame_index, pair[0], :]
        joint2 = keypoints3d[frame_index, pair[1], :]
        ax.plot([joint1[0], joint2[0]], [joint1[1], joint2[1]], [joint1[2], joint2[2]], color=color)

# 畫出骨架 分左右
def plot_BonePairs_LR(ax, keypoints3d, frame_index):
    plot_BonePairs(ax, keypoints3d, frame_index, BonePairs_Body,  'green')
    plot_BonePairs(ax, keypoints3d, frame_index, BonePairs_Right, 'blue') #右藍色
    plot_BonePairs(ax, keypoints3d, frame_index, BonePairs_Left,  'red')

# 只顯示某一帳
def plot_3d_keypoints_single_frame(keypoints3d, frame_index, axis_ranges):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = keypoints3d[frame_index, :, 0], keypoints3d[frame_index, :, 1], keypoints3d[frame_index, :, 2]
    ax.scatter(x, y, z, marker='o', label=f'Frame {frame_index}')

    # 畫骨架
    #plot_BonePairs(ax, keypoints3d, frame_index, BonePairs)
    plot_BonePairs_LR(ax, keypoints3d, frame_index)

    # 顯示joint id
    for i, txt in enumerate(range(len(x))):
        #ax.text(x[i], y[i], z[i], f'{i}({x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f})', fontsize=8)
        ax.text(x[i], y[i], z[i], f'{i}', fontsize=6)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 設定大小值
    if axis_ranges is not None:
        ax.set_xlim(axis_ranges[0]) # ax.set_xlim([-100, 100])
        ax.set_ylim(axis_ranges[1])# ax.set_ylim([57, 255])
        ax.set_zlim(axis_ranges[2])# ax.set_zlim([-100, 100])
    else:
        ax.autoscale() # 自動計算顯示範圍

    plt.legend()
    plt.show()

# 動畫更新方式1
def plot_3d_keypoints_single_frame_anim(frame_index, ax, keypoints3d, axis_ranges, pauseTimer=0):
    #print(f"frame_index:{frame_index}")
    ax.cla()
    x, y, z = keypoints3d[frame_index, :, 0], keypoints3d[frame_index, :, 1], keypoints3d[frame_index, :, 2]
    ax.scatter(x, y, z, marker='o', label=f'Frame {frame_index}')

    # 畫骨架
    # plot_BonePairs(ax, keypoints3d, frame_index, BonePairs)
    plot_BonePairs_LR(ax, keypoints3d, frame_index)

    # 顯示joint id
    for i, txt in enumerate(range(len(x))):
        # ax.text(x[i], y[i], z[i], f'({x[i]:.2f}, {y[i]:.2f}, {z[i]:.2f})', fontsize=8)
        ax.text(x[i], y[i], z[i], f'{i}', fontsize=6)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 設定大小值
    if axis_ranges is not None:
        ax.set_xlim(axis_ranges[0])  # ax.set_xlim([-100, 100])
        ax.set_ylim(axis_ranges[1])  # ax.set_ylim([57, 255])
        ax.set_zlim(axis_ranges[2])  # ax.set_zlim([-100, 100])
    else:
        ax.autoscale()  # 自動計算顯示範圍

    ax.legend()

    # 自已更新時要帶入
    if(pauseTimer>0):
        plt.pause(pauseTimer)


def plot_smpl_keypoints3d(keypoints3d):
    # 找到各轴的最小和最大值
    x, y, z = keypoints3d[:, :, 0], keypoints3d[:, :, 1], keypoints3d[:, :, 2]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    axis_ranges = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])  # 合成1個Array
    #print(f'X軸範圍：{xmin} 到 {xmax}')
    #print(f'Y軸範圍：{ymin} 到 {ymax}')
    #print(f'Z軸範圍：{zmin} 到 {zmax}')

    # 顯示單1張
    # plot_3d_keypoints_single_frame(keypoints3d,1, axis_ranges)
    # return

    # 顯示動劃方式1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for frame_index in range(len(keypoints3d)):
        plot_3d_keypoints_single_frame_anim(frame_index, ax, keypoints3d, axis_ranges, 0.016)

    plt.show()