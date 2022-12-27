import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_stack_list = []
    joint_offset = []
    my_joint_dict = {}

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = [name for name in lines[i].split()]
            if line[0] == "HIERARCHY":
                continue
            if line[0] == "MOTION":
                break
            if line[0] == "ROOT" or line[0] == "JOINT":
                joint_name.append(line[-1])
                joint_stack_list.append(line[-1])
            if line[0] == "End":
                joint_name.append(joint_name[-1]+"_end")
                joint_stack_list.append(joint_name[-1])
            if line[0] == "OFFSET":
                joint_offset.append([float(line[1]), float(line[2]), float(line[3])])
            if line[0] == "}":
                joint_index = joint_stack_list.pop()
                if joint_stack_list == []:
                    continue
                else:
                    my_joint_dict[joint_index] = joint_stack_list[-1]
    for i in joint_name:
        if i == "RootJoint":
            joint_parent.append(-1)
        else:
            joint_parent_name = my_joint_dict[i]
            joint_parent.append(joint_name.index(joint_parent_name))
    joint_offset = np.array(joint_offset).reshape(-1, 3)
    # print(joint_name)
    # print(joint_parent)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = []
    joint_orientations = []

    frame_data = motion_data[frame_id]
    frame_data = frame_data.reshape(-1, 3)

    # calculate all quaternion
    quaternion = R.from_euler('XYZ', frame_data[1:], degrees=True).as_quat()

    # end joint don't have motion data, we will create one here
    end_index = []
    for i in joint_name:
        if "_end" in i:
            end_index.append(joint_name.index(i))
    for i in end_index:
        quaternion = np.insert(quaternion, i, [0,0,0,1], axis=0)

    for index, parent in enumerate(joint_parent):
        if parent==-1:
            joint_positions.append(frame_data[0])
            joint_orientations.append(quaternion[0])
        else:
            quat = R.from_quat(quaternion)
            rotation = R.as_quat(quat[index] * quat[parent])
            joint_orientations.append(rotation)
            joint_orientations_quat = R.from_quat(joint_orientations)

            offset_rotation = joint_orientations_quat[parent].apply(joint_offset[index])
            joint_positions.append(joint_positions[parent] + offset_rotation)

    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = None
    return motion_data
