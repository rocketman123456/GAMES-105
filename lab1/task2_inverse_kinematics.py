from task1_forward_kinematics import *
from scipy.spatial.transform import Rotation as R
from Lab2_IK_answers import *


def part1_simple(viewer, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在腰部，末端节点设在左手
    """
    viewer.create_marker(target_pos, [1, 0, 0, 1])
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    # meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lShoulder', 'lWrist')
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'RootJoint', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()

    joint_position, joint_orientation = part1_inverse_kinematics(meta_data, joint_position, joint_orientation, target_pos)
    viewer.show_pose(joint_name, joint_position, joint_orientation)
    viewer.run()
    pass


def part1_hard(viewer, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在**左脚部**，末端节点设在左手
    """
    viewer.create_marker(target_pos, [1, 0, 0, 1])
    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()

    joint_position, joint_orientation = part1_inverse_kinematics(meta_data, joint_position, joint_orientation, target_pos)
    viewer.show_pose(joint_name, joint_position, joint_orientation)
    viewer.run()
    pass


def part1_animation(viewer, target_pos):
    """
    如果正确完成了part1_inverse_kinematics， 此处不用做任何事情
    可以通过`wasd`控制marker的位置
    """
    marker = viewer.create_marker(target_pos, [1, 0, 0, 1])

    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()
    class UpdateHandle:
        def __init__(self, marker, joint_position, joint_orientation):
            self.marker = marker
            self.joint_position = joint_position
            self.joint_orientation = joint_orientation

        def update_func(self, viewer):
            target_pos = np.array(self.marker.getPos())
            self.joint_position, self.joint_orientation = part1_inverse_kinematics(meta_data, self.joint_position, self.joint_orientation, target_pos)
            viewer.show_pose(joint_name, self.joint_position, self.joint_orientation)
    handle = UpdateHandle(marker, joint_position, joint_orientation)
    handle.update_func(viewer)
    viewer.update_marker_func = handle.update_func
    viewer.run()


def part2_one_pose(viewer, bvh_name, target_pos):
    """
    完成part1_inverse_kinematics，我们将根节点设在**左脚部**，末端节点设在左手
    """
    joint_name, joint_parent, joint_offset = part1_calculate_T_pose(bvh_name)
    motion_data = load_motion_data(bvh_name)
    joint_positions, joint_orientations = part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, 0)
    
    meta_name, meta_parent, joint_initial_position = viewer.get_meta_data()
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lShoulder', 'lWrist')
    joint_position, joint_orientation = part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pos)
    # joint_positions, joint_orientations = part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, 0.1, 0.3, 1.4)

    viewer.show_pose(joint_name, joint_positions, joint_orientations)
    viewer.run()
    pass


def part2(viewer, bvh_name):
    motion_data = load_motion_data(bvh_name)
    bvh_joint_name, bvh_joint_parent, bvh_offset = part1_calculate_T_pose(bvh_name)
    joint_name, _, joint_initial_position = viewer.get_meta_data()
    idx = [joint_name.index(name) for name in bvh_joint_name]
    meta_data = MetaData(bvh_joint_name, bvh_joint_parent, joint_initial_position[idx], 'lShoulder', 'lWrist')
    class UpdateHandle:
        def __init__(self, meta_data, motion_data, joint_offset):
            self.meta_data = meta_data
            self.motion_data = motion_data
            self.joint_name = meta_data.joint_name
            self.joint_parent = meta_data.joint_parent
            self.joint_offset = joint_offset
            self.current_frame = 0

        def update_func(self, viewer):
            joint_position, joint_orientation = part2_forward_kinematics(self.joint_name, self.joint_parent, self.joint_offset, self.motion_data, self.current_frame)
            joint_position, joint_orientation = part2_inverse_kinematics(self.meta_data, joint_position, joint_orientation, 0.1, 0.3, 1.4)
            viewer.show_pose(self.joint_name, joint_position, joint_orientation)
            self.current_frame = (self.current_frame + 1) % self.motion_data.shape[0]
    handle = UpdateHandle(meta_data, motion_data, bvh_offset)
    viewer.update_func = handle.update_func
    viewer.run()
    pass


def bonus(viewer, left_target_pos, right_target_pos):
    left_marker = viewer.create_marker(left_target_pos, [1, 0, 0, 1])
    right_marker = viewer.create_marker2(right_target_pos, [0, 0, 1, 1])

    joint_name, joint_parent, joint_initial_position = viewer.get_meta_data()

    # 为了兼容如此设置，实际上末端节点应当为左右手
    meta_data = MetaData(joint_name, joint_parent, joint_initial_position, 'lToeJoint_end', 'lWrist_end')
    joint_position = viewer.get_joint_positions()
    joint_orientation = viewer.get_joint_orientations()

    class UpdateHandle:
        def __init__(self, left_marker, right_marker, joint_position, joint_orientation):
            self.left_marker = left_marker
            self.right_marker = right_marker
            self.joint_position = joint_position
            self.joint_orientation = joint_orientation
            
        def update_func(self, viewer):
            left_target_pos = np.array(self.left_marker.getPos())
            right_target_pos = np.array(self.right_marker.getPos())
            self.joint_position, self.joint_orientation = bonus_inverse_kinematics(meta_data, self.joint_position, self.joint_orientation, left_target_pos, right_target_pos)
            viewer.show_pose(joint_name, self.joint_position, self.joint_orientation)
    handle = UpdateHandle(left_marker, right_marker, joint_position, joint_orientation)
    handle.update_func(viewer)

    viewer.update_marker_func = handle.update_func
    viewer.run()


def main():
    viewer = SimpleViewer()

    # part1
    part1_simple(viewer, np.array([0.5, 0.75, 0.5]))
    # part1_hard(viewer, np.array([0.5, 0.5, 0.5]))
    # part1_animation(viewer, np.array([0.5, 0.5, 0.5]))

    # part2
    # part2_one_pose(viewer, 'data/walk60.bvh', np.array([0.5, 0.75, 0.5]))
    # part2(viewer, 'data/walk60.bvh')

    # bonus
    # bonus(viewer, np.array([0.5, 0.5, 0.5]), np.array([0, 0.5, 0.5]))

if __name__ == "__main__":
    main()