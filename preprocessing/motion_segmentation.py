
import os
import sys
os.chdir(os.path.dirname(__file__))
sys.path.append(r'../')
from mosi_utils_anim.animation_data import BVHWriter, BVHReader, SkeletonBuilder
from mosi_utils_anim.animation_data.utils import rotate_euler_frames, convert_euler_frames_to_cartesian_frames, pose_orientation_general, translate_euler_frames, \
    shift_euler_frames_to_ground
from mosi_utils_anim.retargeting.directional_constraints_retargeting import estimate_scale_factor, get_kinematic_chain
import numpy as np 
import glob



def main():
    """segment and preprocess captured motions from different capturing systems into aligned motion clips
    """
    systems = ['ART', 'ViCon', 'CapturyStudio', 'OptiTrack']
    scale_factors = get_scaling_factors()

    ref_direction = np.array([0, 1])
    ref_position = np.array([0, 0, 0])

    torso_joints = {
        'ART': ['lowerLumbarSpine', 'rightUpperLeg', 'leftUpperLeg'],
        'ViCon': ['LowerBack', 'R_Femur', 'L_Femur'],
        'CapturyStudio': ['Spine2', 'RightUpLeg', 'LeftUpLeg'],
        'OptiTrack': ['Spine', 'RightUpLeg', 'LeftUpLeg']
    }

    foot_joints = {
        'ART': ['leftFoot', 'rightFoot'],
        'ViCon': ['L_Foot', 'R_Foot'],
        'CapturyStudio': ['LeftFoot', 'RightFoot'],
        'OptiTrack': ['LeftFoot', 'RightFoot']
    }
    root_joints = {
        'ART': 'pelvis',
        'ViCon': 'Root',
        'CapturyStudio': 'Hips',
        'OptiTrack': 'Hips'
    }
    rotation_orders = {
        'ART': ['Xrotation', 'Yrotation', 'Zrotation'],
        'ViCon': ['Xrotation', 'Yrotation', 'Zrotation'],
        'CapturyStudio': ['Zrotation', 'Xrotation', 'Yrotation'],
        'OptiTrack': ['Zrotation', 'Xrotation', 'Yrotation'] 
    }

    for sys in systems:

        segment_motions(sys, section_length=1200, ref_direction=ref_direction, ref_position=ref_position, torso=torso_joints[sys], 
                        foot_joints=foot_joints[sys], root_joint=root_joints[sys], rotation_order=rotation_orders[sys], scale_factor=scale_factors[sys])


def get_data_folder(system_name):
    data_folder = r'../data'
    return os.path.join(data_folder, system_name)


def get_skeleton_length(start_joint, end_joint, skeleton):
    """calculate the bone length from start_joint to end_joint

    Args:
        start_joint (str): [starting joint name]
        end_joint (str): [description]
        skeleton ([type]): [description]
    """
    kinematic_chain = [end_joint]
    current_joint = skeleton.nodes[end_joint]
    while current_joint.parent is not None:

        current_joint = current_joint.parent
        kinematic_chain.append(current_joint.node_name)
        if current_joint.node_name == start_joint:
            break    
    if start_joint not in kinematic_chain:

        root_joint = kinematic_chain[-1]
        start_chain = get_kinematic_chain(root_joint, start_joint, skeleton)
        start_chain.reverse()
        kinematic_chain.reverse()
        kinematic_chain = start_chain + kinematic_chain[1:]

    bone_length = 0

    for joint in kinematic_chain:
        if skeleton.nodes[joint].parent is not None:
            bone_length += np.linalg.norm(skeleton.nodes[joint].offset)
    return bone_length


def get_scaling_factors():
    """create scaling factor from different capturing systems to Vicon
    """
    ## Vicon reference
    skeleton_file_vicon = r'..\data\Vicon\segments\Skeleton.bvh'
    foot_joint = 'L_Toe'
    head_joint = 'Head_EndSite'
    root_joint = 'Root'
    bvhreader = BVHReader(skeleton_file_vicon)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_length_vicon = get_skeleton_length(foot_joint, head_joint, skeleton)

    ## ART
    skeleton_file_art = r'..\data\ART\segments\Skeleton.bvh'
    foot_joint = 'leftFoot'
    head_joint = 'head_EndSite'
    root_joint = 'pelvis'
    bvhreader = BVHReader(skeleton_file_art)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_length_art = get_skeleton_length(foot_joint, head_joint, skeleton)

    ## captury
    skeleton_file_captury = r'..\data\CapturyStudio\segments\Skeleton.bvh'
    foot_joint = 'RightToeBase'
    head_joint = 'Head_EndSite'
    root_joint = 'Hips'
    bvhreader = BVHReader(skeleton_file_captury)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_length_captury = get_skeleton_length(foot_joint, head_joint, skeleton)

    ## OptiTrack
    skeleton_file_OptiTrack = r'..\data\OptiTrack\segments\Skeleton.bvh'
    foot_joint = 'RightToeBase'
    head_joint = 'Head_EndSite'
    root_joint = 'Hips'
    bvhreader = BVHReader(skeleton_file_OptiTrack)
    skeleton = SkeletonBuilder().load_from_bvh(bvhreader)
    skeleton_length_OptiTrack = get_skeleton_length(foot_joint, head_joint, skeleton) 

    ArtToVicon = skeleton_length_art / skeleton_length_vicon
    CapturyToVicon = skeleton_length_captury / skeleton_length_vicon
    OptiTrackToVicon = skeleton_length_OptiTrack / skeleton_length_vicon
    scale_factors = {"ART": ArtToVicon,
                     "CapturyStudio": CapturyToVicon,
                     "ViCon": 1.0,
                     "OptiTrack": OptiTrackToVicon}
    return scale_factors       
       

def segment_motions(system_name, section_length, ref_direction, ref_position, torso, foot_joints, root_joint, rotation_order, scale_factor):
    """[summary]

    Args:
        system ([type]): [description]
        section_length ([type]): [description]
    """
    data_folder = get_data_folder(system_name)
    bvhfiles = glob.glob(os.path.join(data_folder, '*.bvh'))
    save_path = os.path.join(data_folder, 'segmentation')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for bvhfile in bvhfiles:
        filename = os.path.split(bvhfile)[-1]
        bvhreader = BVHReader(bvhfile)
        skeleton = SkeletonBuilder().load_from_bvh(bvhreader)

        ## preprocess motion data
        skeleton.scale(1.0/scale_factor)
        bvhreader.frames[:, :3] = bvhreader.frames[:, :3] * (1.0/scale_factor)
        rotated_frames = rotate_euler_frames(bvhreader.frames, 0, ref_direction, torso, skeleton, rotation_order)
        translated_frames = translate_euler_frames(rotated_frames, 0, ref_position, root_joint=root_joint, skeleton=skeleton)
        translated_frames = shift_euler_frames_to_ground(translated_frames, foot_joints, skeleton)
        
        ## segment motion data
        n_clips = len(translated_frames) // section_length
        for i in range(n_clips - 1):
            start_index = i * section_length
            end_index = (i+1) * section_length
            export_filename = '_'.join([system_name, filename[:-4], str(start_index), str(end_index)]) + '.bvh'

            BVHWriter(os.path.join(save_path, export_filename), skeleton, translated_frames[start_index: end_index], skeleton.frame_time)



if __name__ =="__main__":
    main()
