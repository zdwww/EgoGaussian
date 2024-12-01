import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import euler_angles_to_matrix

def to_transform_mat(rot):
    device = rot.device
    M = torch.eye(4, device=device).to(torch.float32)
    M[:3, :3] = rot.to(torch.float32)
    # M[:3, 3] = trans.to(torch.float32)
    return M

class ObjectMove(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_translation = nn.Parameter(torch.zeros(3, dtype=torch.float))
        self.obj_rotation_6d = nn.Parameter(matrix_to_rot6d(torch.eye(3, dtype=torch.float)))
    def forward(self, xyz):
        if xyz.shape[1] == 3:
            xyz = torch.matmul(xyz, rot6d_to_matrix(self.obj_rotation_6d).t()) + self.obj_translation
        elif xyz.shape[1] == 4: # then its Gaussians' rotation
            transform_mat = to_transform_mat(rot6d_to_matrix(self.obj_rotation_6d))
            xyz = torch.matmul(xyz, transform_mat.t())
        return xyz
    def rot_L(self, L):
        L = torch.matmul(rot6d_to_matrix(self.obj_rotation_6d), L)
        return L
    def capture(self):
        return self.obj_translation.detach(), rot6d_to_matrix(self.obj_rotation_6d.detach())
    def replace(self, new_trans, new_rot_6d):
        self.obj_translation = new_trans
        self.obj_rotation_6d = new_rot_6d

def object_move(xyz, trans, rot):
    if rot.shape != (3,3):
        rot = rot6d_to_matrix(rot)
    if xyz.shape[1] == 3:
        return torch.matmul(xyz, rot.t()) + trans
    elif xyz.shape[1] == 4: # then its Gaussians' rotation
        transform_mat = to_transform_mat(rot)
        return torch.matmul(xyz, transform_mat.t())

def reverse_object_move(xyz, trans, rot, use_inverse=True):
    # use_inverse is more precise
    assert not trans.requires_grad and not rot.requires_grad
    if rot.shape != (3,3):
        rot = rot6d_to_matrix(rot)
    # assert torch.isclose(rot.transpose(0, 1), torch.inverse(rot))
    if xyz.shape[1] == 3:
        return torch.matmul(xyz - trans, torch.inverse(rot.t()))
    elif xyz.shape[1] == 4:
        return torch.matmul(xyz, torch.inverse(to_transform_mat(rot).t()))
        
def matrix_to_rot6d(rotmat):
    """
    Convert rotation matrix to 6D rotation representation.
    Args:
        rotmat (3 x 3): (Not Batch of) rotation matrices.
    Returns:
        6D Rotations (3 x 2).
    """
    rot6d = rotmat.view(-1, 3, 3)[:, :, :2]
    # Disabled batch processing
    rot6d = torch.squeeze(rot6d)
    assert rot6d.shape == (3, 2)
    return rot6d 

def rot6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Reference: Zhou et al., "On the Continuity of Rotation Representations in Neural
    Networks", CVPR 2019
    Args:
        rot_6d (3 x 2): (Not Batch of) 6D Rotation representation.
    Returns:
        Rotation matrices (3 x 3).
    """
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rotmat = torch.stack((b1, b2, b3), dim=-1)
    # Disabled batch processing
    rotmat = torch.squeeze(rotmat)
    assert rotmat.shape == (3, 3)
    return rotmat

def get_accum_trans_rot(trans_rot):
    """
    Convert a output separate-trans-rot dictionary to accumulative 4x4 trans-rot matrix
    Args:
        trans_rot (dict): in format {image_name {"translation": tensor in shape (3), 
            "rotation": tensor in shape (3x3)}} 
    Returns:
        dict with same keys (image_name) and values 4x4 matrix, 
            each matrix the product of all previous matrices
    """
    accum_trans_rot = {}
    accum_trans_rot_mat = torch.eye(4, dtype=torch.float) # initialize as identity matrix
    for image_name in reversed(list(trans_rot.keys())):
        trans = trans_rot[image_name]["translation"]
        rot = trans_rot[image_name]["rotation"]
        
        trans_rot_mat = torch.zeros(4, 4, dtype=torch.float)
        trans_rot_mat[:3, :3] = rot
        trans_rot_mat[:3, 3] = trans
        trans_rot_mat[3, :] = torch.tensor([0, 0, 0, 1], dtype=torch.float)
        
        accum_trans_rot_mat = torch.matmul(accum_trans_rot_mat, trans_rot_mat)
        accum_trans_rot[image_name] = accum_trans_rot_mat
    return accum_trans_rot

def get_curr_xyz_from_og(trans_rot, image_name, og_xyz):
    """
    Get current gaussian's xyz from og_xyz through previous translation-rotations
    Args:
        trans_rot (dict): in format {image_name: {"translation": tensor in shape (3), 
            "rotation": tensor in shape (3x3)}} 
        image_name (str): one specific key to trans_rot
        og_xyz (torch.tensor): origianl state obtained from gaussians._xyz
    Returns:
        new_xyz 
    """
    new_xyz = og_xyz
    for key,value in trans_rot.items():
        trans = value["translation"].to("cuda")
        rot = value["rotation"].to("cuda")
        new_xyz = torch.matmul(new_xyz, rot) + trans
        if image_name == key:
            break
    return new_xyz

def get_T_seq(obj_pose_sequence):
    T_seq = {}
    for key, value in obj_pose_sequence.items(): # [:-1]:
        if value is not None:
            trans = value["translation"]
            rot = value["rotation"]
            T = torch.zeros(4, 4, dtype=torch.float)
            T[:3, :3] = rot
            T[:3, 3] = trans
            T[3, :] = torch.tensor([0, 0, 0, 1], dtype=torch.float)
            T_seq[key] = T
        else:
            T_seq[key] = None
    T_seq = {key: T_seq[key] for key in sorted(T_seq)}
    return T_seq

def get_accum_T_seq(obj_pose_sequence):
    """
    Get accumulated transformation matrices sequence, i.e. [T1, T2T1, T3T2T1, ...]
    Return:
        dict of 4x4 tensor for transform xyz
    """
    # assert list(obj_pose_sequence.values())[-1] is None, "last entry will be the just-inserted new frame"
    obj_pose_sequence = {key: obj_pose_sequence[key] for key in sorted(obj_pose_sequence)}
    T_seq = get_T_seq(obj_pose_sequence)
    accum_T_seq = {}
    accum_T = torch.eye(4)
    for key, T in T_seq.items():
        if T is not None:
            accum_T = T @ accum_T 
            accum_T_seq[key] = accum_T
        else:
            accum_T_seq[key] = None
    return accum_T_seq

def get_accum_R_seq(obj_pose_sequence):
    """
    Get accumulated rotation sequence, i.e. [R1, R2R1, R3R2R1, ...]
    Return:
        dict of 3x3 tensor for rotating covariance
    """
    obj_pose_sequence = {key: obj_pose_sequence[key] for key in sorted(obj_pose_sequence)}
    accum_R_seq = {}
    accum_R = torch.eye(3)
    for key, value in obj_pose_sequence.items():
        if value is not None:
            accum_R = value["rotation"] @ accum_R
            accum_R_seq[key] = accum_R
        else:
            accum_R_seq[key] = None
    return accum_R_seq

def apply_T_xyz(T, xyz):
    # Transform to homogenous coordinates than apply T
    assert xyz.shape[1] == 3
    xyz = torch.cat((xyz, torch.ones(xyz.shape[0], 1).to(xyz.device)), dim=1)
    xyz = xyz @ T.t().to(xyz.device)
    return xyz[:,:3]

def reverse_T_xyz(T, xyz):
    assert xyz.shape[1] == 3
    T.to(xyz.device)
    xyz = torch.cat((xyz, torch.ones(xyz.shape[0], 1).to(xyz.device)), dim=1)
    xyz = xyz @ torch.inverse(T.t()).to(xyz.device)
    return xyz[:, :3]
