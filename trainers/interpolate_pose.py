import os

import torch
import torch.nn as nn

from scene import Scene, GaussianModel
from utils.console import CONSOLE
from utils.dynamic_utils import *

from utils.geometry_utils import matrix_to_rot6d, rot6d_to_matrix

def save_poses_securely(d, save_dir):
    filename=os.path.join(save_dir, 'obj_pose_sequence.pth')
    try:
        temp_filename = filename + ".tmp"
        torch.save(d, temp_filename)  # Save to a temporary file first
        os.replace(temp_filename, filename)  # Atomically replace the old file with the new
    except Exception as e:
        pass
    return filename

def to_transform_mat(trans, rot):
    M = torch.eye(4).to(torch.float32)
    M[:3, :3] = rot.to(torch.float32)
    M[:3, 3] = trans.to(torch.float32)
    return M

class Decomposition(nn.Module):
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.trans = nn.Parameter(torch.zeros(3, dtype=torch.float))
        self.rot_6d = nn.Parameter(matrix_to_rot6d(torch.eye(3, dtype=torch.float)))

    def forward(self):
        transform_mat = to_transform_mat(self.trans, rot6d_to_matrix(self.rot_6d))
        result = transform_mat
        for _ in range(1, self.num):  
            result = torch.matmul(result, transform_mat) 
        return result

def decompose_transform(transform_mat, num_decompose=1):
    assert transform_mat.shape == (4, 4)
    model = Decomposition(num=num_decompose)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Using SGD which acts as GD if batch size is the entire dataset
    loss_fn = nn.MSELoss() 
    
    epochs = 1500  
    for epoch in range(epochs):
        optimizer.zero_grad()  
        output = model()  
        loss = loss_fn(output, transform_mat)  
        loss.backward()  
        optimizer.step()  

        if torch.all(torch.isclose(output, transform_mat, rtol=1e-7, atol=1e-9)):
            break
    decompose = []
    # matrix_lst = []
    for i in range(num_decompose):
        decompose.append({'translation': model.trans.detach(), 'rotation': rot6d_to_matrix(model.rot_6d.detach())})
        # matrix_lst.append(to_transform_mat(model.trans.detach(), rot6d_to_matrix(model.rot_6d.detach())))
    return decompose # , matrix_lst

def interpolate_pose_seq(dataset, opt, pipe, exp_name, save_dir, dynamic_phases, obj_pose_seq_path):

    os.makedirs(save_dir, exist_ok=True)
    CONSOLE.print(f"Object pose interpolation of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print("Dynamic Phases:", dynamic_phases)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, 
        gaussians, 
        shuffle=False, 
        load_or_create_from=True,
        load_hand_masks=True, 
        load_obj_masks=True)
    viewpoints_og = scene.getTrainCameras().copy()
    viewpoints_og = sorted(viewpoints_og, key=lambda item: int(item.image_name))

    obj_pose_sequence = torch.load(obj_pose_seq_path)
    CONSOLE.log(f"Loaded object poses from {obj_pose_seq_path}")
    # First Pass of obj pose sequence, fill with None
    new_obj_pose_sequence = {}
    phase_idx = 0
    for viewpoint_cam in viewpoints_og:
        if viewpoint_cam.image_name in obj_pose_sequence:
            new_obj_pose_sequence[viewpoint_cam.image_name] = obj_pose_sequence[viewpoint_cam.image_name]
        elif int(viewpoint_cam.image_name) >= int(dynamic_phases[phase_idx][0]) and int(viewpoint_cam.image_name) <= int(dynamic_phases[phase_idx][1]):
            new_obj_pose_sequence[viewpoint_cam.image_name] = None
            
        if int(viewpoint_cam.image_name) > int(dynamic_phases[phase_idx][1]):
            phase_idx += 1
        if phase_idx > len(dynamic_phases) - 1:
            break
    
    frames_none = []
    for image_name, transformation in new_obj_pose_sequence.items():
        if transformation is None:
            frames_none.append(image_name)
        if frames_none and transformation is not None: # if record list is not None
            frames_none.append(image_name)
            translation = transformation['translation']
            rotation = transformation['rotation']
            decomposed_lst = decompose_transform(to_transform_mat(translation, rotation), len(frames_none))
            assert len(decomposed_lst) == len(frames_none) 
            for i, name in enumerate(frames_none):
                new_obj_pose_sequence[name] = decomposed_lst[i] 
            frames_none = [] # empty the list

    CONSOLE.log(f"Old sequence length {len(obj_pose_seq_path)} -> New sequence length {len(new_obj_pose_sequence)}")
    pose_seq_path = save_poses_securely(new_obj_pose_sequence, save_dir)
    CONSOLE.log(f"Interpolated object pose sequence saved at {save_dir}")
    return pose_seq_path
    

