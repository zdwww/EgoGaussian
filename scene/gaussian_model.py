#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
import torch
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.geometry_utils import matrix_to_rot6d, ObjectMove, object_move, reverse_object_move, apply_T_xyz, reverse_T_xyz

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.covariance_activation_w_rot = self.build_covariance_from_scaling_rotation_w_rot

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def build_covariance_from_scaling_rotation_w_rot(self, scaling, scaling_modifier, rotation, accum_R, which_object=None, during_training=False):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        if which_object is not None:
            indices = torch.nonzero(self.get_is_object == which_object).squeeze()
        else:
            indices = torch.nonzero(torch.ones_like(self.get_is_object)).squeeze()
        L_selected = L[indices]
        if accum_R is None:
            accum_R = torch.eye(3)
        if during_training:
            L_selected_transformed = self.trainable_object_move.rot_L(torch.matmul(accum_R.to(L_selected.device), L_selected))
        else:
            L_selected_transformed = torch.matmul(accum_R.to(L_selected.device), L_selected)
        L_transformed = L.clone()
        L_transformed[indices] = L_selected_transformed
        actual_covariance = L_transformed @ L_transformed.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self._label = torch.empty(0) # float: *trainable* Label for object identity
        self._generation = torch.empty(0) # int: Gaussian's generation identifier, 0 when created initially
        self._is_object = torch.empty(0) # int: *static* Dynamic object identifier, if 0 static background, if 1 object
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self._label,
            self._generation,
            self._is_object,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self._label,
        self._generation,
        self._is_object,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_scaling_wo_activation(self):
        return self._scaling
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_rotation_wo_activation(self):
        return self._rotation
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_label(self):
        return self._label
    
    @property
    def get_generation(self):
        return self._generation
    
    @property
    def get_is_object(self):
        return self._is_object
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def get_rotated_covariance(self, accum_R, which_object, during_training, scaling_modifier = 1):
        return self.covariance_activation_w_rot(self.get_scaling, scaling_modifier, self._rotation, accum_R, which_object, during_training)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # =====================================
    # | Training optimizer setup          |
    # =====================================
    def training_setup(self, training_args, reset_densification_stats=True):
        """
        Args:
            reset_densification_stats (bool): if True, reset densification stats (OG implementation)
        """
        self.percent_dense = training_args.percent_dense
        if reset_densification_stats:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._label], 'lr': 0.0, "name": "label"}, # Disable label learning at start
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
    def update_lr_for_label(self, label_lr):
        """Disable learning rate for all other variables except for label"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "label":
                param_group['lr'] = label_lr
            else:
                param_group['lr'] = 0.0

    def update_learning_rate(self, iteration):
        """
        OG 3DGS helper function for scheduling learning rate per step, DO NOT MODIFY
        """
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    # ===========================================================
    # | *Important* Replace / Concatenate tensors to optimizer  |
    # ===========================================================
    def replace_tensor_to_optimizer(self, tensor, name):
        """
        OG 3DGS helper function, DO NOT MODIFY
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        OG 3DGS helper function, DO NOT MODIFY
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # Only cat new tensor parameters to optimizer if not object poses
            if group["name"] in ("obj_translation", "obj_rotation_6d"):
                pass
            else:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # ============================================================================
    # | Functions for .ply file loading (either colmap or trained) and saving    |
    # ============================================================================
    def create_from_pcd(self, 
                        pcd : BasicPointCloud, 
                        spatial_lr_scale : float, 
                        rand_pts_init=None, 
                        rand_label_init=False):
        """
        Initializing parameters from Colmap output's ply
        Args:
            rand_pts_init (int or None): if None load from colmap ply output;
                else if not None and is int, override and initialize with random point clouds 
                with N=rand_pts_init pts and random color
            rand_label_init (bool): if False, initialize trainable self._label to a very small value;
                else if True, initialize at random in range (0, 1)
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = np.asarray(pcd.points)
        fused_color = np.asarray(pcd.colors)

        if rand_pts_init is not None:
            print("Initializing a random point clouds")
            try:
                assert isinstance(rand_pts_init, int)
                fused_point_cloud = np.random.uniform(fused_point_cloud.min(), fused_point_cloud.max(), (rand_pts_init, 3))
                fused_color = np.random.uniform(fused_color.min(), fused_color.max(), (rand_pts_init, 3))
            except:
                print("scene(...,rand_pts_init= ) should be an integer")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(fused_point_cloud).float().cuda()), 0.0000001)
        fused_point_cloud = torch.tensor(fused_point_cloud).float().cuda()
        fused_color = RGB2SH(torch.tensor(fused_color).float().cuda())

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        ### Newly added variables: trainable object identity & Gaussian generation label
        # labels = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda") # zero initialization
        if rand_label_init:
            labels = torch.rand(fused_point_cloud.shape[0], 1, dtype=torch.float, device="cuda")
        else:
            # else initialize label to a very small value
            labels = torch.ones(fused_point_cloud.shape[0], 1, dtype=torch.float, device="cuda") * 0.01

        generations = torch.zeros(fused_point_cloud.shape[0], 1, dtype=torch.int, device="cuda")
        is_objects = torch.zeros(fused_point_cloud.shape[0], 1, dtype=torch.int, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self._label = nn.Parameter(labels.requires_grad_(True))
        self._generation = generations # Not nn.Param
        self._is_object = is_objects # Not nn.Param

    def construct_list_of_attributes(self):
        """
        A helper function for saving .ply
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        # Newly added variables
        l.append('label')
        l.append('generation')
        l.append('is_object')
        return l

    def construct_list_of_attributes_og(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        # New added variables
        labels = self._label.detach().cpu().numpy()
        generations = self._generation.detach().cpu().numpy()
        is_objects = self._is_object.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, 
                                     labels, generations, is_objects), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, train_params=True, is_object=False, force_bg=False):
        """
        Function for loading from trained gaussians from a .ply file
        Args:
            train_params: if True, then require grad on all Gaussians' trainable variables
            is_object: if True, the loaded ply corresponds to a dynamic object and all self._is_object set to 1
        """
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # Load additional variables if exists in ply file
        if "label" in plydata.elements[0].data.dtype.names:
            labels = np.asarray(plydata.elements[0]["label"])[..., np.newaxis]
        else: # if not exist, initialize label to a very small value, i.e. 0.01
            labels = np.ones((xyz.shape[0], 1), dtype=float) * 0.01

        if "generation" in plydata.elements[0].data.dtype.names:    
            generations = np.asarray(plydata.elements[0]["generation"], dtype=int)[..., np.newaxis]
        else: 
            generations = np.zeros((xyz.shape[0], 1), dtype=int)
        
        if "is_object" in plydata.elements[0].data.dtype.names:
            is_object_arr = np.asarray(plydata.elements[0]["is_object"], dtype=int)[..., np.newaxis]
        else:
            if is_object: # if the loaded ply corresponds to a dynamic object, set to 1
                is_object_arr = np.ones((xyz.shape[0], 1), dtype=int)
            else: # else it corresponds to static background, set to 0
                is_object_arr = np.zeros((xyz.shape[0], 1), dtype=int)
        if force_bg:
            is_object_arr = np.zeros((xyz.shape[0], 1), dtype=int)
        
        self._xyz = torch.tensor(xyz, dtype=torch.float, device="cuda")
        self._features_dc = torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._features_rest = torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous()
        self._opacity = torch.tensor(opacities, dtype=torch.float, device="cuda")
        self._scaling = torch.tensor(scales, dtype=torch.float, device="cuda")
        self._rotation = torch.tensor(rots, dtype=torch.float, device="cuda")
        self._label = torch.tensor(labels, dtype=torch.float, device="cuda")

        if train_params:
            self._xyz = nn.Parameter(self._xyz.requires_grad_(True))
            self._features_dc = nn.Parameter(self._features_dc.requires_grad_(True))
            self._features_rest = nn.Parameter(self._features_rest.requires_grad_(True))
            self._opacity = nn.Parameter(self._opacity.requires_grad_(True))
            self._scaling = nn.Parameter(self._scaling.requires_grad_(True))
            self._rotation = nn.Parameter(self._rotation.requires_grad_(True))
            self._label = nn.Parameter(self._label.requires_grad_(True))

        self._generation = torch.tensor(generations, dtype=torch.int, device="cuda")
        self._is_object = torch.tensor(is_object_arr, dtype=torch.int, device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    # ==========================================================================
    # | Helper Functions for setting/resetting opacity (for removing floaters) |
    # ==========================================================================
    def reset_opacity(self):
        """
        OG 3DGS helper function for reseting opacity, DO NOT MODIFY
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_for_object(self, which_object=None):
        """
        Reset the opacity only for specific object
        """
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        if which_object is not None:
            opacities_new = torch.where(self.get_is_object != which_object, self._opacity, opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    # ===================================================
    # | Functions for densifying and pruning points     |
    # ===================================================
    def _prune_optimizer(self, mask):
        """
        Go over the optimizer and prune points by masks
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # Only prune paraemters from optimizer if not object poses
            if group["name"] in ("obj_translation", "obj_rotation_6d"):
                pass
            else:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask, during_training=True):
        valid_points_mask = ~mask

        if during_training:
            optimizable_tensors = self._prune_optimizer(valid_points_mask)

            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._label = optimizable_tensors["label"] # trainable object identity label
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]
        else:
            self._xyz = self._xyz[valid_points_mask]
            self._features_dc = self._features_dc[valid_points_mask]
            self._features_rest = self._features_rest[valid_points_mask]
            self._opacity = self._opacity[valid_points_mask]
            self._scaling = self._scaling[valid_points_mask]
            self._rotation = self._rotation[valid_points_mask]
            self._label = self._label[valid_points_mask]

        self._generation = self._generation[valid_points_mask]
        self._is_object = self._is_object[valid_points_mask]

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, 
                              new_opacities, new_scaling, new_rotation, 
                              new_label, new_generation, new_is_object):
        d = {"xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling" : new_scaling,
            "rotation" : new_rotation,
            "label" : new_label}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._label = optimizable_tensors["label"] # trainable object identity label

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # Directly cat new static variables to existing one instead of passing to optimizer
        self._generation = torch.cat((self.get_generation, new_generation), dim=0)
        self._is_object = torch.cat((self.get_is_object, new_is_object), dim=0)
        
        
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2,
                          curr_gen=None, 
                          split_prev_gen=True, 
                          which_object=None):
        """
        Args:
            curr_gen (int or None): which generation the newly added points in this phase will be,
                if None, then copy the generation of the parent point
            split_prev_gen (bool): if False, then split only the newly generated pts in this phase;
                else if True, split all points regardless of its generation
            which_object (int or None): if not None, split only the points that belong to the object
        """
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if not split_prev_gen: # split only current generation
            gen_mask = (self.get_generation == curr_gen).squeeze()
            selected_pts_mask = torch.logical_and(selected_pts_mask, gen_mask)

        if which_object is not None: # split only the gaussians corresponding to the object
            obj_mask = (self.get_is_object == which_object).squeeze()
            selected_pts_mask = torch.logical_and(selected_pts_mask, obj_mask)

        N = 2 # always split into 2
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_label = self._label[selected_pts_mask].repeat(N,1)
        # Assign a specific int to the generation of new points
        if curr_gen is not None and isinstance(curr_gen, int):
            new_generation = torch.full(self._generation[selected_pts_mask].repeat(N,1).shape, 
                                        curr_gen, dtype=torch.int, device="cuda")
        elif curr_gen is None: # if None will just copy
            new_generation = self._generation[selected_pts_mask].repeat(N,1)
        else:
            print("'curr_gen' variable should either be None or int")
        new_is_object = self._is_object[selected_pts_mask].repeat(N,1)
        
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacity, new_scaling, new_rotation, 
                                   new_label, new_generation, new_is_object)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, 
                          curr_gen=None, 
                          which_object=None):
        """
        Args:
            curr_gen (int or None): which generation the newly added points in this phase will be,
                if None, then copy the generation of the parent point
            which_object (int or None): if not None, clone only the points that belong to the object
        """
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if which_object is not None: # clone only the gaussians corresponding to the object
            obj_mask = (self.get_is_object == which_object).squeeze()
            selected_pts_mask = torch.logical_and(selected_pts_mask, obj_mask)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_label = self._label[selected_pts_mask]
        # Assign a specific int to the generation of new points
        if curr_gen is not None and isinstance(curr_gen, int):
            new_generation = torch.full(self._label[selected_pts_mask].shape, curr_gen, dtype=torch.int, device="cuda")
        elif curr_gen is None: # if None will just copy
            new_generation = self._generation[selected_pts_mask]
        new_is_object = self._is_object[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, 
                                   new_opacities, new_scaling, new_rotation, 
                                   new_label, new_generation, new_is_object)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, 
                          clone=True, split=True, curr_gen=None,
                          prune_prev_gen=True, split_prev_gen=True,
                          which_object=None):
        """
        Args:
            curr_gen: a variable assigned to each newly generated Gaussians
            prune_prev_gen (bool): if False, then prune only the points generated in this phase;
                else if True, prune all points regardless of it gen.
            split_prev_gen (bool): if False, then split only the newly generated pts in this phase;
                else if True, split all points regardless of its generation
            which_object (int or None): if not None, split and clone only the points that belong to the object
        """
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        if clone:
            self.densify_and_clone(grads, max_grad, extent, curr_gen, which_object=which_object)
        if split:
            self.densify_and_split(grads, max_grad, extent, curr_gen, split_prev_gen=split_prev_gen, which_object=which_object)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if not prune_prev_gen: # if not prune previous generation, then only prune the current gen
            gen_mask = (self.get_generation == curr_gen).squeeze()
            prune_mask = torch.logical_and(prune_mask, gen_mask)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def only_prune(self, min_opacity, extent, max_screen_size):
        """
        a test function to see if pruning will remove the black points during fine-tuning objects
        """
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        big_points_vs = self.max_radii2D > max_screen_size
        big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        is_bg = (self.get_is_object == 0).squeeze()
        prune_mask = torch.logical_and(prune_mask, is_bg) # only prune the background points
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def only_clone(self, max_grad, min_opacity, extent, which_object):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        print("How many points to be cloned:", torch.sum(torch.where(torch.norm(grads, dim=-1) >= max_grad, True, False)).item())
        # Clone
        self.densify_and_clone(grads, max_grad, extent, which_object=which_object)
        # Prune low-opacity
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        # self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        OG 3DGS helper function for densification, DO NOT MODIFY
        """
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # =================================================================
    # | Functions for modelling translation & rotation of objects     |
    # =================================================================
    @property
    def get_obj_trans_rot(self):
        return self.trainable_object_move.capture()
        # return {'translation': self._obj_translation, 'rotation': rot6d_to_matrix(self._obj_rotation_6d)}

    def init_obj_pose(self):
        """
        Initialize trainable translation & rotation for objects and return the current values
        """
        self.trainable_object_move = ObjectMove()
        self.trainable_object_move.to("cuda")

    def get_obj_pose_grad(self):
        trans_grad = self.trainable_object_move.obj_translation.grad.norm().item()
        rot_grad = self.trainable_object_move.obj_rotation_6d.grad.norm().item()
        return (trans_grad, rot_grad)

    def train_fine_all_setup(self, training_args, divide_3dgs_lr_by=1.0):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr / divide_3dgs_lr_by, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 / divide_3dgs_lr_by, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr / divide_3dgs_lr_by, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr / divide_3dgs_lr_by, "name": "rotation"},
            {'params': [self._label], 'lr': 0.0, "name": "label"}, # Disable label learning at start
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def train_fine_obj_setup(self, training_args, divide_pose_lr_by = 100.0, divide_3dgs_lr_by = 5.0):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        try:
            _ = self.trainable_object_move
        except AttributeError:
            print("Object pose hasn't been defined")

        l = [
            # {'params': [self.trainable_object_move.obj_translation], 'lr': training_args.obj_translation_lr / divide_pose_lr_by, "name": "obj_translation"},
            # {'params': [self.trainable_object_move.obj_rotation_6d], 'lr': training_args.obj_rotation_lr / divide_pose_lr_by, "name": "obj_rotation_6d"},
            {'params': [self.trainable_object_move.obj_translation], 'lr': 0.0, "name": "obj_translation"},
            {'params': [self.trainable_object_move.obj_rotation_6d], 'lr': 0.0, "name": "obj_rotation_6d"},
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr / divide_3dgs_lr_by, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 / divide_3dgs_lr_by, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr / divide_3dgs_lr_by, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr / divide_3dgs_lr_by, "name": "rotation"},
            {'params': [self._label], 'lr': 0.0, "name": "label"}, # Disable label learning at start
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def train_coarse_obj_setup(self, training_args, divide_pose_lr_by = 1.0, divide_3dgs_lr_by = 10.0):
        """
        Helper function for setting up training for coarse object pose estimation
        """
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        try:
            _ = self.trainable_object_move
        except AttributeError:
            print("Object pose hasn't been defined")
        l = [
            {'params': [self.trainable_object_move.obj_translation], 'lr': training_args.obj_translation_lr / divide_pose_lr_by, "name": "obj_translation"},
            {'params': [self.trainable_object_move.obj_rotation_6d], 'lr': training_args.obj_rotation_lr / divide_pose_lr_by, "name": "obj_rotation_6d"},
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr / divide_3dgs_lr_by, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0 / divide_3dgs_lr_by, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"}, # disable the decrease on opacity learning
            {'params': [self._scaling], 'lr': training_args.scaling_lr / divide_3dgs_lr_by, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr / divide_3dgs_lr_by, "name": "rotation"},
            {'params': [self._label], 'lr': 0.0, "name": "label"}, # Disable label learning at start
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale / divide_3dgs_lr_by,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    # ============================================================================
    # | Functions for zeroing/restoring the learning rates for pose / gaussians  |
    # ============================================================================
    def get_xyz_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                return param_group['lr']
    
    def load_xyz_lr(self, new_xyz_lr):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = new_xyz_lr

    def zero_pose_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "obj_translation":
                param_group['lr'] = 0.0
            elif param_group["name"] == "obj_rotation_6d":
                param_group['lr'] = 0.0

    def zero_gaussians_lr(self):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] != "obj_translation" and param_group["name"] != "obj_rotation_6d":
                param_group['lr'] = 0.0

    def restore_pose_lr(self, training_args):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "obj_translation":
                param_group['lr'] = training_args.obj_translation_lr
            elif param_group["name"] == "obj_rotation_6d":
                param_group['lr'] = training_args.obj_rotation_lr
    
    def get_lrs(self):
        lr_dict = {}
        for param_group in self.optimizer.param_groups:
            lr_dict[param_group["name"]] = param_group['lr']
        return lr_dict
    
    def load_lrs(self, lr_dict):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_dict[param_group["name"]]

    # ==================================================================================
    # | Functions for forward / backward transformation of Gaussians' xyz coordinates  |
    # ==================================================================================

    def apply_trans_rot(self, seq_transform, image_name, which_object=None, during_training=False, rotate_cov=False):
        """
        Update Gaussian's xyz to a new one with a sequence of translation & rotation until a specific frame
        Args:
            seq_transform: a sequence of transformation in dictionary format 
                {image_name: {"translation": tensor in shape (3), "rotation": tensor in shape (3x3)}} 
            image_name (str): one specific key to seq_transform
            which_object (int or None): if None, apply trans&rot to all gaussians pts
                if int, apply only to gaussians with is_object equal to this value
            during_training (bool): if True, then apply trainable translation & rotation at that frame,
                which is the last pair of translation & rotation applicable to the frame;
                also return the initial trans&rot values before update if during_training
        Returns:
            og_xyz: a clone of self._xyz for restoring purpose
        """
        # if image_name not in seq_transform.keys(): # then the previous frame in the static initialization, no need to move
        #     return
        if int(image_name) < int(list(seq_transform)[0]):
            return

        for key,value in seq_transform.items():
            if int(key) > int(image_name):
                break
            # for Debug
            if value is None or value.get("translation") is None or value.get("rotation") is None:
                continue 
            if which_object is not None and isinstance(which_object, int):
                if during_training and int(key) == int(image_name):
                    self._xyz = torch.where(self.get_is_object == which_object, self.trainable_object_move(self._xyz), self._xyz)
                    if rotate_cov:
                        self._rotation = torch.where(self.get_is_object == which_object, self.trainable_object_move(self._rotation), self._rotation)
                else:
                    trans = value["translation"].to("cuda")
                    rot = value["rotation"].to("cuda")
                    self._xyz = torch.where(self.get_is_object == which_object, object_move(self._xyz, trans, rot), self._xyz)
                    if rotate_cov:
                        self._rotation = torch.where(self.get_is_object == which_object, object_move(self._rotation, trans, rot), self._rotation)
            elif which_object is None: # apply to all gaussians
                if during_training and int(key) == int(image_name):
                    self._xyz = self.trainable_object_move(self._xyz)
                    if rotate_cov:
                        self._rotation = self.trainable_object_move(self._rotation)
                else:
                    self._xyz = object_move(self._xyz, value["translation"].to("cuda"), value["rotation"].to("cuda"))
                    if rotate_cov:
                        self._rotation = object_move(self._rotation, value["translation"].to("cuda"), value["rotation"].to("cuda"))
            else:
                print("'which_object' variable should either be None or int")
            if int(key) == int(image_name): # iterate through the sequence until the specific frame is reached
                break
        if during_training:
            return self.trainable_object_move.capture()
        else:
            return
    
    def apply_trans_rot_new(self, accum_T_seq, accum_R_seq, image_name, which_object=None, during_training=False):
        """
        New functions for applying transformations to xyz coordinates
        """
        accum_T_seq = {key: accum_T_seq[key] for key in sorted(accum_T_seq)}
        if which_object is None:
            is_object = torch.ones_like(self.get_is_object, dtype=torch.bool)
        else:
            is_object = (self.get_is_object == which_object)

        if int(image_name) < int(list(accum_T_seq)[0]):
            # Return Option 1: without applying any T
            return (None, None, None) # trainable, fixed T, R
        if during_training: 
            # assert accum_T_seq[image_name] is None, "during coarse training, the pose at targeted frame is initialized to None"
            prev_key, prev_accum_T = None, torch.eye(4)
            prev_accum_R = torch.eye(3)
            for key, accum_T in accum_T_seq.items():
                if int(key) >= int(image_name):
                    self._xyz = torch.where(is_object, self.trainable_object_move(apply_T_xyz(prev_accum_T, self._xyz)), self._xyz)
                    # Return Option 2: during training, both trainable and fixed
                    return (self.trainable_object_move.capture(), prev_accum_T, prev_accum_R) # trainable, fixed
                else:
                    prev_key, prev_accum_T = key, accum_T
                    prev_accum_R = accum_R_seq[key]
        else: # apply a fixed transformation T
            prev_key, prev_accum_T = None, torch.eye(4)
            prev_accum_R = torch.eye(3)
            for key, accum_T in accum_T_seq.items():
                if int(key) >= int(image_name):
                    if int(key) == int(image_name):
                        self._xyz = torch.where(is_object, apply_T_xyz(accum_T_seq[key], self._xyz), self._xyz)
                        # CONSOLE.log(f"New Apply Fixed @ {key} with {accum_T}")
                        return (None, accum_T_seq[key], accum_R_seq[key]) # trainable, fixed T, R
                    else: 
                        assert int(prev_key) <= int(image_name)
                        self._xyz = torch.where(is_object, apply_T_xyz(prev_accum_T, self._xyz), self._xyz)
                        # CONSOLE.log(f"New Apply Fixed @ {prev_key} <= {image_name} with {prev_accum_T}")
                        return (None, prev_accum_T, prev_accum_R) # trainable, fixed T, R
                else: # keep track of previous entry for rewind
                    prev_key, prev_accum_T = key, accum_T
                    prev_accum_R = accum_R_seq[key]
            # if the targeted frame is beyond the scope of saved sequence then apply the last entry
            key, accum_T = list(accum_T_seq.items())[-1]
            accum_R = accum_R_seq[key]
            self._xyz = torch.where(is_object, apply_T_xyz(accum_T, self._xyz), self._xyz)
            # CONSOLE.log(f"New Apply the last Fixed @ {key}")
            return (None, accum_T, accum_R) # trainable, fixed T, R

    def reverse_trans_rot(self, seq_transform, image_name, which_object=None, use_inverse=True, replace_to_optimizer=False, rotate_cov=True):
        new_xyz = self.get_xyz.clone()
        if rotate_cov:
            new_rotation = self._rotation.clone() 
        start_reversing = False
        if int(image_name) < int(list(seq_transform)[0]): # if the static phase after the transformation, change nothing
            return
        if int(image_name) > int(list(seq_transform)[-1]): # if the static phase after the transformation, apply the whole sequence
            start_reversing = True

        for key, value in reversed(list(seq_transform.items())):
            if int(key) == int(image_name):
                start_reversing = True
            elif int(key) < int(image_name):
                start_reversing = True
            if start_reversing:
                trans = value["translation"].to("cuda")
                rot = value["rotation"].to("cuda")
                
                if which_object is not None and isinstance(which_object, int):
                    new_xyz = torch.where(self.get_is_object == which_object, reverse_object_move(new_xyz, trans, rot, use_inverse), new_xyz)
                    if rotate_cov:
                        new_rotation = torch.where(self.get_is_object == which_object, reverse_object_move(new_rotation, trans, rot, use_inverse), new_rotation)
                elif which_object is None:
                    new_xyz = reverse_object_move(new_xyz, trans, rot, use_inverse)
                    if rotate_cov:
                        new_rotation = reverse_object_move(new_rotation, trans, rot, use_inverse)
                else:
                    print("'which_object' variable should either be None or int")
        if replace_to_optimizer:
            try: # for debugging, stored_state["exp_avg"] = torch.zeros_like(tensor), TypeError: 'NoneType' object does not support item assignment
                optimizable_tensors = self.replace_tensor_to_optimizer(new_xyz, "xyz")
                self._xyz = optimizable_tensors["xyz"]
            except TypeError:
                print(f"Stored_state hasn't been initialized when handling frame {image_name}")
                self._xyz = new_xyz
            if rotate_cov:
                try: 
                    optimizable_tensors = self.replace_tensor_to_optimizer(new_rotation, "rotation")
                    self._rotation = optimizable_tensors["rotation"]
                except TypeError:
                    print(f"Stored_state hasn't been initialized when handling frame {image_name}")
                    self._rotation = new_rotation
        else:
            self._xyz = new_xyz
            if rotate_cov:
                self._rotation = new_rotation
        torch.cuda.empty_cache()

    def reverse_trans_rot_new(self, which_object=None, trainable_t_R=None, fixed_T=None, replace_to_optimizer=True):
        new_xyz = self.get_xyz.clone()
        if which_object is None:
            is_object = torch.ones_like(self.get_is_object, dtype=torch.bool)
        else:
            is_object = (self.get_is_object == which_object)

        if trainable_t_R is not None:
            trans = trainable_t_R[0].to(new_xyz.device)
            rot = trainable_t_R[1].to(new_xyz.device)
            assert trans.shape == (3,) and rot.shape == (3,3), "captured trainable not in correct shape"
            new_xyz = torch.where(is_object, reverse_object_move(new_xyz, trans, rot, use_inverse=True), new_xyz)
        if fixed_T is not None:
            assert fixed_T.shape == (4, 4)
            new_xyz = torch.where(is_object, reverse_T_xyz(fixed_T.to(new_xyz.device), new_xyz), new_xyz)
        if replace_to_optimizer:
            try: # for debugging, stored_state["exp_avg"] = torch.zeros_like(tensor), TypeError: 'NoneType' object does not support item assignment
                optimizable_tensors = self.replace_tensor_to_optimizer(new_xyz, "xyz")
                self._xyz = optimizable_tensors["xyz"]
            except TypeError:
                self._xyz = new_xyz
        else:
            self._xyz = new_xyz
        torch.cuda.empty_cache()

    def restore_og_xyz(self, og_xyz):
        """
        a helper function restoring gaussian's xyz with og_xyz by directly replacing
        """
        optimizable_tensors = self.replace_tensor_to_optimizer(og_xyz, "xyz")
        self._xyz = optimizable_tensors["xyz"]

    # ========================================================================
    # | Functions for combining two sets of Gaussians (background & object)  |
    # ========================================================================
    def combine_gaussians(self, gaussians, train_params=True):
        """
        Function for joining the current Gaussians with another one
        by directly concatenate variables, only used *BEFORE* training and init optimizer
        """
        # Simplified version of variable concatenation
        with torch.no_grad():
            attributes = ['_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation', '_label']
            for attr in attributes:
                setattr(self, attr, torch.cat((getattr(self, attr).detach(), getattr(gaussians, attr).detach()), dim=0))
            if train_params:
                for attr in attributes:
                    setattr(self, attr, nn.Parameter(getattr(self, attr).requires_grad_(True)))

        self._generation = torch.cat((self._generation, gaussians._generation), dim=0)
        self._is_object = torch.cat((self._is_object, gaussians._is_object), dim=0)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        torch.cuda.empty_cache()

    def make_it_a_point(self, color):
        with torch.no_grad():
            mean_attributes = ['_xyz', '_rotation', '_label']
            for attr in mean_attributes:
                setattr(self, attr, torch.mean(getattr(self, attr).detach(), dim=0, keepdim=True))
            max_attributes = ['_features_dc', '_features_rest']
            for attr in max_attributes:
                setattr(self, attr, torch.max(getattr(self, attr).detach(), dim=0, keepdim=True).values)
            if color == "red":
                new_values = [[[3, -1, -1]]]
            elif color == "blue":
                new_values = [[[-1, -1, 3]]]

            self._features_dc = torch.tensor(new_values).to(dtype=self._features_dc.dtype, device="cuda")
            
            self._opacity = torch.max(self._opacity, dim=0, keepdim=True).values

            self._scaling = torch.mean(self._scaling, dim=0, keepdim=True)
            self._scaling = torch.full_like(self._scaling, -2.5,  device="cuda") # -3.2 -2 is a sweet value

            self._generation = torch.max(self._generation, dim=0, keepdim=True).values.to(torch.int32)
            self._is_object = torch.max(self._is_object, dim=0, keepdim=True).values.to(torch.int32)
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            torch.cuda.empty_cache()

    def infer_is_object_from_label(self):
        """
        a helper function for setting is_object based on trained label
        """
        self._is_object = torch.where(self.get_label > 0.5, torch.ones_like(self.get_label, dtype=torch.int), 
                                      torch.zeros_like(self.get_label, dtype=torch.int)) # 1 for object, 0 for background