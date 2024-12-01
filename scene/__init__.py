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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.console import CONSOLE

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, 
                 gaussians : GaussianModel, 
                 load_iteration=None, 
                 shuffle=True, 
                 resolution_scales=[1.0], 
                 rand_pts_init=None, 
                 rand_label_init=False,
                 load_or_create_from=True,
                 load_hand_masks=True, 
                 load_obj_masks=True,
                 load_est_depths=False,
                 load_pred_cb=False):
        """
        Parameters:
        shuffle (bool): if True, shuffle viewpoint cams
        rand_pts_init (int or None): if not None, initialize with random point clouds with N pts; 
            else load from colmap ply output
        rand_label_init (bool): if True, initialize Gaussian's label (for object identity) at random,
            else set to very small value
        load_or_create_from: if True, either load trained model from .ply or create from colmap point clouds,
            else keep the original Gaussians
        load_hand_masks (bool): if True, load hand masks
        load_obj_masks (bool): if True, load object masks
        load_est_depth (bool): if True, load estimated depth
        load_pred_cb (bool): if True, load predicted contact boundary from EgoHOS
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # Load Colmap Scene
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, 
                                                          args.images, 
                                                          args.eval,
                                                          load_hand_masks=load_hand_masks,
                                                          load_obj_masks=load_obj_masks,
                                                          load_est_depths=load_est_depths,
                                                          load_pred_cb=load_pred_cb)
        # Disable the Blender option
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        # TODO LOADING CAMERAS
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, 
                                                                            resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, 
                                                                           resolution_scale, args)

        if load_or_create_from:
            if self.loaded_iter:
                CONSOLE.log("Loading from pre-trained 3DGS results")
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"))
            else:
                CONSOLE.log("Creating from Colmap output...")
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent, rand_pts_init,
                                            rand_label_init)
        else:
            self.gaussians.spatial_lr_scale = self.cameras_extent
            CONSOLE.log(f"Assume a gaussians set is already loaded, update spatial_lr_scale to {self.gaussians.spatial_lr_scale}")
            
        # Save some variables for reinitilization
        self.scene_info_point_cloud = scene_info.point_cloud
        self.rand_pts_init = rand_pts_init
        self.rand_label_init = rand_label_init
    
    def re_initialize(self, gaussians : GaussianModel):
        """
        Helper function to reinitialize the scene from an initial colmap output point cloud but keeps camera info
        """
        self.gaussians = gaussians
        self.gaussians.create_from_pcd(self.scene_info_point_cloud, self.cameras_extent, self.rand_pts_init,
                                       self.rand_label_init)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]