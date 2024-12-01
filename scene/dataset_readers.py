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
import re
import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.console import CONSOLE

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_name: str
    image: np.array
    width: int
    height: int
    hand_mask: np.array
    obj_mask : np.array
    est_depth : np.array
    pred_cb : np.array
    
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def load_img_feature(src_dir, extr_name, allow_npy=False):
    """
    Load processed image features (interaction masks, object masks, or depth)
    Args:
        src_dir (str): source directory
        extr_name (str): image name, extr.name
        allow_npy (bool): allow numpy format for depth
    """
    load_path = os.path.join(src_dir, os.path.basename(extr_name))
    # check if the file exists, if not, try to load the other format
    if not os.path.exists(load_path):
        if load_path.endswith('.jpg'):
            load_path = re.sub(r"\.jpg$", ".png", load_path)
        elif load_path.endswith('.png'):
            load_path = re.sub(r"\.png$", ".jpg", load_path)

    try: # Case 1: load as image
        with Image.open(load_path) as temp:
            feature = temp.copy()
    except:
        try: # Case 2: if allowed, load as numpy array
            if allow_npy:
                load_path = re.sub(r"\.jpg|\.png", ".npy", load_path)
                feature = np.load(load_path)
            else:
                raise FileNotFoundError("Image features should be in .jpg or .png format or set allow_npy")
        except:
            raise FileNotFoundError("Image features not found in directory!")
    return feature

def readColmapCameras(cam_extrinsics, 
                      cam_intrinsics, 
                      images_dir, 
                      hand_masks_dir,
                      obj_masks_dir,
                      est_depths_dir,
                      pred_cb_dir):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_dir, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
       
        with Image.open(image_path) as temp:
            image = temp.copy()

        hand_mask = None
        if hand_masks_dir is not None:
            hand_mask = load_img_feature(hand_masks_dir, extr.name)
        
        obj_mask = None
        if obj_masks_dir is not None:
            try:
                obj_mask = load_img_feature(obj_masks_dir, extr.name)
            except:
                print(f"Object masks not found for {extr.name}. Okay if in first static stage")

        est_depth = None
        if est_depths_dir is not None:
            est_depth = load_img_feature(est_depths_dir, extr.name, allow_npy=True)

        pred_cb = None
        if pred_cb_dir is not None:
            pred_cb = load_img_feature(pred_cb_dir, extr.name)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                              image_name=image_name, 
                              image=image, width=width, height=height,
                              hand_mask=hand_mask, obj_mask=obj_mask, est_depth=est_depth, pred_cb=pred_cb
                              )
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, 
                        images="images", 
                        eval=False, 
                        llffhold=8, 
                        load_hand_masks=True,
                        load_obj_masks=True, 
                        load_est_depths=False,
                        load_pred_cb=False):
    """
    Read the scene information from a COLMAP sparse reconstruction
    Args:
        images (str): path to the images folder
        
    """
    images = "images" if images is None else images
    images_dir = os.path.join(path, images)
    # List all images selected in the dataset for only reading necessary cameras_extrinsic
    image_names = sorted([os.path.splitext(file)[0] for file in os.listdir(images_dir) if file.endswith(('.jpg', '.png'))])
    image_type = os.path.splitext(os.listdir(images_dir)[0])[1] # jpg or png

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file, image_names, image_type)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file, image_names, image_type)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    assert len(image_names) == len(cam_extrinsics)

    # Check if load arguments are True, and set the corresponding directories accordingly.
    hand_masks_dir = None
    obj_masks_dir = None
    est_depths_dir = None
    pred_cb_dir = None

    if load_hand_masks:
        hand_masks_dir = os.path.join(path, "hand_masks")
        if not os.path.exists(hand_masks_dir):
            raise FileNotFoundError("All hand masks should be in the sub-dir 'hand_masks'!")
        
    if load_obj_masks:
        obj_masks_dir = os.path.join(path, "obj_masks")
        if not os.path.exists(obj_masks_dir):
            raise FileNotFoundError("All object masks should be in the sub-dir 'obj_masks'!")

    if load_est_depths:
        est_depths_dir = os.path.join(path, "est_depths")
        if not os.path.exists(est_depths_dir):
            raise FileNotFoundError("All estimated depths should be in the sub-dir 'est_depths'!")

    if load_pred_cb:
        pred_cb_dir = os.path.join(path, "pred_cb")
        if not os.path.exists(pred_cb_dir):
            raise FileNotFoundError("All predicted contact boundary should be in the sub-dir 'pred_cb'!")

    cam_infos_unsorted = readColmapCameras(cam_extrinsics, 
                                           cam_intrinsics, 
                                           images_dir, 
                                           hand_masks_dir, 
                                           obj_masks_dir, 
                                           est_depths_dir,
                                           pred_cb_dir)
                                           
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# Disable "Blender" : readNerfSyntheticInfo

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}