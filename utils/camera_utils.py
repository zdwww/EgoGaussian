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

from scene.cameras import Camera
import torch
import numpy as np
from PIL import Image
from utils.general_utils import PILtoTorch, binarize_mask, blur_cb
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution, normalize=True)
    gt_image = resized_image_rgb[:3, ...]

    # Note: for masks processed by colmap's image undistortion, it will convert 1-channel mask to 3-channel, hence a binarize method is required
    if cam_info.hand_mask is not None:
        hand_mask = binarize_mask(PILtoTorch(cam_info.hand_mask, resolution, normalize=True))
        assert gt_image.shape[1:] == hand_mask.shape[1:], "Hand mask resolution mismatch"
    else:
        # hand_mask = None
        raise ValueError("Hand mask always required in all stages.")
    
    if cam_info.obj_mask is not None:
        obj_mask = binarize_mask(PILtoTorch(cam_info.obj_mask, resolution, normalize=True))
        assert gt_image.shape[1:] == obj_mask.shape[1:], "Object mask resolution mismatch"
    else:
        obj_mask = None # object masks not required outside rewind interval
        # raise ValueError("Object mask, at least during interaction, always required.")
    
    if isinstance(cam_info.est_depth, Image.Image):
        est_depth = PILtoTorch(cam_info.est_depth, resolution, normalize=False)
        assert est_depth.shape[0] == 1, "Depth image should have only one channel"
        assert gt_image.shape[1:] == est_depth.shape[1:], "Estimated depth resolution mismatch"
    elif isinstance(cam_info.est_depth, np.ndarray):
        est_depth = np.expand_dims(cam_info.est_depth, axis=0)
        est_depth = torch.from_numpy(est_depth)
    else:
        est_depth = None

    if cam_info.pred_cb is not None:
        # pred_cb = blur_cb(PILtoTorch(cam_info.pred_cb, resolution, normalize=True))
        pred_cb = PILtoTorch(cam_info.pred_cb, resolution, normalize=True)
        assert torch.all((pred_cb == 0) | (pred_cb == 1)), "Mask tensor should have two unique values 0 and 1"
        assert gt_image.shape[1:] == pred_cb.shape[1:], "Predict contact boundary resolution mismatch"
    else:
        pred_cb = None

    loaded_mask = None
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  gt_image=gt_image,
                  image_name=cam_info.image_name, 
                  uid=id, 
                  data_device=args.data_device,
                  gt_alpha_mask=loaded_mask,
                  hand_mask=hand_mask,
                  obj_mask=obj_mask,
                  est_depth=est_depth,
                  pred_cb=pred_cb
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, cam_info in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, cam_info, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
