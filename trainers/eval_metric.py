import os
import gc
import copy
from copy import deepcopy
import torchvision
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tf

from lpipsPyTorch import lpips
from scene import Scene, GaussianModel
from utils.console import CONSOLE
from utils.dynamic_utils import *
from utils.image_utils import psnr
from utils.loss_utils import ssim
from utils.geometry_utils import get_accum_T_seq, get_accum_R_seq

def process_viewer_cam(sample_cam, new_pose):
    assert len(new_pose) == 16, "new input camera pose has to be length 16"
    rot_3x4 = np.array(new_pose[:12]).reshape(3, 4)
    rot_3x3 = rot_3x4[:, :-1]
    trans = new_pose[-4:-1]
    new_cam = copy.deepcopy(sample_cam)
    new_cam.R = rot_3x3
    new_cam.T = trans
    return new_cam

def reconstruct_new_pose(new_cam):
    rot_3x4 = np.hstack((new_cam.R, np.zeros((3, 1))))
    rot_3x4_flat = rot_3x4.flatten().tolist()
    trans = new_cam.T.flatten().tolist()
    new_pose = rot_3x4_flat + trans + [0.0]  
    return new_pose

# ======================================================
# | Standard evaluation and metric calculation for NVS |
# ======================================================

def render_results(dataset, opt, pipe, exp_name, save_dir, 
    obj_pose_seq_path, all_gaussians_path,
    train_eval_split):

    ### Directories stuff ###
    os.makedirs(save_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    train_frames = train_eval_split["training_frames"]
    dynamic_eval_frames = train_eval_split["dynamic_eval_frames"]
    static_eval_frames = train_eval_split["static_eval_frames"]
    train_frames = [int(image_name) for image_name in train_frames]
    dynamic_eval_frames = [int(image_name) for image_name in dynamic_eval_frames]
    static_eval_frames = [int(image_name) for image_name in static_eval_frames]

    train_dir = os.path.join(save_dir, "training")
    os.makedirs(train_dir, exist_ok=True)
    dynamic_eval_dir = os.path.join(save_dir, "dynamic_eval")
    os.makedirs(dynamic_eval_dir, exist_ok=True)
    static_eval_dir = os.path.join(save_dir, "static_eval")
    os.makedirs(static_eval_dir, exist_ok=True)
    hand_dir = os.path.join(save_dir, "hand")
    os.makedirs(hand_dir, exist_ok=True)

    for dir in [train_dir, dynamic_eval_dir, static_eval_dir]:
        os.makedirs(os.path.join(dir, 'gt'), exist_ok=True)
        os.makedirs(os.path.join(dir, 'render'), exist_ok=True)

    CONSOLE.print(f"EXP_NAME {exp_name} will be saved in {save_dir}")
    obj_pose_sequence = torch.load(obj_pose_seq_path)
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)
    CONSOLE.log(f"Loaded object poses from {obj_pose_seq_path}")
    # Load the object and background Gaussians
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(all_gaussians_path, train_params=False)

    CONSOLE.log(f"Loaded Gaussians from {all_gaussians_path}")

    scene = Scene(dataset, 
                gaussians, 
                shuffle=False, 
                load_or_create_from=False, 
                load_hand_masks=True, 
                load_obj_masks=True)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy = deepcopy(gaussians)
            og_xyz = gaussians_copy.get_xyz
            gt_image = viewpoint_cam.gt_image
            hand_mask = viewpoint_cam.hand_mask
            # gt_image = gt_image * viewpoint_cam.obj_mask
            trainable_t_R, fixed_T, fixed_R = gaussians_copy.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
                image_name = viewpoint_cam.image_name, 
                which_object = 1, 
                during_training = False)
            render_img_pkg = render(viewpoint_cam, gaussians_copy, pipe, background, 
                rot_cov=True, accum_R=fixed_R, which_object=1, during_training=False)
            render_image = render_img_pkg["render"]
            gaussians_copy._xyz = og_xyz

        if int(viewpoint_cam.image_name) in train_frames:
            torchvision.utils.save_image(render_image, os.path.join(train_dir, 'render', f"{viewpoint_cam.image_name}.png"))
            torchvision.utils.save_image(gt_image, os.path.join(train_dir, "gt", f"{viewpoint_cam.image_name}.png"))
        elif int(viewpoint_cam.image_name) in dynamic_eval_frames:
            torchvision.utils.save_image(render_image, os.path.join(dynamic_eval_dir, 'render', f"{viewpoint_cam.image_name}.png"))
            torchvision.utils.save_image(gt_image, os.path.join(dynamic_eval_dir, "gt", f"{viewpoint_cam.image_name}.png"))
        elif int(viewpoint_cam.image_name) in static_eval_frames:
            torchvision.utils.save_image(render_image, os.path.join(static_eval_dir, 'render', f"{viewpoint_cam.image_name}.png"))
            torchvision.utils.save_image(gt_image, os.path.join(static_eval_dir, "gt", f"{viewpoint_cam.image_name}.png"))

        # Save the rendered evaluation image --at low resolution
        eval_image = get_eval_img([gt_image, render_image], [f"GT image {viewpoint_cam.image_name}", "Render image"])
        eval_image.save(os.path.join(eval_dir, f"{viewpoint_cam.image_name}.jpg"))
        # Save the hand mask
        torchvision.utils.save_image(1 - hand_mask, os.path.join(hand_dir, f"{viewpoint_cam.image_name}.png")) 
    CONSOLE.log(f"Saved rendered images to {save_dir}")

def calculate_metric(dataset, save_dir, train_eval_split):
    file = open(os.path.join(save_dir, 'results.txt'), 'w')

    for split_type in ['dynamic_eval', 'static_eval']: # ['training']
        CONSOLE.log(f"Calculating metrics of {split_type}")
        file.write(split_type + '\n')

        renders_dir = os.path.join(save_dir, split_type, 'render')
        gts_dir = os.path.join(save_dir, split_type, 'gt')
        hands_dir = os.path.join(save_dir, 'hand')
        renders = []
        gts = []
        hands = []
        image_names = []
        for fname in os.listdir(renders_dir):
            render = Image.open(os.path.join(renders_dir, fname))
            gt = Image.open(os.path.join(gts_dir, fname))
            hand = Image.open(os.path.join(hands_dir, fname))
            renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
            gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
            hands.append(tf.to_tensor(hand).unsqueeze(0)[:, :3, :, :].cuda())
            image_names.append(fname)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            image_name = image_names[idx]
            
            ssims.append(ssim(renders[idx] * hands[idx], gts[idx] * hands[idx]))
            psnrs.append(psnr(renders[idx] * hands[idx], gts[idx] * hands[idx]))
            lpipss.append(lpips(renders[idx] * hands[idx], gts[idx] * hands[idx], net_type='vgg'))

            # ssims.append(ssim(renders[idx], gts[idx]))
            # psnrs.append(psnr(renders[idx], gts[idx]))
            # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

        print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
        file.write("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5") + '\n')
        print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
        file.write("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5") + '\n')
        print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
        file.write("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5") + '\n')

        CONSOLE.log(f"Finished calculating metrics of {split_type}")
    file.close()

# =============================================
# | For visualization demonstrated in webpage |
# =============================================

def render_singleview_w_new_pose(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, new_pose):
    vis_dir = os.path.join(save_dir, "single_w_new_pose")
    os.makedirs(vis_dir, exist_ok=True)
    obj_pose_sequence = torch.load(obj_pose_seq_path)
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(all_gaussians_path, train_params=False)
    gaussians_copy = deepcopy(gaussians)
    scene = Scene(dataset, 
                gaussians, 
                shuffle=False, 
                load_or_create_from=False, 
                load_hand_masks=True, 
                load_obj_masks=True)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()
    new_cam = process_viewer_cam(viewpoints_og[0], new_pose)
    new_cam.reprocess_cam()
    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        progress_bar.update(1)
        og_xyz = gaussians_copy.get_xyz
        trainable_t_R, fixed_T, fixed_R = gaussians_copy.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
            image_name = viewpoint_cam.image_name, 
            which_object = 1, 
            during_training = False)
        render_img_pkg = render(new_cam, gaussians_copy, pipe, background, 
            rot_cov=True, accum_R=fixed_R, which_object=1, during_training=False)
        render_image = render_img_pkg["render"]
        gaussians_copy._xyz = og_xyz
        eval_image = rgb_tensor_to_PIL(render_image)
        eval_image.save(os.path.join(vis_dir, f"{viewpoint_cam.image_name}.png"))

def render_multiview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, new_poses=None):
    """Render multi-views visualization"""
    vis_dir = os.path.join(save_dir, "multi-view")
    os.makedirs(vis_dir, exist_ok=True)
    vis_dir = save_dir

    obj_pose_sequence = torch.load(obj_pose_seq_path)
    CONSOLE.log(f"Loaded object poses from {obj_pose_seq_path}")
    # Load the object and background Gaussians
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(all_gaussians_path, train_params=False)

    scene = Scene(dataset, 
                gaussians, 
                shuffle=False, 
                load_or_create_from=False, 
                load_hand_masks=True, 
                load_obj_masks=True)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    if new_poses is not None:
        new_cams = []
        for new_pose in new_poses:
            new_cam = process_viewer_cam(viewpoints_og[0], new_pose)
            CONSOLE.log(f"New camera Rotation {new_cam.R} Translation {new_cam.T}")
            new_cam.reprocess_cam()
            new_cams.append(new_cam)

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy = deepcopy(gaussians)
            og_xyz = gaussians_copy.get_xyz
            gt_image = viewpoint_cam.gt_image
            hand_mask = viewpoint_cam.hand_mask

            gaussians_copy.apply_trans_rot(obj_pose_sequence, viewpoint_cam.image_name, 
                                            which_object=1, during_training=False, rotate_cov=False)
            
            render_img_pkg_cam1 = render(viewpoint_cam, gaussians_copy, pipe, background) # training cam
            render_image_cam1 = render_img_pkg_cam1["render"]
            # Cam 2-4 fixed viewpoints for evaluating novel view synthesis
            # render_image_cam2 = render(new_cam, gaussians_copy, pipe, background)["render"]
            # render_image_cam2 = render(viewpoints_og[100], gaussians_copy, pipe, background)["render"]
            render_image_cam3 = render(viewpoints_og[int(len(viewpoints_og) / 2)], gaussians_copy, pipe, background)["render"] # Middle camera manually adjusted
            
            if new_poses is not None:
                render_image_cams = []
                cam_names = []
                for i, new_cam in enumerate(new_cams):
                    render_image_cam = render(new_cam, gaussians_copy, pipe, background)["render"]
                    render_image_cams.append(render_image_cam)
                    # cam_names.append(f"Cam {i+1}")
                    cam_names.append("")

            gaussians_copy._xyz = og_xyz

            eval_image = get_eval_img_new([render_image_cam1] + render_image_cams, [""] + cam_names, rows=len(render_image_cams) + 1, cols= 1)
            eval_image.save(os.path.join(vis_dir, f"{viewpoint_cam.image_name}.png"))

def interpolate_lists(list1, list2, N):
    # For interpolation between two cameras
    array1 = np.array(list1)
    array2 = np.array(list2)
    interpolated_lists = []
    steps = np.linspace(0, 1, N+2)  
    for step in steps[1:-1]:
        interpolated_list = array1 + step * (array2 - array1)
        interpolated_lists.append(interpolated_list.tolist())
    
    return interpolated_lists

def render_freeiview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, new_poses=None):
    # assert len(new_poses) == 2 # First be far away Second be close
    vis_dir = os.path.join(save_dir, "freeview")
    os.makedirs(vis_dir, exist_ok=True)
    obj_pose_sequence = torch.load(obj_pose_seq_path)
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)
    CONSOLE.log(f"Loaded object poses from {obj_pose_seq_path}")
    # Load the object and background Gaussians
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(all_gaussians_path, train_params=False)

    scene = Scene(dataset, gaussians, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    if len(new_poses) == 2:
        print("Interpolate between two provided poses")
        interpolate_poses = interpolate_lists(new_poses[0], new_poses[1], len(viewpoints_og))
    elif len(new_poses) == 1:
        print("Interpolate between provided poses and one viewpoint cam")
        new_pose_1 = reconstruct_new_pose(viewpoints_og[223])
        interpolate_poses = interpolate_lists(new_poses[0], new_pose_1, len(viewpoints_og))

    new_cams = []
    for new_pose in interpolate_poses:
        new_cam = process_viewer_cam(viewpoints_og[0], new_pose)
        # CONSOLE.log(f"New camera Rotation {new_cam.R} Translation {new_cam.T}")
        new_cam.reprocess_cam()
        new_cams.append(new_cam)
    assert len(new_cams) == len(viewpoints_og)

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy = deepcopy(gaussians)
            og_xyz = gaussians_copy.get_xyz
            gt_image = viewpoint_cam.gt_image
            hand_mask = viewpoint_cam.hand_mask
            trainable_t_R, fixed_T, fixed_R = gaussians_copy.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
                image_name = viewpoint_cam.image_name, 
                which_object = 1, 
                during_training = False)
            render_img_pkg = render(new_cams[i], gaussians_copy, pipe, background, 
                rot_cov=True, accum_R=fixed_R, which_object=1, during_training=False)
            render_image_cam1 = render_img_pkg["render"]
            # Cam 2 fixed viewpoints for evaluating novel view synthesis
            # render_image_cam2 = render(viewpoints_og[144], gaussians_copy, pipe, background)["render"]
            # render_image_cam2 = render(viewpoints_og[int(len(viewpoints_og) / 2)], gaussians_copy, pipe, background)["render"] # Middle camera manually adjusted
            # render_image_cam3 = render(new_cams[0], gaussians_copy, pipe, background)["render"]
            # render_image_cam4 = render(new_cams[i], gaussians_copy, pipe, background)["render"]

            gaussians_copy._xyz = og_xyz
            eval_image = rgb_tensor_to_PIL(render_image_cam1)
            # eval_image = get_eval_img_new([gt_image, render_image_cam1, render_image_cam3], None, rows=1, cols=3)
            
            eval_image.save(os.path.join(vis_dir, f"{viewpoint_cam.image_name}.png"))

def render_singleview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path):
    """Render multi-views visualization"""
    vis_dir = os.path.join(save_dir, "singleview")
    os.makedirs(vis_dir, exist_ok=True)
    obj_pose_sequence = torch.load(obj_pose_seq_path)
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)
    CONSOLE.log(f"Loaded object poses from {obj_pose_seq_path}")
    # Load the object and background Gaussians
    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(all_gaussians_path, train_params=False)

    scene = Scene(dataset, 
                gaussians, 
                shuffle=False, 
                load_or_create_from=False, 
                load_hand_masks=True, 
                load_obj_masks=True)
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda") # torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy = deepcopy(gaussians)
            og_xyz = gaussians_copy.get_xyz
            gt_image = viewpoint_cam.gt_image
            hand_mask = viewpoint_cam.hand_mask

            trainable_t_R, fixed_T, fixed_R = gaussians_copy.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
                image_name = viewpoint_cam.image_name, 
                which_object = 1, 
                during_training = False)
            render_img_pkg = render(viewpoint_cam, gaussians_copy, pipe, background,
                rot_cov=True, accum_R=fixed_R, which_object=1, during_training=False)
            render_image = render_img_pkg["render"]
            gaussians_copy._xyz = og_xyz
            eval_image = rgb_tensor_to_PIL(render_image)
            eval_image.save(os.path.join(vis_dir, f"{viewpoint_cam.image_name}.png"))

def render_one_img(dataset, pipe, save_dir, gaussians_path, new_pose, img_name):
    os.makedirs(save_dir, exist_ok=True)
    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(gaussians_path, train_params=False)
    scene = Scene(dataset, gaussians, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()
    new_cam = process_viewer_cam(viewpoints_og[0], new_pose)
    new_cam.reprocess_cam()
    render_image_cam = render(new_cam, gaussians, pipe, background)["render"]
    eval_image = rgb_tensor_to_PIL(render_image_cam)
    eval_image.save(os.path.join(save_dir, f"{img_name}.png"))

def render_trajectory(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, obj_gaussians_path, new_pose=None):
    vis_dir = os.path.join(save_dir, "trajectory")
    os.makedirs(vis_dir, exist_ok=True)

    obj_pose_sequence = torch.load(obj_pose_seq_path)

    gaussians_obj = GaussianModel(sh_degree=0)
    gaussians_obj.load_ply(obj_gaussians_path, train_params=False, is_object=True)
    gaussians_obj.make_it_a_point(color="red")

    gaussians_bg = GaussianModel(sh_degree=0)
    gaussians_bg.load_ply(all_gaussians_path, train_params=False, force_bg=True)
    
    scene = Scene(dataset, gaussians_bg, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    if new_pose is not None:
        new_cam = process_viewer_cam(viewpoints_og[0], new_pose)
        new_cam.reprocess_cam()

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy = deepcopy(gaussians_obj)
            gaussians_copy.apply_trans_rot(obj_pose_sequence, viewpoint_cam.image_name, 
                                            which_object=1, during_training=False)
            if i > 50:
                gaussians_bg.combine_gaussians(gaussians_copy, train_params=False)
            render_img_pkg_cam1 = render(new_cam, gaussians_bg, pipe, background) 
            render_image_cam1 = render_img_pkg_cam1["render"]
            eval_image = rgb_tensor_to_PIL(render_image_cam1)
            eval_image.save(os.path.join(vis_dir, f"{viewpoint_cam.image_name}.png"))

    # gaussians_bg.save_ply(os.path.join(save_dir, "gaussians_trajectory.ply"))

def render_double_trajectory(dataset, opt, pipe, exp_name, save_dir, 
    obj_pose_seq_path1, obj_pose_seq_path2,
    all_gaussians_path, obj_gaussians_path):

    os.makedirs(save_dir, exist_ok=True)
    obj_pose_sequence1 = torch.load(obj_pose_seq_path1)
    obj_pose_sequence2 = torch.load(obj_pose_seq_path2)

    gaussians_obj1 = GaussianModel(sh_degree=0)
    gaussians_obj1.load_ply(obj_gaussians_path, train_params=False, is_object=True)
    gaussians_obj1.make_it_a_point(color="red")

    gaussians_obj2 = GaussianModel(sh_degree=0)
    gaussians_obj2.load_ply(obj_gaussians_path, train_params=False, is_object=True)
    gaussians_obj2.make_it_a_point(color="blue")

    gaussians_bg = GaussianModel(sh_degree=0)
    gaussians_bg.load_ply(all_gaussians_path, train_params=False, force_bg=True)
    
    scene = Scene(dataset, gaussians_bg, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") # torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()

    first_iter = 0
    progress_bar = tqdm(range(first_iter, len(viewpoints_og)), desc="Evaluation progress")
    first_iter += 1

    for i, viewpoint_cam in enumerate(viewpoints_og):
        with torch.no_grad():
            progress_bar.update(1)
            gaussians_copy1 = deepcopy(gaussians_obj1)
            gaussians_copy1.apply_trans_rot(obj_pose_sequence1, viewpoint_cam.image_name, 
                                            which_object=1, during_training=False, rotate_cov=False)
            gaussians_bg.combine_gaussians(gaussians_copy1, train_params=False)

            gaussians_copy2 = deepcopy(gaussians_obj2)
            gaussians_copy2.apply_trans_rot(obj_pose_sequence2, viewpoint_cam.image_name, 
                                            which_object=1, during_training=False, rotate_cov=False)
            gaussians_bg.combine_gaussians(gaussians_copy2, train_params=False)
    gaussians_bg.save_ply(os.path.join(save_dir, "double_trajectory.ply"))

def eval_and_metric(dataset, opt, pipe, exp_name, save_dir, 
    obj_pose_seq_path, all_gaussians_path, train_eval_split):

    # Standard evaluation and metric calculation for NVS (Paper Table 1)
    render_results(dataset, opt, pipe, exp_name, save_dir, obj_pose_seq_path, all_gaussians_path, train_eval_split)
    gc.collect()
    torch.cuda.empty_cache()
    calculate_metric(dataset, save_dir, train_eval_split)

    # For visuliazation demonstrated in webpage
    # render_multiview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, new_poses)
    # render_singleview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path)
    # render_freeiview(dataset, pipe, save_dir, obj_pose_seq_path, all_gaussians_path, new_poses)
