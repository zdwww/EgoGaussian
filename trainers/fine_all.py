import os
import random

import torch
import wandb
from tqdm import tqdm

from scene import Scene, GaussianModel
from utils.console import CONSOLE
from gaussian_renderer import render
from utils.dynamic_utils import get_viewpoint_split, get_eval_img
from utils.loss_utils import l1_loss, ssim
from utils.geometry_utils import get_accum_T_seq, get_accum_R_seq

def save_poses_securely(d, save_dir):
    filename=os.path.join(save_dir, 'obj_pose_sequence.pth')
    try:
        temp_filename = filename + ".tmp"
        torch.save(d, temp_filename)  # Save to a temporary file first
        os.replace(temp_filename, filename)  # Atomically replace the old file with the new
    except Exception as e:
        pass
    return filename

def fine_tune_all(dataset, opt, pipe, fine_p, exp_name, save_dir, 
    obj_gaussians_path, bg_gaussians_path, obj_pose_seq_path, 
    static_phases, dynamic_phases, train_frames=None):

    os.makedirs(save_dir, exist_ok=True)

    CONSOLE.print(f"Coarse stage of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print(f"Load Gaussian object model from {obj_gaussians_path}, bg from {bg_gaussians_path} obj pose from {obj_pose_seq_path}")
    CONSOLE.print("Static Phases:", static_phases)
    CONSOLE.print("Dynamic Phases:", dynamic_phases)
    CONSOLE.print("Total number of iterations:", fine_p.total_num_iter)

    obj_pose_sequence = torch.load(obj_pose_seq_path)
    CONSOLE.log(f"Load object poses from {obj_pose_seq_path}")
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)

    gaussians_obj = GaussianModel(sh_degree=0)
    gaussians_obj.load_ply(obj_gaussians_path, train_params=True, is_object=True)

    gaussians_bg = GaussianModel(sh_degree=0)
    gaussians_bg.load_ply(bg_gaussians_path, train_params=True, is_object=False)
    CONSOLE.log(f"Loaded Gaussians background from {bg_gaussians_path}")
    gaussians_obj.combine_gaussians(gaussians_bg, train_params=True)
    gaussians = gaussians_obj
    del gaussians_bg

    scene = Scene(dataset, gaussians, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)
    wandb.init(project=exp_name, name=f"fine-tune-all", dir="/scratch_net/biwidl301/daizhang/wandb")
    gaussians.train_fine_all_setup(opt, divide_3dgs_lr_by=1.0)

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()
    viewpoint_phases = get_viewpoint_split(viewpoints_og, train_frames, 
        static_phases=static_phases, dynamic_phases=dynamic_phases)

    static_vpts_list = [(static_vpt, 'static') for static_viewpoints in viewpoint_phases["static_phases"] for static_vpt in static_viewpoints]
    dynamic_vpts_list = [(dynamic_vpt, 'dynamic') for dynamic_viewpoints in viewpoint_phases["dynamic_phases"] for dynamic_vpt in dynamic_viewpoints]
    viewpoints_list = static_vpts_list + dynamic_vpts_list
    viewpoints_weight = [1] * len(static_vpts_list) + [4] * len(dynamic_vpts_list) # more weights on dynamic frames to be sampled from
    CONSOLE.print(f"{len(viewpoints_list)} frames will be used: {len(static_vpts_list)} static {len(dynamic_vpts_list)} dynamic")

    first_iter = 0
    progress_bar = tqdm(range(first_iter, fine_p.total_num_iter), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, fine_p.total_num_iter + 1):

        gaussians.update_learning_rate(iteration)
        viewpoint_cam, phase_type = random.choices(viewpoints_list, weights=viewpoints_weight, k=1)[0]
        if iteration == 1: 
            # Note: always begin with the first static frame so that we don't have to reverse for the first iteration
            # to prevent TypeError None raised by stored_state["exp_avg"] = torch.zeros_like(tensor)
            viewpoint_cam = viewpoint_phases["static_phases"][0][0]
            phase_type = "static"
        train_pose = False

        gt_image = viewpoint_cam.gt_image
        hand_mask = viewpoint_cam.hand_mask
        obj_mask = viewpoint_cam.obj_mask
        
        # Apply transformation if it's dynamic
        og_xyz = gaussians.get_xyz # save before-optimization for debugging
        trainable_t_R, fixed_T, fixed_R = gaussians.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
            image_name = viewpoint_cam.image_name, 
            which_object = 1, 
            during_training = train_pose)
        
        render_img_pkg = render(viewpoint_cam, gaussians, pipe, background, rot_cov=True, accum_R=fixed_R, which_object=1, during_training=train_pose)
        render_image = render_img_pkg["render"]
        render_image.register_hook(lambda grad: grad * (1 - hand_mask))

        loss = 0.0
        ### Image loss
        Ll1_image = l1_loss(gt_image, render_image)
        loss += (1.0 - fine_p.lambda_dssim) * Ll1_image + fine_p.lambda_dssim * (1.0 - ssim(gt_image, render_image))
        loss.backward()

        if iteration % 10:
            wandb.log({'step': iteration, 'total loss': loss.item(), 'num points': len(gaussians.get_xyz), 'xyz lr': gaussians.get_xyz_lr()})

        with torch.no_grad():
            if (iteration % (fine_p.total_num_iter / 100) == 0):
                new_image = get_eval_img([gt_image, obj_mask, render_image], [f"GT image {viewpoint_cam.image_name}", "Object mask", "Render image"])
                wandb.log({'Image': wandb.Image(new_image), 'step': iteration})
            # 3. always Reverse object before any densification or pruning
            # Note: enable replace_to_optimizer to replace updated xyz to optimizer
            gaussians.reverse_trans_rot_new(which_object=1, 
                trainable_t_R=trainable_t_R, 
                fixed_T=fixed_T, 
                replace_to_optimizer=True)
            assert torch.allclose(gaussians.get_xyz, og_xyz, atol=0.001)
            # 4. remove redudant points
            # if iteration % fine_p.densification_interval == 0:
            #     size_threshold = 20
            #     gaussians.only_prune(0.01, scene.cameras_extent, size_threshold) 
            if iteration < fine_p.densify_until_iter:
                if iteration > fine_p.densify_from_iter:
                    if iteration % fine_p.opacity_reset_interval:
                        gaussians.reset_opacity()

            # 5. Optimizer
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

            if iteration % 2000 == 0:
                progress_bar.update(2000)

    wandb.finish()
    progress_bar.close()
    final_ply_path  = os.path.join(save_dir, f"gaussians_all.ply")
    CONSOLE.log(f"Final combine Gaussians saved at {final_ply_path}")
    gaussians.save_ply(final_ply_path)

    return final_ply_path