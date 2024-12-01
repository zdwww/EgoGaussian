import os
import random

import torch
import wandb
from tqdm import tqdm

from gaussian_renderer import render
from gaussian_renderer.render_helper import get_render_label
from scene import Scene, GaussianModel
from utils.console import CONSOLE
from utils.dynamic_utils import get_viewpoint_split, get_eval_img
from utils.loss_utils import l1_loss, ssim, l2_loss
from utils.geometry_utils import matrix_to_rot6d
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

def fine_tune_obj(dataset, opt, pipe, fine_p, exp_name, save_dir, 
    obj_gaussians_paths, obj_pose_seq_path, 
    static_phases, dynamic_phases, train_frames):

    os.makedirs(save_dir, exist_ok=True)

    CONSOLE.print(f"Coarse stage of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print(f"Load Gaussian object model from {obj_gaussians_paths}, obj pose from {obj_pose_seq_path}")
    CONSOLE.print("Static Phases:", static_phases)
    CONSOLE.print("Dynamic Phases:", dynamic_phases)
    CONSOLE.print("Total number of iterations:", fine_p.total_num_iter)
    CONSOLE.print(f"Densification is from {fine_p.densify_from_iter} to {fine_p.densify_until_iter} every {fine_p.densification_interval} iter")
    CONSOLE.print("Opacity reset interval:", fine_p.opacity_reset_interval)
    CONSOLE.print(f"Loss configuration: {fine_p.lambda_Ll1_image} * L1 image + {fine_p.lambda_Ll1_alpha} * L1 alpha + {fine_p.lambda_Ll2_alpha} * L2 alpha")

    obj_pose_sequence = torch.load(obj_pose_seq_path)
    CONSOLE.log(f"Load object poses from {obj_pose_seq_path}")
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)

    output_path = {}

    for i, obj_gaussians_path in enumerate(obj_gaussians_paths):
        if 'static' in obj_gaussians_path:
            obj_type = 'from-static'
        elif 'coarse' in obj_gaussians_path:
            obj_type = 'from-coarse'
        else:
            raise ValueError("Neither 'static' nor 'coarse' found in obj_gaussians_path")

        this_obj_dir = os.path.join(save_dir, obj_type)
        os.makedirs(this_obj_dir, exist_ok=True)

        gaussians_obj = GaussianModel(sh_degree=0)
        gaussians_obj.load_ply(obj_gaussians_path, train_params=True, is_object=True)
        CONSOLE.log(f"{obj_type} Gaussian objects loaded from {obj_gaussians_path}")

        gaussians = gaussians_obj
        scene = Scene(dataset, gaussians, shuffle=False, load_or_create_from=False, load_hand_masks=True, load_obj_masks=True)

        wandb.init(project=exp_name, name=f"fine-tune-{obj_type}", dir="/scratch_net/biwidl301/daizhang/wandb")
                  
        # Initialize object poses for fine-tuning
        gaussians.init_obj_pose()
        gaussians.train_fine_obj_setup(opt, divide_pose_lr_by = 10.0, divide_3dgs_lr_by = 5.0)

        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        viewpoints_og = scene.getTrainCameras().copy()
        viewpoint_phases = get_viewpoint_split(viewpoints_og, train_frames, 
            static_phases=static_phases, dynamic_phases=dynamic_phases)

        static_vpts_list = [(static_vpt, 'static') for static_viewpoints in viewpoint_phases["static_phases"] for static_vpt in static_viewpoints]
        dynamic_vpts_list = [(dynamic_vpt, 'dynamic') for dynamic_viewpoints in viewpoint_phases["dynamic_phases"] for dynamic_vpt in dynamic_viewpoints]
        viewpoints_list = static_vpts_list + dynamic_vpts_list
        viewpoints_weight = [1] * len(static_vpts_list) + [5] * len(dynamic_vpts_list) # more weights on dynamic frames to be sampled from
        CONSOLE.print(f"{len(viewpoints_list)} frames will be used: {len(static_vpts_list)} static {len(dynamic_vpts_list)} dynamic")

        first_iter = 0
        progress_bar = tqdm(range(first_iter, fine_p.total_num_iter), desc="Training progress")
        first_iter += 1

        for iteration in range(first_iter, fine_p.total_num_iter + 1):
            if iteration == fine_p.densify_from_iter:
                CONSOLE.log(f"Start densification from iter{iteration}...")
                gaussians.zero_pose_lr()
            if iteration == fine_p.densify_until_iter:
                CONSOLE.log(f"End densification from iter{iteration}...Disble ")
                gaussians.restore_pose_lr(opt)

            gaussians.update_learning_rate(iteration)
            viewpoint_cam, phase_type = random.choices(viewpoints_list, weights=viewpoints_weight, k=1)[0]
            if iteration == 1: 
                # Note: always begin with the first static frame so that we don't have to reverse for the first iteration
                # to prevent TypeError None raised by stored_state["exp_avg"] = torch.zeros_like(tensor)
                viewpoint_cam = viewpoint_phases["static_phases"][0][0]
                phase_type = "static"

            gt_image = viewpoint_cam.gt_image
            hand_mask = viewpoint_cam.hand_mask
            obj_mask = viewpoint_cam.obj_mask
            if phase_type == "static":
                train_pose = False
            elif phase_type == "dynamic": # if dynamic frame, then also train pose
                train_pose = True
                # Extract current frame transformation from a static sequence and update it to optimizer
                prior_transform = obj_pose_sequence[viewpoint_cam.image_name] # extract current 
                prior_trans, prior_rot = prior_transform["translation"].to("cuda"), prior_transform["rotation"].to("cuda")
                # for debugging only
                if torch.all(prior_trans == 0) or torch.equal(prior_rot.cpu(), torch.eye(3)):
                    CONSOLE.log(f"{viewpoint_cam.image_name}'s extracted pose is trivial!")
                # Update the pose values in-place by .data
                gaussians.trainable_object_move.obj_translation.data = prior_trans
                gaussians.trainable_object_move.obj_rotation_6d.data = matrix_to_rot6d(prior_rot)
            
            # Apply transformation if it's dynamic
            og_xyz = gaussians.get_xyz # save before-optimization for debugging
            trainable_t_R, fixed_T, fixed_R = gaussians.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
                image_name = viewpoint_cam.image_name, 
                which_object = 1, 
                during_training = train_pose)
            
            render_img_pkg = render(viewpoint_cam, gaussians, pipe, background, rot_cov=True, accum_R=fixed_R, which_object=1, during_training=train_pose)
            render_image = render_img_pkg["render"]
            render_alpha = render_img_pkg["alpha"]
            render_image.register_hook(lambda grad: grad * (1 - hand_mask))
            render_alpha.register_hook(lambda grad: grad * (1 - hand_mask))

            viewspace_point_tensor = render_img_pkg["viewspace_points"]
            visibility_filter = render_img_pkg["visibility_filter"]
            radii = render_img_pkg["radii"]

            loss = 0.0
            ### Image loss
            gt_image = torch.mul(gt_image, obj_mask) # for coarse object pose estimation, mask out background

            Ll1_image = l1_loss(gt_image, render_image)
            image_loss = (1.0 - fine_p.lambda_dssim) * Ll1_image + fine_p.lambda_dssim * (1.0 - ssim(gt_image, render_image))
            loss += image_loss

            ### L1 alpha loss
            Ll1_alpha = l1_loss(obj_mask, render_alpha)
            loss += fine_p.lambda_Ll1_alpha * Ll1_alpha
            ### L2 alpha loss
            Ll2_alpha = l2_loss(obj_mask, render_alpha)
            loss += fine_p.lambda_Ll2_alpha * Ll2_alpha

            loss.backward()

            if iteration % 10 == 0:
                wandb.log({'step': iteration, 'total loss': loss.item(), 
                    'image loss': image_loss.item(), 'l2 alpha loss': Ll2_alpha.item(), 
                    'num points': len(gaussians.get_xyz), 'xyz lr': gaussians.get_xyz_lr()})

            allow_densification = True
            grad_threshold = fine_p.densify_grad_threshold # Value = 0.001 based on our observation

            with torch.no_grad():
                # 1. always Record densification stats: grad_accum & denom
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # 2. Visualization every interval BEFORE and AFTER densification
                if iteration % fine_p.densification_interval == 0:
                    ######## Begin of Visualization Block ########
                    assert gaussians.get_label.shape == gaussians.xyz_gradient_accum.shape
                    grads = gaussians.xyz_gradient_accum / gaussians.denom # normalize the accumulative gradient
                    grads[grads.isnan()] = 0.0
                    gaussians._label = grads
                    render_label = get_render_label(viewpoint_cam, gaussians, background)
                    gaussians._label = torch.where(gaussians._label > grad_threshold, torch.tensor(1.0).to('cuda'), torch.tensor(0.0).to('cuda'))
                    render_label_binary = get_render_label(viewpoint_cam, gaussians, background)
                    new_image = get_eval_img([gt_image, obj_mask, render_image, render_alpha, render_label_binary], 
                        [f"GT image {viewpoint_cam.image_name}", "Object mask", "Render image", "Render alpha", "Binary Grad"])
                    wandb.log({"N_pts above grad threshold": torch.sum(grads > grad_threshold).item(), 
                        'max grad': torch.max(grads).item(),
                        'step': iteration})
                    if iteration % (fine_p.densification_interval * 10) == 0:
                        if iteration <= opt.densify_from_iter:
                            wandb.log({'Before densify': wandb.Image(new_image), 'step': iteration})
                        else:
                            wandb.log({'After densify': wandb.Image(new_image), 'step': iteration})
                    ######## End of Visualization Block ########

                # 3. always Reverse object before any densification or pruning
                # Note: enable replace_to_optimizer to replace updated xyz to optimizer
                gaussians.reverse_trans_rot_new(which_object=1, 
                    trainable_t_R=trainable_t_R, 
                    fixed_T=fixed_T, 
                    replace_to_optimizer=True)
                assert torch.allclose(gaussians.get_xyz, og_xyz, atol=0.001)

                # 4. Densification
                ######## Begin of Densification Block ########
                if allow_densification:
                    if iteration < fine_p.densify_until_iter:
                        if iteration > fine_p.densify_from_iter and iteration % fine_p.densification_interval == 0:
                            # Disable actual densify and prune only record gradient accumulation
                            size_threshold = 20 if iteration > fine_p.opacity_reset_interval else None
                            gaussians.densify_and_prune(grad_threshold, fine_p.min_opacity, scene.cameras_extent, size_threshold, which_object=1) 
                            # ensure accumulative gradient is cleared after densification
                            assert torch.sum(gaussians.xyz_gradient_accum) == 0

                        if iteration % fine_p.opacity_reset_interval == 0 and iteration > fine_p.densify_from_iter:
                            gaussians.reset_opacity_for_object(which_object=1)
                ######## End of Densification Block ########

                # 5. Optimizer
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

                if train_pose:
                    updated_trans, updated_rot = gaussians.get_obj_trans_rot # rot in matrix form
                    # if (not torch.equal(updated_trans, prior_trans)) or (not torch.equal(updated_rot, prior_rot)):
                    #     CONSOLE.log(f"Translation & Rotation updated in iter{iteration} for frame{viewpoint_cam.image_name}")
                    obj_pose_sequence[viewpoint_cam.image_name] = {"translation": updated_trans.cpu(), "rotation": updated_rot.cpu()}
                    # Every time update the obj pose seq, also update accum T & R
                    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
                    accum_R_seq = get_accum_R_seq(obj_pose_sequence)
                    pose_seq_path = save_poses_securely(obj_pose_sequence, this_obj_dir)

                if iteration % 2000 == 0:
                    progress_bar.update(2000)

        wandb.finish()
        progress_bar.close()
        pose_seq_path = save_poses_securely(obj_pose_sequence, this_obj_dir)
        final_ply_path  = os.path.join(this_obj_dir, f"gaussians_fine.ply")
        gaussians.save_ply(final_ply_path)
        output_path[obj_type] = (pose_seq_path, final_ply_path)

    return output_path

    