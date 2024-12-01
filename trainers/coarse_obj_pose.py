import gc
import os
import random
from copy import deepcopy

import torch
import wandb
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render
from gaussian_renderer.render_helper import get_render_label
from scene import Scene, GaussianModel
from utils.console import CONSOLE
from utils.dynamic_utils import get_viewpoint_split, get_eval_img, get_eval_img_new
from utils.loss_utils import l1_loss, ssim, l2_loss
from utils.geometry_utils import get_accum_T_seq, get_accum_R_seq

def save_poses_securely(d, image_name, save_dir):
    filename=os.path.join(save_dir, 'obj_pose_sequence.pth')
    try:
        temp_filename = filename + ".tmp"
        torch.save(d, temp_filename)  # Save to a temporary file first
        os.replace(temp_filename, filename)  # Atomically replace the old file with the new
        CONSOLE.log(f"Obj pose sequence saved successfully for {image_name}.")
    except Exception as e:
        CONSOLE.log(f"An error occurred while saving the dictionary for {image_name}: {e}")
    return filename

def get_previous_viewpoint_list(curr_viewpoint_cam, viewpoint_phases, phase, dynamic_weight=1):
    """
    Get a list of previous static/already-optimized viewpoints and its weights to be sampled from
    Note: only sample from the same dynamic phase and the previous static phase
    """
    prev_viewpoints = []
    prev_viewpoints_weights = []
    # append all previous viewpoints in the static phase
    for i in range(phase + 1):
        for prev_cam in viewpoint_phases["static_phases"][phase]:
            prev_viewpoints.append(prev_cam)
            prev_viewpoints_weights.append(1)
    for i in range(phase + 1):
        for prev_cam in viewpoint_phases["dynamic_phases"][phase]:
            if int(prev_cam.image_name) < int(curr_viewpoint_cam.image_name):
                prev_viewpoints.append(prev_cam)
                if i == phase:
                    prev_viewpoints_weights.append(dynamic_weight) # weights for dynamic frames
                else:
                    prev_viewpoints_weights.append(1)
    assert len(prev_viewpoints) == len(prev_viewpoints_weights)
    CONSOLE.print(f"Number Viewpoints to be sampled for {curr_viewpoint_cam.image_name}: {len(prev_viewpoints)}")
    return prev_viewpoints, prev_viewpoints_weights

def end_of_iter_eval(gaussians, obj_pose_sequence, curr_viewpoint_cam, viewpoint_phases, phase, pipe, background):
    obj_pose_sequence = {key: obj_pose_sequence[key] for key in sorted(obj_pose_sequence)}
    accum_T_seq = get_accum_T_seq(obj_pose_sequence)
    accum_R_seq = get_accum_R_seq(obj_pose_sequence)

    with torch.no_grad():
        cam1 = viewpoint_phases["static_phases"][phase][0]
        cam2 = viewpoint_phases["static_phases"][phase][-1]
        cam3 = curr_viewpoint_cam
        img_lst = []
        for cam in [cam1, cam2, cam3]:
            gaussians_copy = deepcopy(gaussians)
            gt_image = cam.gt_image
            trainable_t_R, fixed_T, fixed_R = gaussians_copy.apply_trans_rot_new(accum_T_seq, accum_R_seq, 
                image_name = cam.image_name, 
                which_object = 1, 
                during_training = False)
            render_img_pkg = render(cam, gaussians_copy, pipe, background, rot_cov=True, accum_R=fixed_R, which_object=1, during_training=False)
            render_image = render_img_pkg["render"]
            render_alpha = render_img_pkg["alpha"]
            new_image = get_eval_img([gt_image, render_image, render_alpha], 
                                     [f"GT image {cam.image_name}", "Render image", "Render alpha"])
            img_lst.append(new_image)
        del gaussians_copy
        gc.collect()
        torch.cuda.empty_cache()
        img_size = img_lst[0].size
        new_image_width = img_size[0] 
        new_image_height = img_size[1] * 3
        new_image = Image.new('RGB', (new_image_width, new_image_height))
        for i, img in enumerate(img_lst):
            new_image.paste(img, (0, img_size[1]*i))
        return new_image

def est_coarse_obj_pose(dataset, opt, pipe, coarse_p,
    exp_name, save_dir, obj_gaussians_path,
    static_phases, dynamic_phases, train_frames):

    os.makedirs(save_dir, exist_ok=True)
    train_dir = os.path.join(save_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    ply_dir = os.path.join(save_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    CONSOLE.print(f"Coarse stage of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print(f"Load Gaussian object model from {obj_gaussians_path}")
    CONSOLE.print("Static Phases:", static_phases)
    CONSOLE.print("Dynamic Phases:", dynamic_phases)

    CONSOLE.print("Warm-up iterations for first frame:", coarse_p.warm_up_iter)
    CONSOLE.print("Total number of iterations:", coarse_p.total_num_iter)
    CONSOLE.print(f"Densification is from {coarse_p.densify_from_iter} to {coarse_p.densify_until_iter} every {coarse_p.densification_interval} iter")
    CONSOLE.print("Opacity reset interval:", coarse_p.opacity_reset_interval)
    CONSOLE.print(f"Loss configuration: {coarse_p.lambda_image}*({1.0 - coarse_p.lambda_dssim}*L1 image+{coarse_p.lambda_dssim}*SSIM)+{coarse_p.lambda_Ll1_alpha}*L1 alpha+{coarse_p.lambda_Ll2_alpha}*L2 alpha")
    CONSOLE.print(f"Training schedule: warm up until {coarse_p.warm_up_iter} " + 
                f"densify from {coarse_p.densify_from_iter}, densify until {coarse_p.densify_until_iter} " + 
                f"total num {coarse_p.total_num_iter}")

    gaussians = GaussianModel(sh_degree=0)
    gaussians.load_ply(obj_gaussians_path, train_params=True, is_object=True)

    gaussians.init_obj_pose()
    scene = Scene(dataset, 
                  gaussians, 
                  shuffle=False, 
                  load_or_create_from=False, 
                  load_hand_masks=True, 
                  load_obj_masks=True, 
                  load_est_depths=False)
    CONSOLE.print(f"Spatial learning rate scale: {gaussians.spatial_lr_scale}")
    
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    viewpoints_og = scene.getTrainCameras().copy()
    viewpoint_phases = get_viewpoint_split(viewpoints_og, train_frames, 
        static_phases=static_phases, dynamic_phases=dynamic_phases)
    
    # One objecrt pose sequence for all phases
    obj_pose_sequence = {}

    for phase, dynamic_viewpoints in enumerate(viewpoint_phases["dynamic_phases"]):
        # add the first frame 
        dynamic_viewpoints.append(viewpoint_phases["static_phases"][phase+1][0])
        CONSOLE.log(f"\nProcessing Dynamic Phase {phase}: frame {dynamic_viewpoints[0].image_name} - {dynamic_viewpoints[-2].image_name} " + 
            f"with an additional frame from next static {dynamic_viewpoints[-1].image_name}")

        for i, curr_viewpoint_cam in enumerate(dynamic_viewpoints):
            #################### BEGIN of processing one frame ####################
            wandb.init(project=exp_name, name=f"frame-{curr_viewpoint_cam.image_name}", dir="/scratch_net/biwidl301/daizhang/wandb")
            CONSOLE.log(f"Optimizing object pose for {curr_viewpoint_cam.image_name}...")

            # Set initial object poses
            obj_pose_sequence[curr_viewpoint_cam.image_name] = None
            obj_pose_sequence = {key: obj_pose_sequence[key] for key in sorted(obj_pose_sequence)}
            # Get accumulated T & R sequence
            accum_T_seq = get_accum_T_seq(obj_pose_sequence)
            accum_R_seq = get_accum_R_seq(obj_pose_sequence)
            # initialize a list of previous static/already-optimized viewpoints to be sampled from
            prev_viewpoints, prev_viewpoints_weights = get_previous_viewpoint_list(curr_viewpoint_cam, viewpoint_phases, phase, dynamic_weight=2)

            gaussians.init_obj_pose()
            gaussians.train_coarse_obj_setup(opt, divide_3dgs_lr_by=10.0, divide_pose_lr_by=1.0) # hyperparameters to modified
            saved_lrs = gaussians.get_lrs()
            saved_xyz_lr = gaussians.get_xyz_lr()
            # CONSOLE.log(f"Saved inital learning rates: {saved_lrs}")

            # record learning rate in wandb config
            param_lrs = {}
            for param_group in gaussians.optimizer.param_groups:
                param_lrs['lr_' + param_group['name']] = param_group['lr']
            wandb.run.config.update(param_lrs, allow_val_change=True) 
            
            # Zeroing Gaussian attributes' learning rate for the warm up training
            if i == 0:
                pass
            else:
                gaussians.zero_gaussians_lr()
            # CONSOLE.log(f"Learning rates after zeroing Gaussians' lr: {gaussians.get_lrs()}") 

            first_iter = 0
            progress_bar = tqdm(range(first_iter, coarse_p.total_num_iter), desc="Training progress")
            first_iter += 1
            
            for iteration in range(first_iter, coarse_p.total_num_iter + 1):
                ### Adjust pose learning rate during different stages of iterations
                if iteration == coarse_p.warm_up_iter:
                    gaussians.load_lrs(saved_lrs)
                    gaussians.load_xyz_lr(saved_xyz_lr / 10.0) # Set xyz lr to a smaller value after warm-up and before densification
                    # CONSOLE.log(f"Warm-up training (pose only) training ends at iter{iteration}, lr after loading: {gaussians.get_lrs()}")
                if iteration == coarse_p.densify_from_iter:
                    gaussians.zero_pose_lr()
                    gaussians.load_xyz_lr(saved_xyz_lr) # Reset xyz lr to original value when densification starts
                    # CONSOLE.log(f"Start densification from iter{iteration}, lr: {gaussians.get_lrs()}")
                if iteration == coarse_p.densify_until_iter:
                    gaussians.restore_pose_lr(opt)
                    # CONSOLE.log(f"End densification from iter{iteration}, lr after restoring: {gaussians.get_lrs()}")
                    
                if iteration > coarse_p.densify_from_iter:
                    gaussians.update_learning_rate(iteration - coarse_p.densify_from_iter) # for xyz only, update only when densification starts
                
                ### Begin of Frame Selection ###
                if random.random() <= coarse_p.curr_vpt_prob: # or iteration < coarse_p.warm_up_iter: # optimize pose of current frame
                    if i == len(dynamic_viewpoints) - 1: 
                        # we sample all frames of next static phase for training if the last frame of dynamic phases
                        viewpoint_cam = random.choice(viewpoint_phases["static_phases"][phase+1])
                        est_pose_image_name = curr_viewpoint_cam.image_name
                    else:
                        viewpoint_cam = curr_viewpoint_cam
                        est_pose_image_name = viewpoint_cam.image_name
                    during_training = True
                else: # regularize Gaussians' variables with previous frame
                    while True:
                        viewpoint_idx = random.choices(range(len(prev_viewpoints)), weights=prev_viewpoints_weights)[0]
                        viewpoint_cam = prev_viewpoints[viewpoint_idx]
                        if torch.any(viewpoint_cam.obj_mask):
                            break
                        else:
                            CONSOLE.log(f"Can't find any obj in {viewpoint_cam.image_name}, random select another one")   
                    est_pose_image_name = viewpoint_cam.image_name
                    during_training = False
                           
                if iteration == 1: 
                    # Note: always begin with the first static frame so that we don't have to reverse for the first iteration
                    # to prevent TypeError None raised by stored_state["exp_avg"] = torch.zeros_like(tensor)
                    viewpoint_cam = viewpoint_phases["static_phases"][0][0]
                    est_pose_image_name = viewpoint_cam.image_name
                    during_training = False
                ### End of Frame Selection ###
                    
                gt_image = viewpoint_cam.gt_image
                hand_mask = viewpoint_cam.hand_mask
                obj_mask = viewpoint_cam.obj_mask

                og_xyz = gaussians.get_xyz # save before-optimization for debugging
                trainable_t_R, fixed_T, fixed_R = gaussians.apply_trans_rot_new(accum_T_seq, accum_R_seq,
                    image_name = est_pose_image_name, 
                    which_object = 1, 
                    during_training = during_training)
                ### Debug ###
                # with torch.no_grad():
                #     debug_xyz = gaussians.apply_trans_rot_debug(og_xyz, obj_pose_sequence, est_pose_image_name, which_object=1, during_training=during_training)
                #     assert torch.allclose(gaussians.get_xyz, debug_xyz, atol=0.001), f"{gaussians.get_xyz} vs. {debug_xyz}"
                
                ### Modified version of renderer with rotated 3D cov
                render_img_pkg = render(viewpoint_cam, gaussians, pipe, background, rot_cov=True, accum_R=fixed_R, which_object=1, during_training=during_training)
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
                image_loss = (1.0 - coarse_p.lambda_dssim) * Ll1_image + coarse_p.lambda_dssim * (1.0 - ssim(gt_image, render_image))
                loss += coarse_p.lambda_image * image_loss
                ### L1 alpha loss
                Ll1_alpha = l1_loss(obj_mask, render_alpha)
                loss += coarse_p.lambda_Ll1_alpha * Ll1_alpha
                ### L2 alpha loss
                Ll2_alpha = l2_loss(obj_mask, render_alpha)
                loss += coarse_p.lambda_Ll2_alpha * Ll2_alpha

                loss.backward()

                if iteration % 10 == 0:
                    try:
                        trans_grad, rot_grad = gaussians.get_obj_pose_grad()
                    except:
                        trans_grad, rot_grad = 0, 0
                    wandb.log({'step': iteration, 
                        'total loss': loss.item(), 
                        'image loss': image_loss.item(), 
                        'l2 alpha loss': Ll2_alpha.item(),
                        'num points': len(gaussians.get_xyz),
                        'xyz lr': gaussians.get_xyz_lr(),
                        't Grad': trans_grad,
                        'R Grad': rot_grad,
                    })
                    if during_training:
                        wandb.log({'step': iteration, 'curr vpt loss': loss.item()})
                
                grad_threshold = coarse_p.densify_grad_threshold # Value = 0.001 based on our observation
                with torch.no_grad():
                    if iteration > coarse_p.warm_up_iter: 
                        # 1. always Record densification stats: grad_accum & denom
                        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # 2. Visualization every interval BEFORE and AFTER densification
                    if iteration % coarse_p.densification_interval == 0 or iteration < 10:
                        ######## Begin of Visualization Block ########
                        assert gaussians.get_label.shape == gaussians.xyz_gradient_accum.shape
                        grads = gaussians.xyz_gradient_accum / gaussians.denom # normalize the accumulative gradient
                        grads[grads.isnan()] = 0.0
                        gaussians._label = grads
                        render_label = get_render_label(viewpoint_cam, gaussians, background)
                        gaussians._label = torch.where(gaussians._label > grad_threshold, torch.tensor(1.0).to('cuda'), torch.tensor(0.0).to('cuda'))
                        render_label_binary = get_render_label(viewpoint_cam, gaussians, background)
                        new_image = get_eval_img_new([gt_image, render_image, render_label_binary, obj_mask, render_alpha], 
                            [f"GT image {viewpoint_cam.image_name}", "Render image", "Binary Grad", "Object mask", "Render alpha"], rows=2, cols=3)
                        wandb.log({"N_pts above grad threshold": torch.sum(grads > grad_threshold).item(), 
                            'max grad': torch.max(grads).item(),
                            'step': iteration})
                        if iteration % (coarse_p.densification_interval*5) == 0:
                            if iteration <= coarse_p.densify_from_iter:
                                wandb.log({'Before densify': wandb.Image(new_image), 'step': iteration})
                            else:
                                wandb.log({'After densify': wandb.Image(new_image), 'step': iteration})
                        if iteration < 10 and during_training:
                            wandb.log({'First iters': wandb.Image(new_image), 'step': iteration})
                        ######## End of Visualization Block ########
                    
                    # 3. reverse Gaussians object to its original place
                    gaussians.reverse_trans_rot_new(which_object=1, 
                        trainable_t_R=trainable_t_R, 
                        fixed_T=fixed_T, 
                        replace_to_optimizer=True)
                    assert torch.allclose(gaussians.get_xyz, og_xyz, atol=0.001)

                    if during_training: # -> viewpoint_cam = curr_viewpoint_cam
                        # Update the sequence with updated poses after optimization
                        updated_trans, updated_rot = gaussians.get_obj_trans_rot
                        obj_pose_sequence[est_pose_image_name] = {"translation": updated_trans.cpu(), "rotation": updated_rot.cpu()}
                    
                    # 4. Densification
                    ######## Begin of Densification Block ########
                    if iteration < coarse_p.densify_until_iter:
                        if iteration > coarse_p.densify_from_iter and iteration % coarse_p.densification_interval == 0:
                            # Disable actual densify and prune only record gradient accumulation
                            size_threshold = 20 if iteration > coarse_p.opacity_reset_interval else None
                            gaussians.densify_and_prune(grad_threshold, coarse_p.min_opacity, scene.cameras_extent, size_threshold, which_object=1)                      
                            # ensure accumulative gradient is cleared after densification
                            assert torch.sum(gaussians.xyz_gradient_accum) == 0
                        if iteration % coarse_p.opacity_reset_interval == 0 and iteration > coarse_p.densify_from_iter:
                            # Only reset opacity during densification
                            CONSOLE.log(f"Reset opacity at iter{iteration}") 
                            gaussians.reset_opacity_for_object(which_object=1)
                    ######## End of Densification Block ########

                    # 5. Optimizer
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    
                    if iteration % 2000 == 0:
                        progress_bar.update(2000)
                    torch.cuda.empty_cache()

            CONSOLE.log(f"Finished optimizing object pose for {curr_viewpoint_cam.image_name}...")
            wandb.finish()
            progress_bar.close()
            #################### END of processing one frame ####################

            # Save the obj_pose_sequence every time it update
            pose_seq_path = save_poses_securely(obj_pose_sequence, curr_viewpoint_cam.image_name, save_dir)

            # Save the object's ply every x iterations
            if i % coarse_p.save_ply_every == 0:
                gaussians.save_ply(os.path.join(ply_dir, f"gaussians_{curr_viewpoint_cam.image_name}.ply"))
                CONSOLE.log(f"Saved the object's ply for {curr_viewpoint_cam.image_name}...")

            # End-of-iteration evaluation for current viewpoint
            eval_img = end_of_iter_eval(gaussians, obj_pose_sequence, curr_viewpoint_cam, viewpoint_phases, phase, pipe, background)
            eval_img.save(os.path.join(train_dir, f"{curr_viewpoint_cam.image_name}.png"))

        ## End of iteration of viewpoints in a dynamic phase
        final_ply_path = os.path.join(ply_dir, "gaussians_final.ply")
        gaussians.save_ply(final_ply_path)
        CONSOLE.log(f"Saved final object ply at {final_ply_path}")

    return pose_seq_path, final_ply_path
        