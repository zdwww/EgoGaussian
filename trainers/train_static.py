import os
import torch
import wandb
import torch.nn as nn
from tqdm import tqdm
from random import randint
from copy import deepcopy

from utils.loss_utils import l1_loss, ssim
from utils.dynamic_utils import get_viewpoint_split, gray_tensor_to_PIL, get_eval_img
from utils.console import CONSOLE
from gaussian_renderer import render
from scene import Scene, GaussianModel
from gaussian_renderer.render_helper import get_render_label

def train_static(dataset, opt, pipe, static_p, exp_name, save_dir, static_phases, train_frames):
    os.makedirs(save_dir, exist_ok=True)
    ply_dir = os.path.join(save_dir, "ply")
    os.makedirs(ply_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    pred_mask_dir = os.path.join(save_dir, "obj_masks")
    os.makedirs(pred_mask_dir, exist_ok=True)

    CONSOLE.print(f"Static stage of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print("Standard training iterations:", static_p.std_train_iter)
    CONSOLE.print("Densify from iter", static_p.densify_from_iter)
    CONSOLE.print("Densify until iter", static_p.densify_until_iter)
    CONSOLE.print("Entropy regularization iterations:", static_p.entropy_reg_iter)
    CONSOLE.print("Label training iterations:", static_p.label_train_iter)
    total_iterations = static_p.std_train_iter + static_p.entropy_reg_iter + static_p.label_train_iter
    CONSOLE.print("Total iterations:", total_iterations)
    CONSOLE.print("Static Phases to use:", static_phases)

    # Setup for first round
    assert dataset.sh_degree == 0 # SH degree should be 0 for video reconstruction
    gaussians = GaussianModel(dataset.sh_degree) 
    scene = Scene(dataset, 
                  gaussians, 
                  shuffle=False, # never shuffle viewpoints for video
                  load_or_create_from=True, # create_from_pcd (colmap output)
                  load_hand_masks=True, 
                  load_obj_masks=True # hand & obj masks always required
                  )
    gaussians.training_setup(opt)
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_og = scene.getTrainCameras().copy()
    # Load only frames from static phases
    viewpoint_dict = get_viewpoint_split(viewpoint_og, train_frames=train_frames, static_phases=static_phases) # Set train_frames to None if use all

    for phase, static_viewpoints in enumerate(viewpoint_dict["static_phases"]):

        wandb.init(project=exp_name, name=f"Static Phase-{phase+1} / {len(viewpoint_dict['static_phases'])}", 
            dir="/scratch_net/biwidl301/daizhang/wandb")

        ############################### Training ###############################
        CONSOLE.log(f"\nProcessing Static Phase {phase}: frame {static_viewpoints[0].image_name} - {static_viewpoints[-1].image_name}")
        CONSOLE.log(f"Number of training frames in this phase: {len(static_viewpoints)}")
        viewpoint_stack = static_viewpoints.copy()
        first_iter = 0
        progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
        first_iter += 1
        ema_loss_for_log = 0.0
        
        CONSOLE.log("Standard 3DGS training starts...")
        for iteration in range(first_iter, total_iterations + 1):
            viewpoint_cam = viewpoint_stack[randint(0, len(viewpoint_stack)-1)] # random sample
            # Load input images/masks
            input_img_pkg = viewpoint_cam.get_input_pkg()
            (gt_image, hand_mask, obj_mask) = (
                input_img_pkg["gt_image"],
                input_img_pkg["hand_mask"],
                input_img_pkg["obj_mask"],
            )
            
            # Render RGB images
            render_img_pkg = render(viewpoint_cam, gaussians, pipe, background)
            (render_image, viewspace_point_tensor, visibility_filter, radii) = (
                render_img_pkg["render"],
                render_img_pkg["viewspace_points"],
                render_img_pkg["visibility_filter"],
                render_img_pkg["radii"],
            )
            
            # Loss handler
            loss = 0.0
            if iteration <= static_p.std_train_iter + static_p.entropy_reg_iter: # must be sum of iterations!
                gaussians.update_learning_rate(iteration)
                ## 1. Standard Training
                render_image.register_hook(lambda grad: grad * (1 - hand_mask)) # gradient mask instead of img mask
                Ll1_image = l1_loss(render_image, gt_image)
                image_loss = (1.0 - opt.lambda_dssim) * Ll1_image + \
                    opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
                loss += image_loss

                if iteration > static_p.std_train_iter:
                    ## 2. Entropy Regularization
                    vis_opacities = (gaussians.get_opacity)[visibility_filter]
                    entropy_loss = (- vis_opacities * torch.log(vis_opacities + 1e-10) 
                                    - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
                    loss += 0.1 * entropy_loss # parameter to tune
            else:
                ## 3. Object Label Training
                render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                render_label.register_hook(lambda grad: grad * (1 - hand_mask))
                criterion = nn.BCEWithLogitsLoss()
                assert obj_mask is not None, "Object masks can't be None for rewind interval"
                loss += criterion(input=render_label, target=obj_mask)
            loss.backward()

            wandb.log({'step': iteration, 'total loss': loss.item(), 'image loss': image_loss.item(), 'num points': len(gaussians.get_xyz)})

            with torch.no_grad():
                ## Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 1000 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                              "Num_points": len(gaussians.get_xyz)})
                    progress_bar.update(1000)
                if iteration == total_iterations:
                    progress_bar.close()

                ## Densification
                if iteration <= static_p.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    if iteration > static_p.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                ## Optimizer step
                if iteration <= total_iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                ## For logging purposes
                if iteration == static_p.densify_from_iter: CONSOLE.log(f"Densification starts at iter{iteration}...")
                if iteration == static_p.densify_until_iter: CONSOLE.log(f"Densification ends at iter{iteration}. num_points:", len(gaussians.get_xyz))
                if iteration == static_p.std_train_iter: CONSOLE.log(f"Entropy regularization starts at iter{iteration}...")

                ## Prune low-opacity points at end of entropy regularization
                if iteration == static_p.std_train_iter + static_p.entropy_reg_iter:
                    CONSOLE.log(f"Entropy regularization ends at iter{iteration}") 
                    CONSOLE.log("Before pruning pts with low opacity:", len(gaussians.get_xyz))
                    gaussians.prune_points(mask=(gaussians.get_opacity < 0.5).squeeze())
                    CONSOLE.log("After pruning pts with low opacity:", len(gaussians.get_xyz))

                ## Update viewpoint_stack to fewer frames during label training
                if iteration == static_p.std_train_iter + static_p.entropy_reg_iter:
                    if phase == 0: # if first phase, keep the last few frames
                        viewpoint_stack = viewpoint_stack[-static_p.rewind_frames: ]
                    elif phase == len(viewpoint_dict["static_phases"]) - 1: # if last phase, keep the first few frames
                        viewpoint_stack = viewpoint_stack[:static_p.rewind_frames]
                    else: # if middle phase, keep the first and last few frames
                        viewpoint_stack = viewpoint_stack[:static_p.rewind_frames] + \
                            viewpoint_stack[-static_p.rewind_frames: ]
                    CONSOLE.log(f"Object label training starts at iter{iteration}. num_frames:", len(viewpoint_stack))
                    gaussians.update_lr_for_label(label_lr=static_p.label_lr)
        
        wandb.finish()
        progress_bar.close()
        CONSOLE.log(f"Training complete for phase{phase}. Saving results...")
        gaussians.infer_is_object_from_label()
        gaussians.save_ply(os.path.join(ply_dir, f"static_phase{phase}.ply"))

        ################### Remove background to get Gaussian for object ###################
        CONSOLE.log("Saving object and background gaussians separately...")
        with torch.no_grad():
            gaussians_obj = deepcopy(gaussians)
            gaussians_obj.prune_points(mask=torch.flatten(gaussians.get_is_object.detach() != 1))
            gaussians_obj.save_ply(os.path.join(ply_dir, f"static_phase{phase}_obj.ply"))
            gaussians_bg = deepcopy(gaussians)
            gaussians_bg.prune_points(mask=torch.flatten(gaussians.get_is_object.detach() != 0))
            gaussians_bg.save_ply(os.path.join(ply_dir, f"static_phase{phase}_bg.ply"))
            del gaussians_bg

        ################### Evaluation ###################
        CONSOLE.log("Saving evaluation images and predicted static object masks...")
        for viewpoint_cam in static_viewpoints:
            with torch.no_grad():
                render_image = render(viewpoint_cam, gaussians, pipe, background)["render"]
                render_object = render(viewpoint_cam, gaussians_obj, pipe, background)["render"]
                render_label = torch.mean(get_render_label(viewpoint_cam, gaussians, background), dim=0, keepdim=True)
                binary_label = (render_label > 0.5).float()
                input_img_pkg = viewpoint_cam.get_input_pkg()
                gt_image = input_img_pkg["gt_image"]
                hand_mask = input_img_pkg["hand_mask"]
                
                pred_mask_img = gray_tensor_to_PIL(binary_label)
                eval_img = get_eval_img([gt_image, render_image, render_object, binary_label, hand_mask],
                                        [f"GT img {viewpoint_cam.image_name}", "Render img", "Render obj", "Pred mask", "GT hand mask"])
                pred_mask_img.save(os.path.join(pred_mask_dir, f"{viewpoint_cam.image_name}.png"))
                eval_img.save(os.path.join(eval_dir, f"{viewpoint_cam.image_name}.jpg"))
        
        ################### Reinitialize Gaussians for the next phase ###################
        gaussians = GaussianModel(dataset.sh_degree)
        scene.re_initialize(gaussians)
        gaussians.training_setup(opt)
        torch.cuda.empty_cache()
        # break # only train with the first phase for object initialization
    
    # Return the path of saved object of phase0
    return (os.path.join(ply_dir, "static_phase0_obj.ply")), pred_mask_dir
