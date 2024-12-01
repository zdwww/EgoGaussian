import os
from tqdm import tqdm
from random import randint
import torch
import torch.nn.functional as F
import wandb

from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.dynamic_utils import *
from utils.console import CONSOLE

def dilate_mask(mask, kernel_size):
    with torch.no_grad():
        kernel = torch.ones(1, 1, kernel_size, kernel_size).cuda()
        padding_size = kernel_size // 2
        dilated_mask = F.conv2d(mask.unsqueeze(0).float(), kernel, padding=padding_size) > 0
        dilated_mask = dilated_mask.int().squeeze(0)
        assert dilated_mask.shape == mask.shape
        return dilated_mask

def train_background(dataset, opt, pipe, bg_p, exp_name, save_dir, train_frames, 
    dilate_size=None, use_all_frames=False):
    os.makedirs(save_dir, exist_ok=True)
    eval_dir = os.path.join(save_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    CONSOLE.print(f"Background stage of experiment {exp_name} saved in {save_dir}")
    CONSOLE.print("Standard training iterations:", bg_p.std_train_iter)
    CONSOLE.print("Densify from iter", bg_p.densify_from_iter)
    CONSOLE.print("Densify until iter", bg_p.densify_until_iter)
    CONSOLE.print("Entropy regularization iterations:", bg_p.entropy_reg_iter)

    total_iterations = bg_p.std_train_iter + bg_p.entropy_reg_iter
    CONSOLE.print("Total iterations:", total_iterations)

    if dilate_size is not None:
        CONSOLE.print(f"Dilate masks with a kernal size of {dilate_size}")

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

    # Process the viewpoints to keep only training frames
    viewpoint_og = scene.getTrainCameras().copy()
    if not use_all_frames:
        train_frames = [int(image_name) for image_name in train_frames]
        viewpoint_og = [cam for cam in viewpoint_og if int(cam.image_name) in train_frames]
        assert len(viewpoint_og) == len(train_frames)
    CONSOLE.print(f"Number of training frames: {len(viewpoint_og)}")
    
    if True:
        wandb.init(project=exp_name, name="Background", dir="/scratch_net/biwidl301/daizhang/wandb")
        ############################### Training ###############################
        viewpoint_stack = viewpoint_og.copy()
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
            int_mask = torch.logical_or(hand_mask, obj_mask).int()
            if dilate_size is not None:
                int_mask = dilate_mask(int_mask, dilate_size)
            
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
            if iteration <= bg_p.std_train_iter + bg_p.entropy_reg_iter: # must be sum of iterations!
                gaussians.update_learning_rate(iteration)
                ## 1. Standard Training
                render_image.register_hook(lambda grad: grad * (1 - int_mask)) # gradient mask instead of img mask
                Ll1_image = l1_loss(render_image, gt_image)
                image_loss = (1.0 - opt.lambda_dssim) * Ll1_image + \
                    opt.lambda_dssim * (1.0 - ssim(render_image, gt_image))
                loss += image_loss

                if iteration > bg_p.std_train_iter:
                    ## 2. Entropy Regularization
                    vis_opacities = (gaussians.get_opacity)[visibility_filter]
                    entropy_loss = (- vis_opacities * torch.log(vis_opacities + 1e-10) 
                                    - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)).mean()
                    loss += 0.1 * entropy_loss # parameter to tune
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
                if iteration <= bg_p.densify_until_iter:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration > bg_p.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                    if iteration % opt.opacity_reset_interval == 0:
                        gaussians.reset_opacity()

                ## Optimizer step
                if iteration <= total_iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

                ## For logging purposes
                if iteration == bg_p.densify_from_iter: CONSOLE.log(f"Densification starts at iter{iteration}...")
                if iteration == bg_p.densify_until_iter: CONSOLE.log(f"Densification ends at iter{iteration}. num_points:", len(gaussians.get_xyz))
                if iteration == bg_p.std_train_iter: CONSOLE.log(f"Entropy regularization starts at iter{iteration}...")

                ## Prune low-opacity points at end of entropy regularization
                if iteration == bg_p.std_train_iter + bg_p.entropy_reg_iter:
                    CONSOLE.log(f"Entropy regularization ends at iter{iteration}") 
                    CONSOLE.log("Before pruning pts with low opacity:", len(gaussians.get_xyz))
                    gaussians.prune_points(mask=(gaussians.get_opacity < 0.5).squeeze())
                    CONSOLE.log("After pruning pts with low opacity:", len(gaussians.get_xyz))
        
        wandb.finish()
        progress_bar.close()
        CONSOLE.log(f"Training complete. Saving results...")
        gaussians.save_ply(os.path.join(save_dir, f"static_bg.ply"))

        ################### Evaluation ###################
        CONSOLE.log("Saving evaluation...")
        for viewpoint_cam in viewpoint_og:
            with torch.no_grad():
                render_image = render(viewpoint_cam, gaussians, pipe, background)["render"]
                input_img_pkg = viewpoint_cam.get_input_pkg()
                gt_image = input_img_pkg["gt_image"]
                hand_mask = input_img_pkg["hand_mask"]
                obj_mask = input_img_pkg["obj_mask"]
                int_mask = torch.logical_or(hand_mask, obj_mask).int()
                eval_img = get_eval_img([gt_image, render_image, int_mask],
                                        [f"GT img {viewpoint_cam.image_name}", "Render img", "Interaction mask"])
                eval_img.save(os.path.join(eval_dir, f"{viewpoint_cam.image_name}.jpg"))
    # Return the path of saved background
    return (os.path.join(save_dir, f"static_bg.ply"))