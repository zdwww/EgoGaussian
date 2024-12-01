import os
import sys
import gc
import glob
import shutil
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, StaticParams, StaticBgParams, CoarseParams, FineParams, FineAllParams
from utils.console import CONSOLE

from trainers.train_static import train_static
from trainers.train_static_bg import train_background
from trainers.coarse_obj_pose import est_coarse_obj_pose
from trainers.fine_obj import fine_tune_obj
from trainers.interpolate_pose import interpolate_pose_seq
from trainers.fine_all import fine_tune_all
from trainers.eval_metric import eval_and_metric

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    static_p = StaticParams(parser)
    staticbg_p = StaticBgParams(parser)
    coarse_p = CoarseParams(parser)
    fine_p = FineParams(parser)
    fine_a_p = FineAllParams(parser)

    parser.add_argument("--out_root", type=str, help="Root directory for output")
    parser.add_argument("--data_type", type=str, help="HOI4D, EK (EPIC-KITCHENS), or Real-world Video")
    parser.add_argument("--video", type=str, help="Video name")
    parser.add_argument("--run_name", type=str, help="Run name")

    args = parser.parse_args(sys.argv[1:])
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    out_dir = os.path.join(args.out_root, args.data_type, args.video, args.run_name)
    exp_name = f"{args.data_type}-{args.video}-{args.run_name}"
    data_name = f"{args.data_type}/{args.video}"
    os.makedirs(out_dir, exist_ok=True)

    CONSOLE.print(f"EXP NAME {exp_name} using data {dataset.source_path}, saved in {out_dir}")
    pipe.compute_cov3D_python = True
    CONSOLE.print(f"Set compute_cov3D_python to {pipe.compute_cov3D_python} for all stages.")

    # Extract training / evaluation split
    train_eval_split = {}
    for split_type in ['training_frames', 'dynamic_eval_frames', 'static_eval_frames']:
        with open(os.path.join(dataset.source_path, 'split', split_type + ".txt"), 'r') as file:
            lines = file.readlines()
        train_eval_split[split_type] = [int(line.strip()) for line in lines]
    CONSOLE.print(f"Train_eval_split: {train_eval_split}")
    # Extract static / dynamic split
    with open(os.path.join(dataset.source_path, 'split', 'phase_frame_index.txt'), 'r') as file:
        phases = [tuple(map(int, line.strip().split(','))) for line in file]
    # Processing the phases
    static_phases = [phases[i] for i in range(len(phases)) if i % 2 == 0]
    dynamic_phases = [phases[i] for i in range(len(phases)) if i % 2 != 0]
    CONSOLE.print(f"Static phases: {static_phases}; Dynamic phases {dynamic_phases}")

    """STAGE 1.0 Static Phase Training"""
    CONSOLE.log("Stage 1.0 Static Training starts...")
    static_phase0_obj_path, pred_mask_path = train_static(dataset, opt, pipe, static_p=static_p.extract(args),
        exp_name=exp_name, 
        save_dir=os.path.join(out_dir,'static'), 
        static_phases=static_phases, 
        train_frames=train_eval_split['training_frames'])
    
    # Uncomment the following line to use the saved results 
    # static_phase0_obj_path, pred_mask_path = os.path.join(out_dir, "static/ply/static_phase0_obj.ply"), os.path.join(out_dir, "static/obj_masks")
    CONSOLE.log(f"Stage 1.0 Static Training ends. Result Object PLY saved at {static_phase0_obj_path}")  
    CONSOLE.log(f"{len(glob.glob(os.path.join(pred_mask_path, '*.png')))} predicted object masks saved at {pred_mask_path}")
    
    """STAGE 1.1 Copy the predicted object masks to complete the partial masks set"""
    shutil.copytree(os.path.join(dataset.source_path, 'obj_masks'), os.path.join(dataset.source_path, 'obj_masks_copy'))
    for filename in sorted(os.listdir(pred_mask_path)):
        if filename.endswith(".png"):
            src_file = os.path.join(pred_mask_path, filename)
            dest_file = os.path.join(dataset.source_path, 'obj_masks', filename)
            if not os.path.exists(dest_file):
                shutil.copy2(src_file, dest_file)
            else:
                print(f"Skipped: {filename}, already exists in obj_masks")
    assert len(glob.glob(os.path.join(dataset.source_path, 'obj_masks', '*.png'))) == len(glob.glob(os.path.join(dataset.source_path, 'images', '*.png')))
    
    gc.collect()
    torch.cuda.empty_cache()

    """STAGE 1.2 Static Background Training"""
    CONSOLE.log("Stage 1.1 Background Training starts...")
    static_bg_path = train_background(dataset, opt, pipe, bg_p=staticbg_p.extract(args), 
        exp_name=exp_name, 
        save_dir=os.path.join(out_dir,'background'), 
        train_frames=train_eval_split['training_frames'],
        dilate_size=5)
    
    # Uncomment the following line to use the saved results 
    # static_bg_path = os.path.join(out_dir, "background/static_bg.ply")
    CONSOLE.log(f"Stage 1.1 Background Training ends. Results saved at {static_bg_path}")
   
    gc.collect()
    torch.cuda.empty_cache()

    """STAGE 2 Coarse Object Pose Estimation"""
    CONSOLE.log("Stage 2 Coarse object pose estimation starts...")
    coarse_pose_seq_path, coarse_obj_path = est_coarse_obj_pose(dataset, opt, pipe, coarse_p=coarse_p.extract(args),
        exp_name=exp_name, 
        save_dir=os.path.join(out_dir,'coarse'),
        obj_gaussians_path=static_phase0_obj_path,
        static_phases=static_phases, 
        dynamic_phases=dynamic_phases,
        train_frames=train_eval_split['training_frames']
        )
    
    # Uncomment the following line to use the saved results 
    # coarse_pose_seq_path, coarse_obj_path = os.path.join(out_dir, "coarse/obj_pose_sequence.pth"), os.path.join(out_dir, "coarse/ply/gaussians_final.ply")
    CONSOLE.log(f"Stage 2 Coarse Pose Estimation ends. Pose saved at {coarse_pose_seq_path}, Object Gaussians saved at {coarse_obj_path}")
    gc.collect()
    torch.cuda.empty_cache()

    """STAGE 3 Fine-tune Object"""
    fine_output_paths = fine_tune_obj(dataset, opt, pipe, fine_p=fine_p.extract(args),
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir,'fine_obj'),
        obj_gaussians_paths = [coarse_obj_path, static_phase0_obj_path], # input two types of Gaussians objects for comparison
        obj_pose_seq_path = coarse_pose_seq_path,
        static_phases = static_phases,
        dynamic_phases = dynamic_phases,
        train_frames = train_eval_split['training_frames']
        )
    
    # Uncomment the following line to use the saved results
    # fine_output_paths = {"from-static": (os.path.join(out_dir, "fine_obj/from-static/obj_pose_sequence.pth"), 
    #     os.path.join(out_dir, "fine_obj/from-static/gaussians_fine.ply")),
    #     "from-coarse": (os.path.join(out_dir, "fine_obj/from-coarse/obj_pose_sequence.pth"), 
    #     os.path.join(out_dir, "fine_obj/from-coarse/gaussians_fine.ply"))}
    CONSOLE.log(f"Stage 3 Fine-tune object ends. Results saved at {fine_output_paths}")
    gc.collect()
    torch.cuda.empty_cache()

    """STAGE 4 Interpolate Pose Sequence"""
    interpolate_pose_seq_path_static = interpolate_pose_seq(dataset, opt, pipe, 
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir,'interpolate_pose_static'),
        dynamic_phases = dynamic_phases,
        obj_pose_seq_path = fine_output_paths["from-static"][0])
    
    interpolate_pose_seq_path_coarse = interpolate_pose_seq(dataset, opt, pipe, 
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir,'interpolate_pose_coarse'),
        dynamic_phases = dynamic_phases,
        obj_pose_seq_path = fine_output_paths["from-coarse"][0])
    
    # Uncomment the following line to use the saved results
    # interpolate_pose_seq_path_static = os.path.join(out_dir, "interpolate_pose_static/obj_pose_sequence.pth")
    # interpolate_pose_seq_path_coarse = os.path.join(out_dir, "interpolate_pose_coarse/obj_pose_sequence.pth")
    CONSOLE.log(f"Stage 5 Object Pose Interpolation ends. Results saved at {interpolate_pose_seq_path_static} and {interpolate_pose_seq_path_coarse}")

    """STAGE 5 Fine-tune Both object and background"""
    all_gaussians_path = fine_tune_all(dataset, opt, pipe, fine_p=fine_a_p.extract(args), 
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir, 'fine_all'),
        obj_gaussians_path = fine_output_paths["from-coarse"][1], 
        bg_gaussians_path = static_bg_path, 
        obj_pose_seq_path = interpolate_pose_seq_path_coarse,
        static_phases = static_phases,
        dynamic_phases = dynamic_phases,
        train_frames=train_eval_split['training_frames'])
    # Uncomment the following line to use the saved results
    # all_gaussians_path = os.path.join(out_dir, 'fine_all', 'gaussians_all.ply')
    CONSOLE.log(f"Stage 4 Fine-tune all Gaussians ends. Results saved at {all_gaussians_path}")
    gc.collect()
    torch.cuda.empty_cache()

    """"STAGE 6 Obtain evaluation metrics"""
    eval_and_metric(dataset, opt, pipe, 
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir,'evaluation'),
        obj_pose_seq_path = interpolate_pose_seq_path_coarse,
        all_gaussians_path = all_gaussians_path, 
        train_eval_split = train_eval_split)