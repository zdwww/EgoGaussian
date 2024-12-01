import os
import sys
from argparse import ArgumentParser


from arguments import ModelParams, PipelineParams, OptimizationParams, StaticParams, StaticBgParams, CoarseParams, FineParams, FineAllParams
from utils.console import CONSOLE

from trainers.eval_metric import eval_and_metric, render_singleview_w_new_pose

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
    for split_type in ['training_frames']: # , 'dynamic_eval_frames', 'static_eval_frames']:
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
    static_phase0_obj_path = os.path.join(out_dir, "static/ply/static_phase0_obj.ply")
    CONSOLE.log(f"Stage 1.0 Static Training ends. Result Object PLY saved at {static_phase0_obj_path}")
    
    """STAGE 1.1 Static Background Training"""
    static_bg_path = os.path.join(out_dir, "background/static_bg.ply")
    CONSOLE.log(f"Stage 1.1 Background Training ends. Results saved at {static_bg_path}")

    """STAGE 2 Coarse Object Pose Estimation"""
    coarse_pose_seq_path = os.path.join(out_dir, "coarse/obj_pose_sequence.pth")
    coarse_obj_path = os.path.join(out_dir, "coarse/ply/gaussians_final.ply")
    CONSOLE.log(f"Stage 2 Coarse Pose Estimation ends. Pose saved at {coarse_pose_seq_path}, Object Gaussians saved at {coarse_obj_path}")

    """STAGE 3 Fine-tune Object"""
    fine_output_paths = {"from-static": (os.path.join(out_dir, "fine_obj/from-static/obj_pose_sequence.pth"), 
        os.path.join(out_dir, "fine_obj/from-static/gaussians_fine.ply")),
        "from-coarse": (os.path.join(out_dir, "fine_obj/from-coarse/obj_pose_sequence.pth"), 
        os.path.join(out_dir, "fine_obj/from-coarse/gaussians_fine.ply"))}
    CONSOLE.log(f"Stage 3 Fine-tune object ends. Results saved at {fine_output_paths}")

    """STAGE 4 Interpolate Pose Sequence"""
    interpolate_pose_seq_path_static = os.path.join(out_dir, "interpolate_pose_static/obj_pose_sequence.pth")
    interpolate_pose_seq_path_coarse = os.path.join(out_dir, "interpolate_pose_coarse/obj_pose_sequence.pth")
    CONSOLE.log(f"Stage 5 Object Pose Interpolation ends. Results saved at {interpolate_pose_seq_path_static} and {interpolate_pose_seq_path_coarse}")

    """STAGE 5 Fine-tune Both object and background"""
    all_gaussians_path = os.path.join(out_dir, 'fine_all', 'gaussians_all.ply')
    CONSOLE.log(f"Stage 4 Fine-tune all Gaussians ends. Results saved at {all_gaussians_path}")

    """"STAGE 6 Obtain evaluation metrics"""
    eval_and_metric(dataset, opt, pipe, 
        exp_name = exp_name, 
        save_dir = os.path.join(out_dir,'evaluation'),
        obj_pose_seq_path = interpolate_pose_seq_path_coarse,
        all_gaussians_path = all_gaussians_path, 
        train_eval_split = train_eval_split)

    """"For additional visualization demonstrated on the webpage"""
    render_singleview_w_new_pose(dataset, pipe,
        save_dir = os.path.join(out_dir, 'vis'),
        obj_pose_seq_path = interpolate_pose_seq_path_coarse,
        all_gaussians_path = all_gaussians_path,
        new_pose = [0.97,-0.25,0.04,0,0.24,0.94,0.25,0,-0.11,-0.23,0.97,0,-0.69,1.22,9.02,1])

    # render_trajectory(dataset, pipe,
    #     save_dir = os.path.join(out_dir, 'vis'),
    #     obj_pose_seq_path = interpolate_pose_seq_path_coarse,
    #     all_gaussians_path = all_gaussians_path,
    #     obj_gaussians_path = fine_output_paths["from-coarse"][1],
    #     new_pose = [0.88,-0.41,0.24,0,0.46,0.87,-0.17,0,-0.14,0.26,0.96,0,1.3,-1.28,8.48,1])

    # render_freeiview(dataset, pipe,
    #     save_dir = os.path.join(out_dir, 'vis'), 
    #     obj_pose_seq_path = interpolate_pose_seq_path_coarse, 
    #     all_gaussians_path = all_gaussians_path, 
    #     new_poses = [[0.89,0.46,-0.03,0,-0.43,0.86,0.27,0,0.15,-0.23,0.96,0,6.23,1.59,13.64,1]])