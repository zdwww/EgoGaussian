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
import argparse
from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 0
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path) # absolute path
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        # densification interval parameters
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        # statuic training parameters
        self.std_train_iter = 20_000 # use this instead of iterations
        self.entropy_reg_iter = 5000
        self.label_train_iter = 5000
        self.label_lr = 0.001
        # COARSE object pose estimation parameters
        self.obj_translation_lr = 0.0001
        self.obj_rotation_lr = 0.0001
        self.pose_opt_iterations = 9000
        # object pose parameters for loss combination
        self.lambda_Ll1_image = 1.0
        self.lambda_Ll1_alpha = 0.0
        self.lambda_Ll2_alpha = 0.5
        self.lambda_Ldice_alpha = 0.0
        # FINE object pose densification parameters
        self.obj_densify_from_iter = 500
        self.obj_densify_until_iter = 15_000
        self.obj_densification_interval = 100
        self.obj_opacity_reset_interval = 3000
        # FINE tune object parameters
        self.fine_obj_opt_iterations = 10000
        # Train w.r.t contact boundary parameters
        self.cb_train_iter = 30000

        super().__init__(parser, "Optimization Parameters")

class StaticParams(ParamGroup):
    def __init__(self, parser):
        self.s_std_train_iter = 50_000
        self.s_densify_from_iter = 500 
        self.s_densify_until_iter = 40_000
        self.s_entropy_reg_iter = 10_000
        self.s_label_train_iter = 30_000
        self.s_label_lr = 0.001
        self.rewind_frames = 15
        super().__init__(parser, "Training Static Phases Parameters")
    def extract(self, args):
        g = super().extract(args)
        for attr in dir(g):
            if attr.startswith('s_'):
                setattr(g, attr[2:], getattr(g, attr))
        return g

class StaticBgParams(ParamGroup):
    def __init__(self, parser):
        self.b_std_train_iter = 80_000
        self.b_densify_from_iter = 500 
        self.b_densify_until_iter = 60_000
        self.b_entropy_reg_iter = 10_000
        super().__init__(parser, "Training Static Background Parameters")
    def extract(self, args):
        g = super().extract(args)
        for attr in dir(g):
            if attr.startswith('b_'):
                setattr(g, attr[2:], getattr(g, attr))
        return g

class CoarseParams(ParamGroup):
    def __init__(self, parser):
        # Num iteration parameters
        # Pre-training stage
        self.warm_up_iter = 5000 + 15000 # during warm-up, opt pose only
        # Normal training
        self.c_total_num_iter = 20_000 + 10000
        self.c_densify_from_iter = 10000 + 500 + 10000
        self.c_densify_until_iter = 10000 + 5500 + 10000
        self.c_opacity_reset_interval = 2500
        self.c_densification_interval = 500
        # Loss parameters
        self.c_lambda_dssim = 0.1
        self.c_lambda_image = 1.0
        self.c_lambda_Ll1_alpha = 0.0
        self.c_lambda_Ll2_alpha = 0.5
        
        self.c_curr_vpt_prob = 0.4 # probability of using current frame, previously 0.5
        self.c_densify_grad_threshold = 0.1 / 100
        self.c_min_opacity = 0.0025 # used when pruning, previously 0.005
        self.c_save_ply_every = 6
        super().__init__(parser, "Coarse Object Pose Estimation Parameters")
    def extract(self, args):
        g = super().extract(args)
        for attr in dir(g):
            if attr.startswith('c_'):
                setattr(g, attr[2:], getattr(g, attr))
        return g

class FineParams(ParamGroup):
    def __init__(self, parser):
        self.f_total_num_iter = 70_000
        self.f_densify_from_iter = 500
        self.f_densify_until_iter = 45_000
        self.f_opacity_reset_interval = 3000
        self.f_densification_interval = 50 # previously 250
        # Loss parameters
        self.f_lambda_dssim = 0.2
        self.f_lambda_Ll1_image = 1.0
        self.f_lambda_Ll1_alpha = 0.0
        self.f_lambda_Ll2_alpha = 0.2
        self.f_densify_grad_threshold = 0.1 / 100 / 3 # divide by 2 in fine_obj case
        self.f_min_opacity = 0.003 # used when pruning, previously 0.005
        super().__init__(parser, "Fine-tune Object Parameters")
    def extract(self, args):
        g = super().extract(args)
        for attr in dir(g):
            if attr.startswith('f_'):
                setattr(g, attr[2:], getattr(g, attr))
        return g

class FineAllParams(ParamGroup):
    def __init__(self, parser):
        self.a_total_num_iter = 40_000
        self.a_opacity_reset_interval = 3000
        self.a_densify_from_iter = 500
        self.a_densify_until_iter = 5_000
        # Loss parameters
        self.a_lambda_dssim = 0.2
        self.a_lambda_opa_entropy = 0.01
        self.a_densify_grad_threshold = 0.1 / 100
        super().__init__(parser, "Fine-tune All Gaussians Parameters")
    def extract(self, args):
        g = super().extract(args)
        for attr in dir(g):
            if attr.startswith('a_'):
                setattr(g, attr[2:], getattr(g, attr))
        return g

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def tuple_list(string):
    try:
        # Split the string by commas
        parts = string.split(',')
        numbers = []
        for part in parts:
            # Attempt to convert each part to an integer
            try:
                number = int(part)
                numbers.append(number)
            except ValueError:
                # If conversion fails, raise a more specific error
                raise ValueError(f"Each input must be an integer. '{part}' is not an integer.")

        # Ensure we have an even number of elements to form pairs
        if len(numbers) % 2 != 0:
            raise ValueError("The list must contain an even number of elements to form pairs.")

        # Group the numbers into tuples (pairs)
        return [(numbers[i], numbers[i+1]) for i in range(0, len(numbers), 2)]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple list format: {e}")
