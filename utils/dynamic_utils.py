import os
import torch
import numpy as np
from PIL import Image
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim

def find_idx_by_image_name(viewpoint_stack, image_name):
    for index, viewpoint in enumerate(viewpoint_stack):
        if int(getattr(viewpoint, 'image_name')) == int(image_name):
            return index
    raise ValueError(f"No frame named {image_name} found in dataset.")

def get_viewpoint_split(viewpoint_stack, train_frames=None, static_phases=None, dynamic_phases=None):
    if train_frames is None:
        print("Training frames index is None, use all frames")
    viewpoint_stack = sorted(viewpoint_stack, key=lambda item: int(item.image_name))
    if train_frames is not None:
        train_frames = [int(image_name) for image_name in train_frames]
    viewpoint_dict = {}
    if static_phases is not None:
        static_viewpts_lst = []
        for phase in static_phases:
            start_idx = find_idx_by_image_name(viewpoint_stack, phase[0])
            end_idx = find_idx_by_image_name(viewpoint_stack, phase[1])
            if train_frames is not None:
                curr_viewpts = [cam for cam in viewpoint_stack[start_idx: end_idx+1] 
                    if int(cam.image_name) in train_frames]
            else:
                curr_viewpts = [cam for cam in viewpoint_stack[start_idx: end_idx+1]]
            static_viewpts_lst.append(curr_viewpts)
        viewpoint_dict["static_phases"] = static_viewpts_lst
    if dynamic_phases is not None:
        dynamic_viewpts_lst = []
        for phase in dynamic_phases:
            start_idx = find_idx_by_image_name(viewpoint_stack, phase[0])
            end_idx = find_idx_by_image_name(viewpoint_stack, phase[1])
            if train_frames is not None:
                curr_viewpts = [cam for cam in viewpoint_stack[start_idx: end_idx+1] 
                    if int(cam.image_name) in train_frames]
            else:
                curr_viewpts = [cam for cam in viewpoint_stack[start_idx: end_idx+1]]
            dynamic_viewpts_lst.append(curr_viewpts)
        viewpoint_dict["dynamic_phases"] = dynamic_viewpts_lst
    return viewpoint_dict

def rgb_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((np.transpose(torch.clamp(tensor.detach().cpu(), 0, 1).numpy(), (1, 2, 0)) * 255.0).astype(np.uint8))

def gray_tensor_to_PIL(tensor : torch.Tensor):
    return Image.fromarray((torch.clamp(tensor.detach().cpu(), 0, 1).numpy().squeeze() * 255.0).astype(np.uint8))

def scale_img_to_0_1(img_tensor):
    img_tensor_min = img_tensor.min()
    img_tensor_max = img_tensor.max()
    out = (img_tensor - img_tensor_min) / (img_tensor_max - img_tensor_min)
    return out

def standardize_img(img_tensor, eps=1e-10):
    img_tensor = img_tensor.float()
    return (img_tensor - torch.mean(img_tensor)) / (torch.std(img_tensor) + eps)

def get_eval_img(tensor_lst, txt_lst):
    """
    A helper function to return the combined evaluation image with text on it
    Parameters:
    tensor_lst (list of torch tensors)
    txt_lst (list of string)
    """
    assert len(tensor_lst) == len(txt_lst)
    assert all(tensor.shape[1:] == tensor_lst[0].shape[1:] for tensor in tensor_lst)
    # img_lst = [rgb_tensor_to_PIL(tensor) for tensor in tensor_lst]
    img_lst = [rgb_tensor_to_PIL(tensor) if tensor.shape[0] == 3 else gray_tensor_to_PIL(tensor) for tensor in tensor_lst]
    img_size = img_lst[0].size
    font = ImageFont.truetype("Serif.ttf", 25)
    x_position = img_size[0] / 2
    y_position = 0 + 25
    draw_lst = [ImageDraw.Draw(img) for img in img_lst]
    for i, draw in enumerate(draw_lst):
        draw.text((x_position, y_position), txt_lst[i], fill="white", anchor="ms", font=font)

    new_image_width = img_size[0] * len(tensor_lst)
    new_image_height = img_size[1] * 1
    new_image = Image.new('RGB', (new_image_width, new_image_height))
    for i, img in enumerate(img_lst):
        new_image.paste(img, (img_size[0]*i, 0))
    return new_image

def get_eval_img_new(tensor_lst, txt_lst=None, rows=1, cols=1):
    """
    A helper function to return the combined evaluation image with text on it
    Parameters:
    tensor_lst (list of torch tensors)
    txt_lst (list of string)
    """
    # assert all(tensor.shape[1:] == tensor_lst[0].shape[1:] for tensor in tensor_lst)
    # img_lst = [rgb_tensor_to_PIL(tensor) for tensor in tensor_lst]
    img_lst = [rgb_tensor_to_PIL(tensor) if tensor.shape[0] == 3 else gray_tensor_to_PIL(tensor) for tensor in tensor_lst]
    img_size = img_lst[0].size
    font = ImageFont.truetype("Serif.ttf", 25)
    x_position = img_size[0] / 2
    y_position = 0 + 25
    draw_lst = [ImageDraw.Draw(img) for img in img_lst]
    if txt_lst is not None:
        assert len(tensor_lst) == len(txt_lst)
        for i, draw in enumerate(draw_lst):
            draw.text((x_position, y_position), txt_lst[i], fill="white", anchor="ms", font=font)

    new_image_width = img_size[0] * cols
    new_image_height = img_size[1] * rows
    new_image = Image.new('RGB', (new_image_width, new_image_height), "black")
    for i, img in enumerate(img_lst):
        if i >= rows * cols:
            break  # Stop if we have more images than slots
        row = i // cols
        col = i % cols
        new_image.paste(img, (img_size[0] * col, img_size[1] * row))
    return new_image

