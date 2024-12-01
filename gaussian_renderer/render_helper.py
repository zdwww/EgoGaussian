import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer as Renderer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def get_raster_settings(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0):
    """
    Helper function to get raster settings instead of performing rendering
    """
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )
    return raster_settings

def get_pts_label_as_rgb(gaussians_label : torch.Tensor):
    label_as_rgb = torch.zeros([gaussians_label.shape[0], 3]).float().cuda()
    label_as_rgb[:, 0] = gaussians_label.squeeze()
    label_as_rgb[:, 1] = gaussians_label.squeeze()
    label_as_rgb[:, 2] = gaussians_label.squeeze()
    return label_as_rgb

def gaussians_to_label_rendervar(gaussians : GaussianModel):
    means3D = gaussians.get_xyz.detach()
    means2D = torch.zeros_like(gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda") + 0
    opacity = gaussians.get_opacity.detach()
    scales = gaussians.get_scaling.detach()
    rotations = gaussians.get_rotation.detach()
    colors_precomp = get_pts_label_as_rgb(gaussians.get_label)

    rendervar = {
        'means3D': means3D,
        'colors_precomp': colors_precomp,
        'rotations': rotations,
        'opacities': opacity,
        'scales': scales,
        'means2D': means2D
    }
    return rendervar

def get_render_label(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor):
    """
    Helper function to get rendered label with another renderer initialized inside the function
    """
    curr_raster_settings = get_raster_settings(viewpoint_camera, pc, bg_color)
    label_renderer = Renderer(curr_raster_settings)
    label_rendervar = gaussians_to_label_rendervar(pc)
    render_label, _, _, _ = label_renderer(**label_rendervar)
    return render_label


