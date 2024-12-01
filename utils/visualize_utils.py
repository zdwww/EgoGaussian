from utils.dynamic_utils import *
from PIL import Image, ImageDraw, ImageFont

def vis_input_and_render(viewpoint_cam, render_img_pkg, render_label, font_size=25):
    gt_image = rgb_tensor_to_PIL(viewpoint_cam.gt_image)
    int_mask = gray_tensor_to_PIL(viewpoint_cam.int_mask)
    obj_mask = gray_tensor_to_PIL(viewpoint_cam.obj_mask)
    est_depth = gray_tensor_to_PIL(scale_img_to_0_1(viewpoint_cam.est_depth))

    render_image = rgb_tensor_to_PIL(render_img_pkg["render"])
    render_depth = gray_tensor_to_PIL(scale_img_to_0_1(render_img_pkg["depth"]))
    render_alpha = gray_tensor_to_PIL(render_img_pkg["alpha"])
    render_label = rgb_tensor_to_PIL(render_label)

    image_size = gt_image.size
    font = ImageFont.truetype("Serif.ttf", font_size)
    x_position = image_size[0] / 2
    y_position = 0 + font_size

    I1 = ImageDraw.Draw(gt_image)
    I2 = ImageDraw.Draw(int_mask)
    I3 = ImageDraw.Draw(obj_mask)
    I4 = ImageDraw.Draw(est_depth)

    I5 = ImageDraw.Draw(render_image)
    I6 = ImageDraw.Draw(render_depth)
    I7 = ImageDraw.Draw(render_alpha)
    I8 = ImageDraw.Draw(render_label)

    I1.text((x_position, y_position), f"Input GT Image {viewpoint_cam.image_name}", fill="white", anchor="ms", font=font)
    I2.text((x_position, y_position), "Input Interaction Mask", fill="white", anchor="ms", font=font)
    I3.text((x_position, y_position), "Input Object Mask", fill="white", anchor="ms", font=font)
    I4.text((x_position, y_position), "Input Estimated Depth", fill="black", anchor="ms", font=font)

    I5.text((x_position, y_position), f"Rendered Image {viewpoint_cam.image_name}", fill="white", anchor="ms", font=font)
    I6.text((x_position, y_position), "Rendered Depth", fill="black", anchor="ms", font=font)
    I7.text((x_position, y_position), "Rendered Alpha", fill="black", anchor="ms", font=font)
    I8.text((x_position, y_position), "Rendered Label", fill="white", anchor="ms", font=font)

    # Create a new image to hold the combined result
    images = [gt_image, int_mask, obj_mask, est_depth, render_image, render_alpha, render_label, render_depth]
    new_image_width = image_size[0] * 4
    new_image_height = image_size[1] * 2
    new_image = Image.new('RGB', (new_image_width, new_image_height))

    # Paste each image onto the combined image
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (image_size[0], 0))
    new_image.paste(images[2], (image_size[0] * 2, 0))
    new_image.paste(images[3], (image_size[0] * 3, 0))
    new_image.paste(images[4], (0, image_size[1]))  # Blank space
    new_image.paste(images[5], (image_size[0], image_size[1]))
    new_image.paste(images[6], (image_size[0] * 2, image_size[1]))
    new_image.paste(images[7], (image_size[0] * 3, image_size[1]))
    return new_image