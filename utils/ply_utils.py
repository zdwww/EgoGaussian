import os
import numpy as np
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def color_by_label(plydata, out_path):
    label = plydata['vertex']['label']
    old_vertex_data = plydata['vertex'].data
    new_properties = old_vertex_data.dtype.descr[:6] + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.zeros(old_vertex_data.shape[0], dtype=new_properties)
    for prop in old_vertex_data.dtype.names[:6]:
        new_vertex_data[prop] = old_vertex_data[prop]
    new_vertex_data['red'] = 0  
    new_vertex_data['green'] = 0  
    new_vertex_data['blue'] = 0  
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    plydata.elements = [new_vertex_element if e.name == 'vertex' else e for e in plydata.elements]
    plydata['vertex']['red'] = ((sigmoid(label) > 0.5) * 255).astype(np.uint8)
    plydata['vertex']['green'] = np.full(plydata['vertex']['green'].shape, 0,  dtype=np.uint8)
    plydata['vertex']['blue'] = np.full(plydata['vertex']['blue'].shape, 0,  dtype=np.uint8)
    plydata.write(out_path)

def color_by_generation(plydata, out_path):
    label = plydata['vertex']['generation']
    old_vertex_data = plydata['vertex'].data
    new_properties = old_vertex_data.dtype.descr[:6] + [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    new_vertex_data = np.zeros(old_vertex_data.shape[0], dtype=new_properties)
    for prop in old_vertex_data.dtype.names[:6]:
        new_vertex_data[prop] = old_vertex_data[prop]
    new_vertex_data['red'] = 0  
    new_vertex_data['green'] = 0  
    new_vertex_data['blue'] = 0  
    new_vertex_element = PlyElement.describe(new_vertex_data, 'vertex')
    plydata.elements = [new_vertex_element if e.name == 'vertex' else e for e in plydata.elements]
    plydata['vertex']['red'] = ((label == label.max()) * 255 / 2).astype(np.uint8)
    plydata['vertex']['green'] = np.full(plydata['vertex']['green'].shape, 0,  dtype=np.uint8)
    plydata['vertex']['blue'] = np.full(plydata['vertex']['blue'].shape, 0,  dtype=np.uint8)
    plydata.write(out_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="PLY format 3DGS output handler")
    parser.add_argument('--ply_path', type=str, default=None, help='Path to the .ply file')
    parser.add_argument('--out_path', type=str, default=None, help='Output path to write')
    parser.add_argument('--method', choices=['color_by_label', 'color_by_generation', 'separate_by_color'], 
                        help='Choose method to apply to .ply')
    args = parser.parse_args()

    with open(args.ply_path, 'rb') as f:
        plydata = PlyData.read(f, mmap=False)

    if args.out_path is None:
        args.out_path = os.path.dirname(args.ply_path)
    
    if args.method == "color_by_label":
        out_path = os.path.join(args.out_path, 
                                os.path.basename(args.ply_path).split('.')[0]+'_lab_color.ply')
        color_by_label(plydata, out_path)
    elif args.method == "color_by_generation":
        out_path = os.path.join(args.out_path, 
                                os.path.basename(args.ply_path).split('.')[0]+'_gen_color.ply')
        color_by_generation(plydata, out_path)
    else:
        pass
