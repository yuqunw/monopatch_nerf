import torch
import numpy as np
import re
from struct import pack, unpack
import sys        
from PIL import Image
import torchvision.transforms.functional as F
import json

def read_pfm(pfm_file_path: str)-> torch.Tensor:
    """parses PFM file into torch float tensor

    :param pfm_file_path: path like object that contains full path to the PFM file

    :returns: parsed PFM file of shape CxHxW
    """
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    with open(pfm_file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.ascontiguousarray(np.flip(data, 0))
    return torch.from_numpy(data).view(height, width, -1).permute(2, 0, 1)

def write_pfm(fp, image, scale=1):
    """writes torch.floatarray into PFM file

    :param fp: path like string to write file to
    :param image: numpy binary image that should be of shape HxWx3 or HxW
    :param scale: little / big endian based scale
    """
    color = None
    image = image.detach().cpu().numpy()
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    with open(fp, 'wb') as f:
        header = 'PF\n' if color else 'Pf\n'
        shape = '%d %d\n' % (image.shape[1], image.shape[0])
        f.write(header.encode('utf-8'))
        f.write(shape.encode('utf-8'))

        endian = image.dtype.byteorder

        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale

        scale = '%f\n' % scale
        f.write(scale.encode('utf-8'))

        image_string = image.tostring()
        f.write(image_string)

def read_dmb(fp):
    '''read Gipuma .dmb format image'''

    with open(fp, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_dmb(fp, image):
    '''write Gipuma .dmb format image'''
    
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(fp, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return 

def read_projection(fp):
    pass

def write_projection(fp: str, projection_matrix: np.array):
    with open(fp, "w") as f:
        for i in range(0, 3):
            for j in range(0, 4):
                f.write(str(projection_matrix[i][j]) + ' ')
            f.write('\n')
        f.write('\n')
        f.close()


def read_camera(camera_file_path: str)-> tuple:
    '''Loads camera.txt file given

    Loads camera from path
        Camera format:
            extrinsic
            E11 E12 E13 E14
            E21 E22 E23 E24
            E31 E32 E33 E34
            E41 E42 E43 E44

            intrinsic
            K11 K12 K13
            K21 K22 K23
            K31 K32 K33

            T1 T2 ...

    :param camera_file_path: full path to camera file

    :returns: tuple of intrinsic, extrinsic, min_d, max_d
    '''
    with open(camera_file_path, 'r') as f:
        lines = f.readlines()

        E = np.array([
            [float(v) for v in line.split()] for line in lines[1:5]
        ], dtype=np.float32)

        K = np.array([
            [float(v) for v in line.split()] for line in lines[7:10]
        ], dtype=np.float32)

        tokens = [float(v) for v in lines[11].split()]
        min_d = tokens[0]
        max_d = tokens[-1]
    K = torch.from_numpy(K)
    E = torch.from_numpy(E)
    return K, E, min_d, max_d

def write_camera(fp:str, 
                 intrinsic: np.ndarray, 
                 extrinsic: np.ndarray, 
                 tokens:np.ndarray):
    '''
    Writes camera to path given intrinsics / extrinsics / tokens
        Camera format:
            extrinsic
            E11 E12 E13 E14
            E21 E22 E23 E24
            E31 E32 E33 E34
            E41 E42 E43 E44

            intrinsic
            K11 K12 K13
            K21 K22 K23
            K31 K32 K33

            T1 T2 ...
    Arguments:
        fp(str): path to camera file
        intrinsic(np.ndarray): 3x3 camera intrinsic matrix
        extrinsic(np.ndarray): 3x3 camera intrinsic matrix
        tokens(np.ndarray): Nx1 tokens for extra data
    '''
    k_str= '\n'.join([' '.join([str(v) for v in l]) for l in intrinsic])
    e_str = '\n'.join([' '.join([str(v) for v in l]) for l in extrinsic])
    t_str = ' '.join([str(t) for t in tokens])
    with open(fp, 'w') as f:
        f.write('extrinsic\n')
        f.write(e_str)
        f.write('\n')
        f.write('\n')
        f.write('intrinsic\n')
        f.write(k_str)
        f.write('\n')
        f.write('\n')
        f.write(t_str)

def read_pair_txt(pair_file_path:str)->dict:
    '''
    Parses pair.txt and returns dictionary of neighbors in sorted list
    Pair.txt file format: 
        NUM_IMAGES
        IMAGE_ID_0
        NUM_NEIGHBORS NEIGHBOR_ID_0 SCORE_0 NEIGHBOR_ID_1 SCORE_1 ...
        IMAGE_ID_1
        ...
    '''
    pair_dict = {}
    with open(pair_file_path, 'r') as f:
        lines = f.readlines()
        num_images = int(lines[0])
        for i in range(num_images):
            image_id = int(lines[i * 2 + 1])
            pair_dict[image_id] = []
            image_neighbor_lines = lines[i * 2 + 2]
            tokens = image_neighbor_lines.split()
            num_neighbors = int(tokens[0])
            for n in range(num_neighbors):
                neighbor_id = int(tokens[n * 2 + 1])
                # neighbor_score = float(tokens[n * 2 + 2])
                # if (n < 15) or (neighbor_score > 5):
                pair_dict[image_id].append(neighbor_id)
    return pair_dict

def read_image(image_file_path, device=None)->torch.Tensor:
    with Image.open(image_file_path) as img:
        img = F.to_tensor(img)

    if(device is not None):
        img = img.to(device)
    return img

def write_file(fp, data):
    torch.save(data, fp)

def write_normal(fp, normal):
    F.to_pil_image((normal * 0.5 + 0.5)[0].cpu().permute(2, 0, 1)).save(fp)

def write_depth(fp, depth):
    torch.save(depth, fp)

def write_images(fp, images):
    """ saves images of shape 1xCxHxW """
    F.to_pil_image(images[0].cpu().clamp(0, 1)).save(fp)

def read_file(fp, device=None):
    return torch.load(fp, map_location=device)

def read_normal(fp, device=None):
    with Image.open(fp) as nml:
        normal = F.to_tensor(nml).permute(1, 2, 0).unsqueeze(0) * 2 - 1
    if(device is not None):
        normal = normal.to(device)
    return normal 

def read_depth(fp, device=None):
    return torch.load(fp, map_location=device)

def read_images(fp, device=None):
    with Image.open(fp) as img:
        img = F.to_tensor(img)
    if(device is not None):
        img = img.to(device)
    return img.unsqueeze(0)

def read_pose_file(pose_path):
    with open(pose_path) as f:
        pose = np.array([
            [float(word) for word in line.split()]
            for line in f.readlines()
        ]).reshape(4, 4)
    return pose


def read_info_file(info_file):
    ret = {}
    with open(info_file) as f:
        lines = f.readlines()
        for line in lines:
            words = line.split('=')
            key = words[0].strip()
            val = words[1].strip()
            chars = val.split()
            # check if the chars contain more than 1 numbers
            if (len(chars) > 1) and all([char.isnumeric() for char in chars]):
                val = np.array([float(char) for char in chars]).reshape(4, 4)
                ret[key] = val
            ret[key] = val
    return ret

def write_rec_json(rec_json, camera_attrs, images_attrs):
    '''
    {
        'camera': {
            camera_id: {
                intrinsic: 3x3 array,
                width,
                height
            },
            ...
        },
        'images': {
            image_id: {
                name: ,
                

            }
        },
        'points': {

        }
    }
    '''
    rec_obj = {}
    rec_obj['camera'] = {

    }

    with open(rec_json, 'w') as f:
        json.dump(rec_obj, f)
    return