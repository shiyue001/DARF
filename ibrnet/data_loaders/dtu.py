import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from skimage.transform import resize
import sys
sys.path.append('../')
from .data_utils import deepvoxels_parse_intrinsics, get_nearest_pose_ids, rectify_inplane_rotation
import pdb

class DTUDataset(torch.utils.data.Dataset):

    def __init__(self, args, subset,
                 scenes='vase',  # string or list
                 **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'dataset/data/dtu/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.subset = subset  # train / test / validation
        # print(self.subset)
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        if isinstance(scenes, str):
            scenes = [scenes]

        self.scenes = scenes
        self.all_rgb_files = []
        self.all_depth_files = []
        self.all_pose_files = []
        self.all_intrinsics_files = []
        self.all_scenes = []
        self.file_path = os.path.join(self.folder_path, subset)
        for scene in scenes:
            # print(self.folder_path)
            self.scene_path = os.path.join(self.folder_path, subset, scene)
            # self.scene_depth_path = os.path.join('/cluster/work/cvl/yueshi/dataset/data/dtu_depth/', scene)
            # rgb_files = [os.path.join(self.scene_path, 'image', f)
            #              for f in sorted(os.listdir(os.path.join(self.scene_path, 'newdepth', '*.npy')))]
            # print(self.scene_path)

            depth_files = [os.path.join(self.scene_path, 'newdepth', f)
                         for f in sorted(os.listdir(os.path.join(self.scene_path, 'newdepth'))) if '.npy' in f]                   
            # depth_files = [os.path.join(self.scene_depth_path, '/depth', f)
            #         for f in sorted(os.listdir(os.path.join(self.scene_depth_path, 'depth'))) if '.npy' in f]
            if self.subset != 'train':
                depth_files = depth_files[::self.testskip]
            rgb_files = [f.replace('newdepth', 'image').replace('npy', 'png') for f in depth_files]    
            pose_files = [f.replace('image', 'pose').replace('png', 'txt') for f in rgb_files]
            self.intrinsics_file = os.path.join(self.scene_path, 'cameras.npz')
            self.all_rgb_files.extend(rgb_files)
            self.all_depth_files.extend(depth_files)
            self.all_pose_files.extend(pose_files)
            self.all_scenes.extend([scene]*len(pose_files))
            self.all_intrinsics_files.extend([self.intrinsics_file]*len(rgb_files))


    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.all_rgb_files)
        new_scene = self.all_scenes[idx]
        # print(self.scene_old)
        rgb_file = self.all_rgb_files[idx]
        depth_file = self.all_depth_files[idx]
        pose_file = self.all_pose_files[idx]
        intrinsics_file = self.all_intrinsics_files[idx]
        intrinsics = camera_intrinsics(intrinsics_file, self.all_rgb_files)
        # train_rgb_files = sorted(glob.glob(os.path.join(self.scene_path.replace('/scan2/','/'+new_scene+'/'), 'image', '*')))
        # train_rgb_files = train_rgb.format(self.scene_old)
        train_rgb_files = sorted(glob.glob(os.path.join(self.file_path, new_scene, 'image', '*')))                                                                      
        train_poses_files = [f.replace('image', 'pose').replace('png', 'txt') for f in train_rgb_files]
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in train_poses_files], axis=0)
        if self.subset == 'train':
            id_render = train_poses_files.index(pose_file)
            subsample_factor = np.random.choice(np.arange(1, 5))
            num_source_views = self.num_source_views
            # num_source_views = np.random.randint(low=self.num_source_views-2, high=self.num_source_views+2)
        else:
            id_render = -1
            subsample_factor = 1
            num_source_views = self.num_source_views
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        depth = np.load(depth_file).astype(np.float32)
        depth = resize(depth, (rgb.shape[0], rgb.shape[1]))
        render_pose = np.loadtxt(pose_file).reshape(4, 4)

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(num_source_views*subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include target image in the source views
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.subset == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            if self.rectify_inplane_rotation:
                src_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        origin_depth = np.linalg.inv(render_pose.reshape(4, 4))[2, 3] 

        if 'cube' in rgb_file:
            near_depth = origin_depth - 1.
            far_depth = origin_depth + 1
        else:
            near_depth = origin_depth - 0.8
            far_depth = origin_depth + 0.8

        depth_range = torch.tensor([near_depth, far_depth])
        
        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': torch.from_numpy(depth),
                'range': torch.from_numpy(depth), #depth_range, TODO
                }


def camera_intrinsics(intrinsics_file, all_rgb_files):
    # print(intrinsics_file)
    all_cam = np.load(intrinsics_file)
    fx, fy, cx, cy = 0.0, 0.0, 0.0, 0.0
    for i in range(49):
        P = all_cam["world_mat_" + str(i)]
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        fx += torch.tensor(K[0, 0]) * 1.0
        fy += torch.tensor(K[1, 1]) * 1.0
        cx += (torch.tensor(K[0, 2]) + 0.0) * 1.0
        cy += (torch.tensor(K[1, 2]) + 0.0) * 1.0


    fx /= 49
    fy /= 49
    cx /= 49
    cy /= 49
    intrinsics = np.array([[fx, 0., cx, 0.],
                    [0., fy, cy, 0],
                    [0., 0, 1, 0],
                    [0, 0, 0, 1]])  
    return intrinsics        