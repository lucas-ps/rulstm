import os
import sys
import torch
from torch import nn
from pretrainedmodels import bninception
from torchvision import transforms
from glob import glob
from PIL import Image
import lmdb
from tqdm import tqdm
from os.path import basename
from argparse import ArgumentParser
import cv2
from multiprocessing import Pool

# eg. python extract_rgb.py --dataset_path "/media/lucas/Linux SSD/50-Salads/" --dataset 50-Salads --generate_jpgs --extract_features
parser = ArgumentParser(description='RGB Feature Extraction')
parser.add_argument('--dataset_path', type=str, required=False,
                    help='Path to the dataset.')
parser.add_argument('--dataset', type=str, choices=['50-Salads', 'Breakfast', 'Breakfast1'], required=True,
                    help='Dataset to be used (50-Salads/Breakfast).')
parser.add_argument('--generate_jpgs', action='store_true', default=False,
                    help='Flag to generate frame JPGs.')
parser.add_argument('--extract_features', action='store_true', default=False,
                    help='Flag to extract features.')

args = parser.parse_args()

def process_video(video_path):
    # Extract directory name and filename from the video path
    relative_path = os.path.relpath(video_path, args.dataset_path)
    dir_structure = os.path.dirname(relative_path)
    dir_name = dir_structure.replace(os.path.sep, '_')
    filename = os.path.splitext(os.path.basename(video_path))[0]
    
    file_output_folder = output_folder + "/" + filename

    if not os.path.exists(file_output_folder):
        os.makedirs(file_output_folder)


    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened(): 
        print(f"Error opening video file {video_path}")
        return
    success, image = vidcap.read()
    if not success or image is None:
        print(f"Error reading frame from video file {video_path}")
        return
    count = 0
    while success:
        image = cv2.resize(image, (454, 256))
        padded_count = str(count).zfill(10)

        # Generate features with slightly different names as FUTR expects
        if args.dataset == 'Breakfast1':
            frame_name = f"{dir_name}_{filename}_frame_{padded_count}.jpg"
        else:
            frame_name = f"{filename}_frame_{padded_count}.jpg"
        #print(frame_name)
        cv2.imwrite(os.path.join(file_output_folder, frame_name), image)     

        cv2.imwrite(os.path.join(file_output_folder, f"{filename}_frame_{padded_count}.jpg"), image)     
        success, image = vidcap.read()
        count += 1


if args.generate_jpgs:
    if args.dataset_path is None:
        print("No dataset path provided. Please provide this using the --dataset_path argument")
    else:
        files = glob(os.path.join(args.dataset_path, '**', '*.avi'), recursive=True)
        output_folder = "./frames/"+args.dataset+"_frames/"

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Modify according to number of cpu threads
        with Pool(processes=16) as p:
            list(tqdm(p.imap(process_video, files), total=len(files), desc='Generating frames'))

# eg. python extract_rgb.py --dataset Breakfast --extract_features
# if args.extract_features:
#     imgs = None
#     if args.dataset == "50-Salads":
#         env = lmdb.open('features/50-Salads/rgb', map_size=1099511627776)
#         imgs = sorted(glob('./frames/50-Salads_frames/**/*.jpg'))
#     else:
#         env = lmdb.open('features/Breakfast/rgb', map_size=1099511627776)
#         imgs = sorted(glob('./frames/Breakfast1_frames/**/*.jpg'))

#     if not imgs:
#         print("No images found. Use the --generate_jpgs argument to generate these.")
    
#     else:
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'

#         model = bninception(pretrained=None)
#         state_dict = torch.load('models/TSN-rgb.pth.tar')['state_dict']
#         state_dict = {k.replace('module.base_model.','') : v for k,v in state_dict.items()}
#         model.load_state_dict(state_dict, strict=False)


#         model.last_linear = nn.Identity()
#         model.global_pool = nn.AdaptiveAvgPool2d(1)

#         model.to(device)

#         transform = transforms.Compose([
#             transforms.Resize([256, 454]),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x[[2,1,0],...]*255), #to BGR
#             transforms.Normalize(mean=[104, 117, 128],
#                                 std=[1, 1, 1]),
#         ])

#         model.eval()
#         for im in tqdm(imgs,'Extracting features'):
#             key = basename(im)
#             img = Image.open(im)
#             data = transform(img).unsqueeze(0).to(device)
#             feat = model(data).squeeze().detach().cpu().numpy()
#             with env.begin(write=True) as txn:
#                 txn.put(key.encode(),feat.tobytes())

if args.extract_features:
    imgs = None
    if args.dataset == "50-Salads":
        env = lmdb.open('features/50-Salads/rgb', map_size=1099511627776)
        imgs = sorted(glob('./frames/50-Salads_frames/**/*.jpg'))
    else:
        env = lmdb.open('features/Breakfast1/rgb', map_size=1099511627776)
        imgs = sorted(glob('./frames/Breakfast1_frames/**/*.jpg'))

    if not imgs:
        print("No images found. Use the --generate_jpgs argument to generate these.")
    
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model = bninception(pretrained=None)
        state_dict = torch.load('models/TSN-rgb.pth.tar')['state_dict']
        state_dict = {k.replace('module.base_model.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

        model.last_linear = nn.Identity()
        model.global_pool = nn.AdaptiveAvgPool2d(1)
        model.to(device)

        transform = transforms.Compose([
            transforms.Resize([256, 454]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255),  # to BGR
            transforms.Normalize(mean=[104, 117, 128],
                                std=[1, 1, 1]),
        ])

        model.eval()

        BATCH_SIZE = 128 # Adjust this based on GPU memory
        num_batches = len(imgs) // BATCH_SIZE + (1 if len(imgs) % BATCH_SIZE != 0 else 0)

        for batch_idx in tqdm(range(num_batches), desc='Extracting features'):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = (batch_idx + 1) * BATCH_SIZE
            batch_imgs = imgs[batch_start:batch_end]

            batch_data = []
            keys = []

            for im in batch_imgs:
                key = basename(im)
                img = Image.open(im)
                data = transform(img).unsqueeze(0)
                batch_data.append(data)
                keys.append(key)

            batch_data = torch.cat(batch_data).to(device)
            batch_features = model(batch_data).squeeze().detach().cpu().numpy()

            with env.begin(write=True) as txn:
                for key, feat in zip(keys, batch_features):
                    txn.put(key.encode(), feat.tobytes())