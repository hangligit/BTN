import torch
import torch.nn as nn
import torchvision
import os
import tqdm
import cv2

import numpy as np
import pickle as pkl
from PIL import Image
from torchvision import transforms


def get_bbox_p(bbox_s, bbox_o):
    y1_s = bbox_s[0]
    y2_s = bbox_s[1]
    x1_s = bbox_s[2]
    x2_s = bbox_s[3]

    y1_o = bbox_o[0]
    y2_o = bbox_o[1]
    x1_o = bbox_o[2]
    x2_o = bbox_o[3]

    y1_p = min(y1_s, y1_o, y2_s, y2_o)
    y2_p = max(y1_s, y1_o, y2_s, y2_o)
    x1_p = min(x1_s, x1_o, x2_s, x2_o)
    x2_p = max(x1_s, x1_o, x2_s, x2_o)

    bbox_p = [y1_p, y2_p, x1_p, x2_p]

    return bbox_p

def extract_region(bgr, bbox, rcnn=False):

    if rcnn:
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
    else:
        y1 = bbox[0]
        y2 = bbox[1]
        x1 = bbox[2]
        x2 = bbox[3]

    region_length_x = x2 - x1
    region_length_y = y2 - y1

    x_scale = 224. / region_length_x
    y_scale = 224. / region_length_y

    p = 16

    padding_x = p / x_scale / 2
    padding_y = p / y_scale / 2

    region_y1 = y1 - padding_y
    region_y2 = y2 + padding_y
    region_x1 = x1 - padding_x
    region_x2 = x2 + padding_x

    if region_y1 < 0:
        region_y1 = 0
    if region_y2 > bgr.shape[0]:
        diff = region_y2 - bgr.shape[0]
        bgr = cv2.copyMakeBorder(bgr, 0, int(diff) + 5, 0, 0, cv2.BORDER_CONSTANT, value=bgr.mean(axis=(0, 1)))
    if region_x1 < 0:
        region_x1 = 0
    if region_x2 > bgr.shape[1]:
        diff = region_x2 - bgr.shape[1]
        bgr = cv2.copyMakeBorder(bgr, 0, 0, 0, int(diff) + 5, cv2.BORDER_CONSTANT, value=bgr.mean(axis=(0, 1)))

    sub_img = bgr[int(region_y1):int(region_y2), int(region_x1):int(region_x2)].copy()

    return sub_img


from pathlib import Path


class VGGRepresentationModel(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = torchvision.models.vgg19(pretrained=True)
        if weights is not None:
            ckpt=torch.load(weights, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            self.model.classifier[6] = nn.Linear(4096, ckpt['classifier.6.weight'].size(0))
            self.model.load_state_dict(ckpt)
            print('load vgg weights size %d'%ckpt['classifier.6.weight'].size(0))


    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self(x)


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_transform_resize = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_process_image(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return data_transform_resize(img.convert('RGB'))


def get_region(box, bgr):
    region = extract_region(bgr, box, rcnn=False)
    region = cv2.resize(region.astype('float32'), (224, 224))
    cv2.imwrite('/tmp/img.jpg', region)
    region = np.array(Image.open(open('/tmp/img.jpg', 'rb')).convert('RGB'))
    return data_transform(region)


def get_region_ext(box, bgr):
    if is_empty_box(box):
        return EMPTYREGION
    else:
        return get_region(box, bgr)


def get_bbox_p_ext(box1, box2):
    if is_empty_box(box1):
        return box2
    elif is_empty_box(box2):
        return box1
    else:
        return get_bbox_p(box1, box2)


def is_empty_box(box):
    return np.equal(box, EMPTYBOX).all()


EMPTYBOX = np.zeros((4), dtype=np.uint16)
EMPTYREGION = torch.zeros((3, 224, 224))


class ImageModel:
    def __init__(self, ckpts):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.vgg_rep_model = VGGRepresentationModel(weights=ckpts)
        self.vgg_rep_model.to(self.device)

    def get(self, img_path, tuples):
        bgr = cv2.imread(img_path)
        full_img = load_process_image(img_path)
        img_rep = self.vgg_rep_model.predict(full_img.unsqueeze_(0).to(self.device)).cpu().data.numpy()[0]

        s, bs, o, bo, p, s_class, o_class = zip(*tuples)
        regions_s = []
        regions_o = []
        regions_p = []

        for i in range(len(bs)):
            bbox_p = get_bbox_p_ext(bs[i], bo[i])
            region_p = get_region_ext(bbox_p, bgr)
            region_s = get_region_ext(bs[i], bgr)
            region_o = get_region_ext(bo[i], bgr)
            regions_s.append(region_s)
            regions_o.append(region_o)
            regions_p.append(region_p)

        s_reps = self.vgg_rep_model.predict(torch.stack(regions_s).to(self.device)).cpu().data.numpy()
        o_reps = self.vgg_rep_model.predict(torch.stack(regions_o).to(self.device)).cpu().data.numpy()
        p_reps = self.vgg_rep_model.predict(torch.stack(regions_p).to(self.device)).cpu().data.numpy()
        return img_rep, s_reps, o_reps, p_reps


def checkout_pickles(annotation_refactor, image_folder, out_dir, ckpts):
    image_model = ImageModel(ckpts)

    os.makedirs(out_dir, exist_ok=False)

    scene_dict = annotation_refactor['scenes']

    global_count = 0
    for image_count, image_name in enumerate(
            tqdm.tqdm(annotation_refactor['filenames'], total=len(annotation_refactor['filenames']))):

        annotation = scene_dict[image_count]
        if annotation['tuples'] == []:
            continue

        tuples = annotation['tuples']
        img_path = image_folder + '/' + image_name
        img_rep, s_reps, o_reps, p_reps = image_model.get(img_path, tuples)

        for idx in range(len(tuples)):
            s, sb, o, ob, p, sn, on = tuples[idx]
            s_rep, o_rep, p_rep = s_reps[idx], o_reps[idx], p_reps[idx]

            sample = [img_rep, s_rep, o_rep, p_rep, image_count, s, o, p, sn, on]
            pkl.dump(sample, open(f'{out_dir}/{global_count:06d}', 'wb'))
            global_count += 1


if __name__=='__main__':
    import sys
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--annotation_file', type=str, help='path to the annotation file')
    parser.add_argument('--image_folder', type=str, help='path to the images folder')
    parser.add_argument('--output_folder', type=str, help='path to the output folder of feature maps')
    parser.add_argument('--extractor', type=str, help='path to the pretrained feature extractor')
    args=parser.parse_args()

    annotation_file=args.annotation_file
    image_folder=args.image_folder
    output_folder=args.output_folder
    extractor=args.extractor

    annotation_refactor = pkl.load(open(annotation_file,'rb'))
    ckpts = str(Path.home()) + '/data/model_best_68_106.pth'
    checkout_pickles(annotation_refactor=annotation_refactor, image_folder=image_folder, out_dir=output_folder, ckpts=extractor)
