from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import MultiTaskDataset
from model import BTN
from config import cfg_dict
from scipy.io.matlab import loadmat, savemat
import os
import pickle as pkl
import argparse
import sys
import logging
logger = logging.getLogger('eval')
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def load_weights(model, filepath):
    logger.info(model.load_state_dict(torch.load(filepath, map_location=next(model.parameters()).device),strict=True))

def evaluation(model, dataloader, mode, config, device, logger, sub_bboxes_ours, obj_bboxes_ours, image_index, out_root_dir):

    model.eval()

    collector = dict(
        subjects=[],
        objects=[],
        predicates=[]
    )
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = [x.to(device) for x in inputs]
            targets = [x.to(device) for x in targets]
            outputs, samples = model.predict(inputs, config.test_task, config.test_sampling_mode)

            assert len(outputs)==6
            assert len(targets)==6
            subjects = outputs[2][0]
            objects = outputs[4][0]
            predicates = outputs[5]

            if mode=='predicate':
                subjects = targets[2][:,0]
                objects = targets[4][:,0]
                if subjects.ndim==1:
                    subjects = int_to_one_hot(subjects, 100)
                    objects = int_to_one_hot(objects, 100)

            if subjects.shape[1]>100:
                subjects = subjects[:,:100]
                objects = objects[:,:100]
            if predicates.shape[1]>70:
                predicates = predicates[:,:70]

            collector['subjects'].append(subjects.data.cpu().numpy())
            collector['objects'].append(objects.data.cpu().numpy())
            collector['predicates'].append(predicates.data.cpu().numpy())

    for k,v in collector.items():
        collector[k] = np.concatenate(v,0)


    predicted_s=collector['subjects']
    predicted_o=collector['objects']
    predicted_p=collector['predicates']

    rlp_confs_ours = [[]] * 1000
    rlp_labels_ours = [[]] * 1000
    for i in range(1000): rlp_labels_ours[i] = []
    for i in range(1000): rlp_confs_ours[i] = []

    for i in range(len(image_index)):

        s = np.argmax(predicted_s[i])
        o = np.argmax(predicted_o[i])
        full_P = np.copy(predicted_p[i]) # * P[:,s,o]

        for top_k in range(0, 70):
            p = np.argmax(full_P)
            rlpScore = predicted_s[i, s] * predicted_o[i, o] * full_P[p]

            rlp_labels_ours[image_index[i]].append((float(s + 1), float(p + 1), float(o + 1)))
            rlp_confs_ours[image_index[i]].append(rlpScore)
            full_P[p] = -1

    save_name = out_root_dir + '/result_vkg_' + str(mode)
    savemat(save_name, {'rlp_labels_ours': rlp_labels_ours,
                        'rlp_confs_ours': rlp_confs_ours,
                        'sub_bboxes_ours': sub_bboxes_ours,
                        'obj_bboxes_ours': obj_bboxes_ours})

def int_to_one_hot(arr, num_classes):
    assert arr.ndim <= 1
    return (torch.arange(num_classes,device=arr.device).reshape(1, num_classes) == arr.reshape(-1, 1)).float()

def main(config, outdir, save_name):
    
    out_root_dir = config.HOME + '/Visual_Relation_Cog/evaluation/' + save_name + '/'
    if not os.path.isdir(out_root_dir):
        os.makedirs(out_root_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = BTN(config)
    model.to(device)
    load_weights(model, config.test_weights)

    for mode in ['predicate', 'classification', 'phrase']:

        if mode == 'phrase':
            sub_bboxes_ours = pkl.load(open(config.HOME + '/data/sub_bboxes_ours_phrase_70', 'rb'), encoding='latin1')
            obj_bboxes_ours = pkl.load(open(config.HOME + '/data/obj_bboxes_ours_phrase_70', 'rb'), encoding='latin1')
            image_index = pkl.load(open(config.HOME + '/data/image_index_phrase', 'rb'), encoding='latin1')
            test_root = config.HOME+'/data/VRD/test/phrase'
        else:
            sub_bboxes_ours = pkl.load(open(config.HOME + '/data/sub_bboxes_ours_pred_70', 'rb'), encoding='latin1')
            obj_bboxes_ours = pkl.load(open(config.HOME + '/data/obj_bboxes_ours_pred_70', 'rb'), encoding='latin1')
            image_index = pkl.load(open(config.HOME + '/data/image_index_pred', 'rb'), encoding='latin1')
            test_root = config.HOME+'/data/VRD/test/predicate'
            
        dataset = MultiTaskDataset(test_root, task=config.test_task, mode=config.test_data_mode, is_rel=True)
        dataloader = DataLoader(dataset, batch_size=config.batch_size_test)

        evaluation(model, dataloader, mode, config, device, logger, sub_bboxes_ours, obj_bboxes_ours, image_index, out_root_dir)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args=parser.parse_args()
    
    config = cfg_dict[args.config]()

    outdir=os.path.dirname(config.test_weights)
    
    save_name = os.path.basename(outdir)
    
    config._load_config(outdir)
    
    logfile = logging.FileHandler(os.path.join(outdir, "sgd.txt"))
    logger.addHandler(logfile)
    logger.info("==================== New Run ==================")

    for k,v in config._show_config().items():            
        logger.info(k + ': %s', v)

    main(config, outdir, save_name)
