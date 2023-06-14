import pickle as pkl
import argparse

# script to create a submission template


parser=argparse.ArgumentParser()
parser.add_argument('annotation_file', type=str)
args=parser.parse_args()

annotation_file=pkl.load(open(args.annotation_file,'rb'))

template=dict()
for k, v in annotation_file['entities'].items():
    b_class,p_class,g_class,age,color,activity,risk,entity=v['label']
    template[k]=dict(
        b_class_rank=0,
        p_class_rank=0,
        g_class_rank=0,
        age_rank=0,
        color_rank=0,
        activity_rank=0,
        risk_rank=0,
        entity_rank=0,
    )

pkl.dump(template, open('template.pkl','wb'))
