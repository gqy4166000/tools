
import pickle as pkl
import numpy as np
from pycocotools.coco import COCO
import time
import random
import json
import pprint
import argparse

parser = argparse.ArgumentParser(description='Generate SimpleDet GroundTruth Database')
parser.add_argument('--json_path', default="instances_minival2014.json" ,help='the path of json', type=str)
parser.add_argument('--num',default=100, help='the num of split pictures', type=int)

args = parser.parse_args()
SPLITNUM = args.num
mydata = {
        'licenses': [],
        'info': {},
        'categories': [],   # Required
        'images': [],       # Required
        'annotations': []   # Required
    }
def getcoco(path):
    # path = "minitrain_10.json"
    dataset = COCO(path)
    img_ids = dataset.getImgIds()
    choice_ids = random.sample(img_ids,SPLITNUM)
    cat_info = dataset.getCatIds()
    cats = dataset.loadCats(cat_info)
    mydata['categories'] = cats
    for ids in choice_ids:
        img_info = dataset.loadImgs(ids)[0]
        mydata['images'].append(img_info)
        ins_anno_ids = dataset.getAnnIds(imgIds=ids, iscrowd=False)
        ann_info = dataset.loadAnns(ins_anno_ids)
        mydata['annotations']+=ann_info

    output_file = "minitrain_%s.json" %(SPLITNUM)
    with open(output_file, 'w') as ofile:
        json.dump(mydata, ofile, sort_keys=True, indent=2)

def test(path):
    path = "minitrain_10.json"
    dataset = COCO(path)
    img_ids = dataset.getImgIds()
    cat_info = dataset.getCatIds()
    cats = dataset.loadCats(cat_info)
    pprint.pprint(cats)
    for ids in img_ids:
        img_info = dataset.loadImgs(ids)[0]
        pprint.pprint(img_info)
        ins_anno_ids = dataset.getAnnIds(imgIds=ids, iscrowd=False)
        ann_info = dataset.loadAnns(ins_anno_ids)
        pprint.pprint(ann_info)
        time.sleep(10)

if __name__ == "__main__":
    getcoco(args.json_path)
    # test(args.json_path)
    