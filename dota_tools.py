import numpy as np
import os
import cv2
from tqdm import tqdm
from ResultMerge_multi_process import mergebypoly

classnames_15 = [
          'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
          'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
          'basketball-court', 'storage-tank',  'soccer-ball-field', 
          'roundabout', 'harbor', 'swimming-pool', 'helicopter', 
          'container-crane']

color = {'plane':[255,0,0], 
        'baseball-diamond':[0,255,0], 
        'bridge':[0,0,255], 
        'ground-track-field':[255,255,0],
        'small-vehicle':[255,0,255], 
        'large-vehicle':[0,255,255], 
        'ship':[153,50,204], 
        'tennis-court':[255,185,15],
        'basketball-court':[0,139,69], 
        'storage-tank':[175,238,238],  
        'soccer-ball-field':[238,220,130], 
        'roundabout':[139,101,8], 
        'harbor':[0,0,139], 
        'swimming-pool':[144,238,144], 
        'helicopter':[191,239,255], 
        'container-crane':[255,255,255]}

def save_box_to_local(img_name, epoch, dets):
    '''
    img_name(string): the name of single photo. 
    epoch: any name if it can be distinguished
    dets(numpy): the result of your detector
    '''
    result = {}
    result["name"] = img_name
    result["cate"] = []
    result["poly"] = []
    result["score"] = []
    filedict = {}
    for data in dets:
        cat = classnames_15[int(data[6])]
        score = data[5]
        result["cate"].append(cat)
        result["score"].append(score)
        box = data[:5]

        #====================== insert box decode method====================#
        # box = ...
        # ...
        #===================================================================#
        
        Bbox = box.reshape(4,2)
        result["score"].append(Bbox)

    path_txt = "../result/{}/labeltxt/".format(epoch)
    if not os.path.exists(path_txt):
        os.makedirs(path_txt)

    for clses in classnames_15:
        fd = open(os.path.join(path_txt, 'Task1_') + clses + '.txt', 'a')
        filedict[clses] = fd

    for index, cat in enumerate(result["cate"]):
        label = result['name']+ " " + str(result["score"][index])
        poly = result["poly"][index]
        poly = poly.tolist()

        for point in poly:
            x,y = point
            label = label + " " + str(x) + " " + str(y)
        filedict[cat].write(label+"\n")
    return path_txt


def eval_map(epoch, gt_path, results_path, save_dir = '../'):
    '''
    epoch: any name if it can be distinguished
    results_path: the path which you saved by the function 'save_box_to_local'
    save_dir: the path you want to save merge results
    gt_path: the path of ground truth
    '''
    # gt_path = "/root/GQY/CenterNet/data/val/labelTxt"

    save_dir_merge = os.path.join(save_dir, "merge_result", str(epoch))

    if not os.path.exists(save_dir_merge):
        os.makedirs(save_dir_merge)
    # util.Task2groundtruth_poly(results,"data/val")
    if results_path!=[]:
        mergebypoly(results_path,save_dir_merge)

    from dota_evaluation_task1 import main
    with open(save_dir_merge+"/valset.txt","w") as f:
        img_list = os.listdir(gt_path)
        for img in img_list:
            img = img.split(".")[0]
            f.write(img + "\n")
    mAP,classAP = main(save_dir_merge + "/Task1_{:s}.txt", gt_path+"/{:s}.txt",save_dir_merge+"/valset.txt")
    saveAP = ''
    for d in classAP:
        norm_d = format(d,'.3f')
        saveAP = saveAP + str(norm_d) + " "
    with open("ap_log.txt","a") as p:
        p.write("epoch:{}    {}     mAP:{}\n".format(str(epoch),str(saveAP),str(mAP)))
    return

def merge_draw(merge_path, see, save_name, isShow=True, isSplit=True):
    '''
    merge_path: the path which is the same as the 'save_dir_merge' in function 'eval_map'
    see: the path of images
    save_name: the name of save-folder which you want.
               if save_name = None , it will not save images at the disk
    isShow: show image or not
    isSplit: split the result or not
    '''
    import dota_utils as util
    from DOTA import DOTA
    if isSplit:
        util.Task2groundtruth_poly(merge_path,
                            r'see/labelTxt')
    example = DOTA(r'see')
    imgids = example.getImgIds(catNms=[])
    for imgid in tqdm(imgids):
        anns = example.loadAnns(imgId=imgid)
        imgs = example.loadImgs(imgids=imgid)[0]
        for ann in anns:
            cate = ann['name']
            c = color[cate]
            poly = (np.array([ann['poly']],dtype = np.int32))
            img = cv2.drawContours(imgs,poly,-1,c,2)
        if save_name != None:
            cv2.imwrite('see/{}/'.format(save_name) + imgid+'.jpg',img)
        if isShow:
            cv2.imshow('{}'.format(imgid),img)
            cv2.waitKey()