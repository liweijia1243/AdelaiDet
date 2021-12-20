import numpy as np
import cv2
import os
import json
import argparse
import glob
import matplotlib.pyplot as plt
import pandas

def create_coco_files(datapath : str):
    result_json= {
        "info": {"contributor": "li weijia"},
        "categories" :[
            {"id": 0, "name": "dash"},
            {"id": 1, "name": "solid"}
        ]
    }
    data_list = glob.glob(os.path.join(datapath, "*.json"))
    img_names = [file.split('/')[-1][:-4]+'jpg' for file in data_list]
    #print(len(img_names))
    img_info=[]
    annotations_info = []
    #get images dict
    ids = np.arange(len(img_names),dtype=np.int0)
    #ids = np.arange(2,dtype=np.int0)
    for id, img_name in zip(ids, img_names):
        img_info.append(
            {
                "width": 512,
                "height": 512,
                "id" : int(id),
                "file_name": "/home/liweijia/cloud_ws/voc/JPEGImages/"+img_name
            },
        )

    #get annotations dict
    ins_id =0  
    for img_id, ori_json_path in zip(ids, data_list):
        with open(ori_json_path,"r") as f:
            ori_json = json.load(f)
        for instance in ori_json["shapes"]:
            (bbox, segmentation, area) = get_rle_seg(instance["points"])
            ins_dic ={}
            ins_dic["id"] = ins_id
            ins_dic["image_id"] = int(img_id)
            ins_dic["category_id"] = 0 if instance["label"]=="dash_line" else 1
            ins_dic["segmentation"] = segmentation
            ins_dic["bbox"] = bbox
            ins_dic["ignore"] = 0
            ins_dic["iscrowd"] = 0
            ins_dic["area"] = int(area)
            annotations_info.append(ins_dic)
            ins_id = ins_id+1
        pass

    result_json["images"] = img_info
    result_json["annotations"] = annotations_info
    #save json_file
    with open('./' + "train.json", 'w') as f:
        json.dump(result_json, f, indent=2)

    pass

def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    h = int(np.max(rows)-y1)
    w = int(np.max(clos)-x1)
    return (x1, y1, w, h)

def get_rle_seg(points : list):
    #TODO convert point list to RLE
    rle_result = [] 

    for point in points:
        rle_result.append(point[0])
        rle_result.append(point[1])
    img = np.zeros((512,512),np.uint8)
    line = np.array(points, dtype=np.double).reshape(-1,1,2)
    line = np.array(np.floor(line),dtype=np.int32)
    cv2.polylines(img,[line],False,1,3)
    bbox = mask2box(img)

    contours,_ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            segmentation.append(contour)
    
    area = np.sum(img==1)

    # print(segmentation)
    # img_new = np.zeros((512,512),np.uint8)
    # line = np.array(segmentation, dtype=np.double).reshape(-1,1,2)
    # line = np.array(np.floor(line),dtype=np.int32)
    # cv2.polylines(img_new,[line],True,255,1)
    # plt.imshow(img_new)
    # plt.show()

    return (bbox, segmentation, area)

def get_parser():
    parser = argparse.ArgumentParser(description="the voc json_path you want to convert")
    parser.add_argument(
        "--data-path",
        default=" ",
        help="dataset to generate",
    )
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    create_coco_files(args.data_path)
