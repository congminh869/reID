# This is the python code for calculating iou between pred_box and gt_box
# author:Forest 2019.7.19

import numpy as np


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax


if __name__ == "__main__":
    # test1
    pred_box = np.array([50, 50, 90, 100]) #x1, y1, x2, y2
    gt_box = np.array([70, 80, 120, 150])
    print("The overlap of pred_box and gt_box:", get_iou(pred_box, gt_box))

    # test2
    # pred_bboxes = np.array([[15, 18, 47, 60],
    #                         [50, 50, 90, 100],
    #                         [70, 80, 120, 145],
    #                         [130, 160, 250, 280],
    #                         [25.6, 66.1, 113.3, 147.8]])
    # gt_bbox = np.array([70, 80, 120, 150])
    # print(get_max_iou(pred_bboxes, gt_bbox))

    #test3 
    pred_bboxes = np.array([[1,         431,          22,         540,           0,           0,           0,           0,          10],
                             [396,          70,         427,         161,           0,           0,           0,           0,           9],
                             [524,          63,         554,         155,           0,           0,           0,           0,           8],
                             [551,         121,         586,         201,           0,           0,           0,           0,           7],
                             [133,         188,         164,         296,           0,           0,           0,           0,           6],
                             [675,         289,         726,         410,           0,           0,           0,           0,           5],
                             [551,         489,         587,         539,           0,           0,           0,           0,           4],
                             [866,         232,         902,         337,           0,           0,           0,           0,           3],
                             [185,         204,         225,         320,           0,           0,           0,           0,           2],
                             [50,         275,          89,         391,           0,           0,           0,           0,           1]])
    gt_bbox = np.array([526, 63, 556, 151])
    iou, iou_max, nmax = get_max_iou(pred_bboxes, gt_bbox)
    print(iou, iou_max, nmax)
    print(pred_bboxes[iou.argmax(axis=0)])


     #     [[          1         431          22         540           0           0           0           0          10]
     # [        396          70         427         161           0           0           0           0           9]
     # [        524          63         554         155           0           0           0           0           8]
     # [        551         121         586         201           0           0           0           0           7]
     # [        133         188         164         296           0           0           0           0           6]
     # [        675         289         726         410           0           0           0           0           5]
     # [        551         489         587         539           0           0           0           0           4]
     # [        866         232         902         337           0           0           0           0           3]
     # [        185         204         225         320           0           0           0           0           2]
     # [         50         275          89         391           0           0           0           0           1]]




