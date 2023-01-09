import numpy as np
import cv2
import matplotlib.pyplot as plt

def vis_segmap(segmentation, cm_name='gist_ncar', image=[]):
    cm = plt.get_cmap(cm_name)

    x,y = segmentation.shape
    seg_cl = np.zeros((x,y,3))

    for i in list(np.unique(segmentation)):
        # 背景はスキップ
        if i==0:
            continue

        cl_num = i/23
        cl_num = cl_num - int(cl_num)

        seg_cl[segmentation==i] = cm(cl_num)[:3]

    if image==[]:
        return seg_cl
    else:
        seg_cl_concat = (cv2.cvtColor(image, cv2.COLOR_BGR2RGB)/255 + seg_cl)*0.5
        return seg_cl, seg_cl_concat