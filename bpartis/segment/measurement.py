import numpy as np
import pandas as pd
import cv2


class Measurer:
    def __init__(self, segmentation, image=[]):
        self.segmentation = segmentation
        if image!=[]:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            self.image = image
        # unit8に変換しないとcv2で処理できない
        self.contours, self.hierrarchy = cv2.findContours(segmentation.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        self.rects = []
        self.circles = []
        self.ellipses = []
        self.params = pd.DataFrame(self.calc_param())
        self.params.index += 1


    def calc_param(self):
        param_li = []
        
        for cnt in self.contours:
            # 面積
            area = cv2.contourArea(cnt)
            # 周囲長
            perimeter = cv2.arcLength(cnt,True)
            # 最小外接短形
            rect = cv2.minAreaRect(cnt)
            rect_l = [rect[1][0],rect[1][1]]
            rect_l.sort()
            # 最小外接円
            circle = cv2.minEnclosingCircle(cnt)
            # 楕円近似
            ellipse = cv2.fitEllipse(cnt)
            ellipse_r = [ellipse[1][0],ellipse[1][1]]
            ellipse_r.sort()

            param = {
                    'area':area,
                    'perimeter':perimeter,
                    'rect_short':rect_l[0],
                    'rect_long':rect_l[1],
                    'rect_angle':rect[2],
                    'ellipse_short':ellipse_r[0],
                    'ellipse_long':ellipse_r[1],
                    'ellipse_angle':ellipse[2],
                    'circle_r':circle[1],
            }
            param_li.append(param)
            self.rects.append(rect)
            self.circles.append(circle)
            self.ellipses.append(ellipse)
        
        return param_li

    def draw_contours(self):
        img = self.image.copy()
        img = cv2.drawContours(img, self.contours, -1, (0,255,0), 2)
        return img

    def draw_rect(self):
        img = self.image.copy()
        for rect in self.rects:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            img = cv2.drawContours(img,[box],0,(0,255,0),2)
        return img

    def draw_circle(self):
        img = self.image.copy()
        for circle in self.circles:
            (x,y),radius = circle
            center = (int(x),int(y))
            radius = int(radius)
            img = cv2.circle(img,center,radius,(0,255,0),2)
        return img
    
    def draw_ellipse(self):
        img = self.image.copy()
        for ellipse in self.ellipses:
            img = cv2.ellipse(img,ellipse,(0,255,0),2)
        return img

    def draw_num(self, img_):
        img = img_.copy()
        for i,rect in enumerate(self.rects):
            x,y = int(rect[0][0]),int(rect[0][1])
            img = cv2.putText(
                            img,
                            text=f'{i+1}',
                            org=(x,y),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.0,
                            color=(0,0,255),
                            thickness=2,
                            )
        return img
