import cv2
import os
import numpy as np


frame_width = 640
frame_height = 480


out_rgb = cv2.VideoWriter('Images/Depth/Not Fall/Encoded_Videos/encoded_depth.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
def main(indx):
    filenames = [f.path for f in os.scandir('./Images/Depth/Not Fall/Images/adl-'+str(indx)+'-cam0-d') if f.is_file() and f.path.endswith(('.png', '.jpg'))]
            
    for f in sorted(filenames):
        frame = cv2.imread(f,1)
        """ im = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        im = im//32
        im = im*32
        im = np.where(im == 0, 100, im)
        im = cv2.equalizeHist(im)
        #im = cv2.bilateralFilter(im,9,75,75)
        imC = cv2.applyColorMap(im, cv2.COLORMAP_HOT)
        #cv2.imshow('frame',imC) """
        out_rgb.write(frame)
    print(indx)
    #out_d.release()


if __name__ == "__main__":
    for i in range(1,2):
        main(i)
    out_rgb.release()
