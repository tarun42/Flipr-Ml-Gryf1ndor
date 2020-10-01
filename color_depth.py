import cv2
import numpy as np
im = cv2.imread('fall-01-cam0-d-001.png', cv2.IMREAD_GRAYSCALE)
im = im//16
im = im*16
im = np.where(im == 0, 100, im)

#im = cv2.bilateralFilter(im,9,75,75)
for i in range(22):
    print(i)
    equ = cv2.equalizeHist(im)
    imC = cv2.applyColorMap(im, i)
    cv2.imshow('Colour', imC)
    imE = cv2.applyColorMap(equ, i)
    cv2.imshow('EQU', imE)
    blur = cv2.bilateralFilter(imE,9,75,75)
    cv2.imshow('blur', blur)
    cv2.waitKey(0)





# c = 255/(np.log(1 + np.max(im))) 
# log_transformed = c * np.log(1 + im) 
# # Specify the data type. 
# log_transformed = np.array(log_transformed, dtype = np.uint8) 
# equ_log = cv2.equalizeHist(log_transformed)
# cv2.imshow('log', equ_log)

 
# # Open the image. 
  
# # Trying 4 gamma values. 
# gamma = 0.1
# while(gamma<4): 
      
#     # Apply gamma correction. 
#     gamma_corrected = np.array(255*(im / 255) ** gamma, dtype = 'uint8') 
  
#     # Save edited images. 

#     x = cv2.applyColorMap(gamma_corrected, cv2.COLORMAP_JET)
#     name = int(gamma*100)
#     cv2.imwrite('Gamma/gamma_transformed'+str(name)+'.jpg', x) 
#     gamma+=0.1


# cv2.waitKey(0)
# cv2.imwrite("color_encoded.png", imC)
# cv2.imwrite("color_EQU_encoded.png", imE)
# res = np.hstack((imC,imE))
# cv2.imwrite("color&EQU_encoded.png", res)
cv2.destroyAllWindows()