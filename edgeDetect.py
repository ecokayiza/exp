import cv2
import numpy as np
import os
import random

def edge_detection(image_path):
    # 读取图像并转为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 检查
    if image is None:
        print("READ IMAGE ERROR")
        return
    print(image.shape)
    
    #边缘检测
    my_processed=my_detector(image,kernel_size=5,sigma=1.67,high=0.6,low=0.45) 
      
    processed_1=sobel(image)
    processed_2=canny(image)
    processed_3=laplac(image)

    #结果
    cv2.imshow('Original',image)
    cv2.imshow('My EdgeDetector',my_processed)
    # cv2.imshow('sobel', processed_1)
    # cv2.imshow('canny', processed_2)
    # cv2.imshow('laplac', processed_3)   
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#非最大值抑制处理(细化边缘)
def non_maximum_suppression(magnitude, direction):
    """
    1.对于每个像素点，根据其梯度方向，将其与梯度方向相邻的两个像素点进行比较。
    2.如果当前像素点的梯度幅值大于相邻的两个像素点，则保留该像素点的梯度幅值。
    3.否则，将当前像素点的梯度幅值设为零
    """
    M, N = magnitude.shape
    res = np.zeros((M, N), dtype=np.int32)
    angle = direction * 180. / np.pi    #弧度转角度
    angle[angle < 0] += 180             #取正角度

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                #选择梯度方向上相邻的两个像素点
                q = 255
                r = 255
                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j+1]
                    r = magnitude[i, j-1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i+1, j-1]
                    r = magnitude[i-1, j+1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i+1, j]
                    r = magnitude[i-1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i-1, j-1]
                    r = magnitude[i+1, j+1]

                #保留或抑制
                if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                    res[i, j] = magnitude[i, j]
                else:
                    res[i, j] = 0
                    
            except IndexError as e:
                pass
    return res

#双阈值处理
def double_threshold(edges,lowThreshold, highThreshold):
    highThreshold = edges.max() * highThreshold
    lowThreshold = highThreshold * lowThreshold
    print(f'max:{edges.max()} high:{highThreshold} low:{lowThreshold}')
    """
    1.如果像素的梯度幅值大于等于高阈值，则将其标记为强边缘。
    2.介于低阈值和高阈值之间，则将其标记为弱边缘。
    3.小于低阈值，则将其标记为非边缘。
    """
    print(edges)
    strong_i, strong_j= np.where(edges >= highThreshold)
    weak_i, weak_j = np.where((edges <= highThreshold) & (edges >= lowThreshold))
    zeros_i, zeros_j = np.where(edges < lowThreshold)
    edges[zeros_i, zeros_j] = 0
    
    strong_edges = np.zeros(edges.shape, dtype=bool)
    weak_edges = np.zeros(edges.shape, dtype=bool)
    strong_edges[strong_i, strong_j] = True
    weak_edges[weak_i, weak_j] = True
    
    return edges,strong_edges,weak_edges

#滞后阈值处理（连接边缘）
def hysteresis(edges,strong,weak):
    #强边缘作为起始，弱边缘连接到强边缘
    M, N = edges.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if weak[i,j]: #如果是弱边缘
                try:
                    #检查周围8个像素是否有强边缘
                    if (strong[i,j+1] or strong[i,j-1] or strong[i+1,j] or strong[i-1,j]or
                        strong[i-1,j-1] or strong[i+1,j+1] or strong[i-1,j+1] or strong[i+1,j-1]):
                        #有则设置为强边缘，否则消去
                        pass
                    else:
                        edges[i, j] = 0
                except IndexError as e:
                    pass
    return edges

def my_detector(image,kernel_size=5,sigma=1.67,high=0.4,low=0.5):
    #平滑
    kernel=cv2.getGaussianKernel(kernel_size,sigma)  #使用高斯核进行平滑处理
    gaussian_kernel=kernel*kernel.T  
    blur=cv2.filter2D(image,-1,gaussian_kernel)
    #边缘增强
    #sobel核矩阵
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) #水平方向
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) #垂直方向
    #获取梯度
    Ix = cv2.filter2D(blur, -1, Kx)
    Iy = cv2.filter2D(blur, -1, Ky)
    #梯度值和方向
    magnitude = np.sqrt(Ix**2 + Iy**2)
    direction = np.arctan2(Iy, Ix)
    #非最大值边缘抑制（边缘细化）
    edges=non_maximum_suppression(magnitude,direction)
    # alpha=blur.max()/edges.max()
    # cv2.imshow('NMS',cv2.convertScaleAbs(edges,alpha=alpha))
    #双阈值检测
    threshold_image,strong,weak=double_threshold(edges,highThreshold=high,lowThreshold=low)
    #边缘连接
    final_image=hysteresis(threshold_image,strong,weak)
    #梯度值放大到像素值
    alpha=blur.max()/final_image.max()
    final_image=cv2.convertScaleAbs(final_image,alpha=alpha)
    return final_image
    
    
    
def sobel(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)
    return sobel_magnitude

def canny(image):
    return cv2.Canny(image,100,200)

def laplac(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian  

def get_a_pic(dir):
    files=os.listdir(dir)
    images = [file for file in files if file.endswith(('.jpg', '.png'))]
    return os.path.join(dir, random.choice(images))

if __name__=='__main__':
    path=get_a_pic(r'sample\BSDS\images\train')
    edge_detection(path)