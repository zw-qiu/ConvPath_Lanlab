```
import os
import cv2 
path = os.getcwd() #获取当前代码文件的路径
tif_list = [x for x in os.listdir(path) if x.endswith(".tif")] #找到当前路径下的所有.tif文件
for num, i in enumerate(tif_list):
    img = cv2.imread(i,-1)  #这里选择-1，不进行转化
    cv2.imwrite(i.split('.')[0]+".jpg", img)
    print('总共:',len(tif_list),'张，剩余:',len(tif_list)-num-1,'张')
## img=cv2参数：
#IMREAD_UNCHANGED = -1   #不进行转化，比如保存为16位的图片，读取出来仍然为16位。
#IMREAD_GRAYSCALE = 0    #进行转化为灰度图，比如保存为16位的图片，读取出来为8位，类型为CV_8UC1。
#IMREAD_COLOR = 1        #进行转化为RGB三通道图像，图像深度转为8位
#IMREAD_ANYDEPTH = 2     #保持图像深度不变，进行转化为灰度图。
#IMREAD_ANYCOLOR = 4     #若图像通道数小于等于3，则保持原通道数不变；若通道数大于3，则只取取前三个通道，图像深度转为8位。
```

