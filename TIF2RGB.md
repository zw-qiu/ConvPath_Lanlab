```
import os
import cv2 
import glob

def mkdir(path):   
    isExists=os.path.exists(path)# 判断路径是否存在 # 存在 True # 不存在 False
    if not isExists:
        os.makedirs(path) 
        return True
    else:
        return False
    
path = os.getcwd() #获取当前代码文件的路径 C:\Python\JupyterNotebook\ConPath\ImagePatchInfo
path_list = glob.glob(path+'\\*_patch')#patch文件夹的个数：10个
for i in range(1,len(path_list)+1):#ROI一共10个文件夹”
    data_father_name='382412-2_'+str(i)+'_ROI_Image_Patch'#patch文件夹名
    data_base_dir=path+'\\'+data_father_name#图像文件存储目录
    #print(data_base_dir)   #C:\Python\JupyterNotebook\ConPath\ImagePatchInfo\382412-2_1_ROI_Image_Patch
    data_save_dir=path+'\\'+'ROI'+str(i)#图像储存目录
    mkdir(data_save_dir)#创建存储目录
    print(data_save_dir)  #C:\Python\JupyterNotebook\ConPath\ImagePatchInfo\ROI1
    tif_list = [x for x in os.listdir(data_base_dir) if x.endswith(".tif")] #找到当前路径下的所有.tif文件
    for num, imgfile in enumerate(tif_list,start=1):#num为索引，imgfile为图像名
        img = cv2.imread(data_base_dir+'\\'+imgfile,1)  #通道为BGR
        cv2.imwrite(data_save_dir + '\\'+imgfile.split('.')[0]+".jpg", img)
        #print('总共:',len(tif_list),'张，剩余:',len(tif_list)-num,'张')
## img=cv2参数：
#IMREAD_UNCHANGED = -1   #不进行转化，比如保存为16位的图片，读取出来仍然为16位。
#IMREAD_GRAYSCALE = 0    #进行转化为灰度图，比如保存为16位的图片，读取出来为8位，类型为CV_8UC1。
#IMREAD_COLOR = 1        #进行转化为RGB三通道图像，图像深度转为8位
#IMREAD_ANYDEPTH = 2     #保持图像深度不变，进行转化为灰度图。
#IMREAD_ANYCOLOR = 4     #若图像通道数小于等于3，则保持原通道数不变；若通道数大于3，则只取取前三个通道，图像深度转为8位。
```

