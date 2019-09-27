# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ### getlables

import numpy as np
from scipy import io
from PIL import Image
import glob
import os
path = os.getcwd() #获取当前代码文件的路径 C:\Python\JupyterNotebook\ConPath\Deeplearning_results
path_list = glob.glob(path+'\\382412-2*')#patch文件夹的个数：10个
for k in range(1,len(path_list)+1):
    temp = glob.glob(path+'\\382412-2_'+str(k)+'\\*')
    numfiles = len(temp)#每个文件夹下的文件数
    #print(numoffiles)
    lines=[]
    for i in range(1,numfiles+1):
        file = path+'\\382412-2_'+str(k)+'\\382412-2_'+str(k)+'_ROI_Image_Patch_' + str(i) + '.txt_results.txt'
        f = open(file, 'rb')#以二进制格式打开一个文件用于只读。文件指针将会放在文件的开头。
        for item in f.readlines():#将文件中的内容按换行符进行切分，全部放在列表中
            lines.append(item)#单个文件夹中所有文件合成到一个列表中
    numlabel=len(lines)
    print(numlabel)
    listoflabel=[0]*numlabel
    for line in lines:
        line = str(line.decode('gb2312').encode('utf8'))
        #b'.../ImagePatchInfo/382412-2_1_ROI_Image_Patch/ROI_382412-2_1_Patch_1.tif\t0.00058211\t0.743156\t0.256262\n'
        a = line.split("\\t")
        for i in range(1,4):#提取三个概率比大小,取出最大值及索引
            if i==3:
                a[i]=a[i].split("\\n")[0]
            a[i]=float(a[i])
        max_a=max(a[1:4])
        for i in range(1,4):
            if a[i]==max_a:
                break
        col1=a[0].split("Patch_")[1].split(".")[0]
        col11=int(col1)#int() 函数用于将一个字符串或数字转换为整型，图片索引
        listoflabel[col11-1]=i
    io.savemat(path+"\\382412-2_" +str(k)+ ".mat", {'label': listoflabel}) 

