### 复制training中的某一种样本N遍

```
import numpy as np
from scipy import io
from PIL import Image
import glob
from shutil import copyfile
import os
path = os.getcwd()

def LoadInMat(path):
        # the path should also including the name of the .mat file
        File_Disp = io.loadmat(path)
        # Establish an empty list to save the valid name of variables
        NameOfVariable = []
        for key in File_Disp.keys():
            if key == '__globals__' or key == '__version__' or key == '__header__':
                continue
            else:
                NameOfVariable.append(key)
        MatData = {}
        for t in range(len(NameOfVariable)):
            MatData[NameOfVariable[t]] = File_Disp[NameOfVariable[t]].tolist()
        MatData['Keys'] = NameOfVariable
        return MatData
    
temp = glob.glob('E:\\Jupyter-notebook\\ConvPath\\train1\\*')
numoffiles = len(temp)#9
#print(numoffiles)
lines=[]
for i in range(numoffiles):
    loadmat=LoadInMat(temp[i])
    img=loadmat['img']
    label=loadmat['label']
    if label[0][0]==1:
        for k in range(8):
            copyfile(temp[i],temp[i].split('.mat')[0]+'_'+str(k)+'.mat')
```

