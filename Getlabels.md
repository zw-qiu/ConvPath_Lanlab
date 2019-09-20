

```
import numpy as np
from scipy import io
from PIL import Image
import glob
for k in range(1,11):
    temp = glob.glob('E:\\Jupyter-notebook\\ConvPath\\deeplearning_results\\382412-2_'+str(k)+'\\*')
    numoffiles = len(temp)#9
    #print(numoffiles)
    lines=[]
    for i in range(numoffiles):
        file = 'E:\\Jupyter-notebook\\ConvPath\\deeplearning_results\\382412-2_'+str(k)+'\\382412-2_'+str(k)+'_ROI_Image_Patch_' + str(i+1) + '.txt_results.txt'
        f = open(file, 'rb')
        for item in f.readlines():
            lines.append(item)
    numlabel=len(lines)
    print(numlabel)
    listoflabel=[0]*numlabel
    for line in lines:
        line = str(line.decode('gb2312').encode('utf8'))
        a = line.split("\\t")
        for i in range(1,4):
            if i==3:
                a[i]=a[i].split("\\n")[0]
            a[i]=float(a[i])
        max_a=max(a[1:4])
        for i in range(1,4):
            if a[i]==max_a:
                break
        col1=a[0].split("Patch_")[1].split(".")[0]
        col11=int(col1)
        listoflabel[col11-1]=i
    io.savemat("E:\\Jupyter-notebook\\ConvPath\\deeplearning_results\\382412-2_" +str(k)+ ".mat", {'label': listoflabel}) 
```



```
import numpy as np
from scipy import io
from PIL import Image
import glob
from shutil import copyfile

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

