import os
import imageio
import numpy as np
from skimage.io import imread



class_names = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
data_labels = []
data_list = []


'''
#creating a list of labels
for folder in range(0,36):
    for element in range(0,1016):
        data_labels.append(class_names[folder])
'''
for folder in range(0,36):
    for element in range(0,1016):
        data_labels.append(folder)



path = "/home/faust/Documents/Python_Code/424Project/data_dir"

#creating vector of flattened images
for(path, dirs, files) in os.walk(path):
    dirs.sort()
   # print(path)
   # print(dirs)
    for each in files:
        temp =  imread(path+'/'+each, as_grey = True)
        temp_flat = temp.reshape(-1)
        data_list.append(temp_flat)

data_array = np.array(data_list)
labels_array = np.array(data_labels)

print(np.shape(data_array))
print(np.shape(labels_array))

#print(len(data_list))



