from keras.models import load_model
from PIL import Image
import numpy as np
import os

os.chdir("/home/faust/Documents/Python_Code/keras_LP");
input_path = "input_dir"
model = load_model('model.h5')


input_files = os.listdir(input_path)
input_files.sort()
img = input_files[0]
im = Image.open(input_path + '/' + img)
imrs = im.resize((m,n))
imrs = imrs.convert('1')
imrs = img_to_array(imrs)/255
imrs = imrs.transpose(2,0,1)
imrs = imrs.reshape(1,m,n)

x = []
x.append(imrs)
x = np.array(x)
predictions = model.predict(x)










