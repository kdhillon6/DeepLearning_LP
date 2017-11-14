import os
import numpy as np
#import sklearn.svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

letters = [
        '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z' ]

def readData(dataFolder):
    imageData = []
    targetData = []
    for eachLetter in letters:
        for each in range(10):
            imagePath = os.path.join(dataFolder, eachLetter, eachLetter + '_' + str(each) + '.jpg')
            imageDetails = imread(imagePath, as_grey = True)
            imageBW = imageDetails < threshold_otsu(imageDetails)

            #Flattening image
            flattenedBW = imageBW.reshape(-1)
            imageData.append(flattenedBW)
            targetData.append(eachLetter)

        return (np.array(imageData), np.array(targetData))

def crossValidation(mlModel, foldNum, trainData, trainLabel):
     accuracyResult = cross_val_score(mlModel, trainData, trainLabel, cv=foldNum)
     print("Cross Validation Result for ", str(foldNum), " -fold")
     print(accuracyResult * 100 + "%")

currentFolder = os.path.dirname(os.path.realpath(__file__))
datasetFolder = os.path.join(currentFolder, 'train')
imageData, targetData = readData(datasetFolder)

clf = SVC(kernel = 'linear', probability = True)
crossValidation(clf, 4, imageData, targetData)
clf.fit(imageData, targetData)

saveFolder = os.path.join(currentFolder, "models/svc/")
if not os.path.exists(saveFolder):
    os.makedirs(saveDirectory)

joblib.dump(clf, saveDirectory + "svc.pkl")



