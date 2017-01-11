"""
Created on Wed Dec 02 11:04:18 2015

@author: MaheshaK
"""
import numpy as np
import scipy
from scipy import io

#Import all the data from files
test_image = np.load("data/test1.npy")
test11_image = np.load("data/test1.1.npy")
test12_image = np.load("data/test1.2.npy")
test13_image = np.load("data/test1.3.npy")
test14_image = np.load("data/test1.4.npy")

test2_image = np.load("data/test2.npy")
test21_image = np.load("data/test2.1.npy")
test22_image = np.load("data/test2.2.npy")
test23_image = np.load("data/test2.3.npy")
test24_image = np.load("data/test2.4.npy")

test_boxinfo = np.loadtxt('data/test1.dat',dtype=str)
test2_boxinfo = np.loadtxt('data/test2.dat',dtype=str)

train1_image = np.load("data/train1.npy") #train data
train2_image = np.load("data/train2.npy")
train3_image = np.load("data/train3.npy")
train4_image = np.load("data/train4.npy")

train1_boxinfo = np.loadtxt('data/train1.dat',dtype=str)
train2_boxinfo = np.loadtxt('data/train2.dat',dtype=str)
train3_boxinfo = np.loadtxt('data/train3.dat',dtype=str)
train4_boxinfo = np.loadtxt('data/train4.dat',dtype=str)

enword = np.loadtxt('data/words.txt',dtype=str)

#Function takes raw data and bounding box info to produce a features array with all padded features and an array with labels
def features(data,image):
    letters_array = np.empty((0,900))
    labels = np.empty((1))
    i=0
    data_height = data.shape[0]
    for line in xrange(data_height):
        #Get co-ordinated for bounding boxes
        height = image.shape[0]
        xleft = int(data[line,1])
        xright = int(data[line,3])
        ybot = int(data[line,2])
        ytop = int(data[line,4])
        single_letter =  image[(height-ytop):(height-ybot),xleft:xright]
        asize = single_letter.shape
        #Pad each box so they are all the same size (30,30)
        if (asize[0]<30) and (asize[1]<30):
            intermediate_array1 = np.pad(single_letter,((0,30-asize[0]), (0,30-asize[1])),'constant',constant_values=(255,255))
            intermediate_array3 = np.reshape(intermediate_array1,(1,900),order='F')

        if (asize[0]<30) and (asize[1]>=30):
            intermediate_array1 = single_letter[:,0:30]
            intermediate_array2 = np.pad(intermediate_array1,((0,30-asize[0]), (0,0)),'constant',constant_values=(255,255))
            intermediate_array3 = np.reshape(intermediate_array2,(1,900),order='F')

        if (asize[0]>=30) and (asize[1]<30):
           intermediate_array1 = single_letter[0:30,:]
           intermediate_array2 = np.pad(intermediate_array1,((0,0), (0,30-asize[1])),'constant',constant_values=(255,255))
           intermediate_array3 = np.reshape(intermediate_array2,(1,900),order='F')

        if (asize[0]>=30) and (asize[1]>=30):
           intermediate_array1 = single_letter[0:30,0:30]
           intermediate_array3 = np.reshape(intermediate_array1,(1,900),order='F')

        letters_array = np.vstack((letters_array,intermediate_array3))
        #Get labels from .dat file
        individual_label = np.array(data[line,0])
        individual_label_reshaped = np.reshape(individual_label,(1,-1))
        labels = np.vstack((individual_label_reshaped,labels))
        labels_reshaped1 = np.reshape(labels[0:((data.shape)[0])],(1,-1))
        labels_reshaped = np.fliplr(labels_reshaped1)
    i+=1
    return(letters_array,labels_reshaped)

#Get all features (Stack all training features)
train1_features,train1_labels = features(train1_boxinfo,train1_image)
train2_features,train2_labels = features(train2_boxinfo,train2_image)
train3_features,train3_labels = features(train3_boxinfo,train3_image)
train4_features,train4_labels = features(train4_boxinfo,train4_image)
train12_features = np.vstack((train1_features,train2_features))
train12_labels = np.hstack((train1_labels,train2_labels))
train34_features = np.vstack((train3_features,train4_features))
train34_labels = np.hstack((train3_labels,train4_labels))
trainall_features = np.vstack((train12_features,train34_features))
trainall_labels = np.hstack((train12_labels,train34_labels))

#test_features,test_labels = features(test_boxinfo,test_image)
#test11_features,test11_labels = features(test_boxinfo,test11_image)
#test12_features,test12_labels = features(test_boxinfo,test12_image)
#test13_features,test13_labels = features(test_boxinfo,test13_image)
#test14_features,test14_labels = features(test_boxinfo,test14_image)
test2_features,test2_labels = features(test2_boxinfo,test2_image)
#test21_features,test21_labels = features(test2_boxinfo,test21_image)
#test22_features,test22_labels = features(test2_boxinfo,test22_image)
#test23_features,test23_labels = features(test2_boxinfo,test23_image)
#test24_features,test24_labels = features(test2_boxinfo,test24_image)

# DIMENTIONALITY_REDUCTION
def dimentionality_reduction(test_array,train_array,features):
    #Compute principle components
    covx = np.cov(train_array, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N-features, N-1))
    v = np.fliplr(v)

    #Domentionality reduction
    pcatrain = np.dot((train_array - np.mean(train_array)), v)
    pcatest = np.dot((test_array - np.mean(train_array)), v)

    return pcatrain,pcatest

#CLASSIFY
def classify(train_array, train_labels, test_array, test_labels, features=None):
    pcatrain_data,pcatest_data = dimentionality_reduction(test_array,train_array,12)

    if features is None:
        features=np.arange(0, pcatrain_data.shape[1])

    train = pcatrain_data[:, features]
    test = pcatest_data[:, features]
    #Computer cosine distance
    x= np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test*test,axis=1))
    modtrain=np.sqrt(np.sum(train*train,axis=1))
    dist = x/(np.outer(modtest.transpose(), modtrain));
    nearest=np.argmax(dist, axis=1)
    #Generate classified labels
    label = train_labels[0, nearest]
    #Calculate accuracy
    correct = 0.0
    for i in xrange(test_labels.size):
        if test_labels[0,i]==label[i]:
            correct = correct+1
    i+=1
    score = (correct/label.shape[0])*100.0

    return score,label
#Test 1 classifications
#score1,classified1labels = classify(trainall_features, trainall_labels, test_features, test_labels, xrange(1,11))
#score11,classified11labels = classify(trainall_features, trainall_labels, test11_features, test11_labels, xrange(1,11))
#score12,classified12labels = classify(trainall_features, trainall_labels, test12_features, test12_labels, xrange(1,11))
#score13,classified13labels = classify(trainall_features, trainall_labels, test13_features, test13_labels, xrange(1,11))
#score14,classified14labels = classify(trainall_features, trainall_labels, test14_features, test14_labels, xrange(1,11))
#Test 2 classifications
score2,classified2labels = classify(trainall_features, trainall_labels, test2_features, test2_labels, xrange(1,11))
#score21,classified21labels = classify(trainall_features, trainall_labels, test21_features, test21_labels, xrange(1,11))
#score22,classified22labels = classify(trainall_features, trainall_labels, test22_features, test22_labels, xrange(1,11))
#score23,classified23labels = classify(trainall_features, trainall_labels, test23_features, test23_labels, xrange(1,11))
#score24,classified24labels = classify(trainall_features, trainall_labels, test24_features, test24_labels, xrange(1,11))

#print("Test 1 accuracy: " + str(score1))
#print("Test 1.1 accuracy: " + str(score11))
#print("Test 1.2 accuracy: " + str(score12))
#print("Test 1.3 accuracy: " + str(score13))
#print("Test 1.4 accuracy: " + str(score14))

print("Test 2 accuracy: " + str(score2))
#print("Test 2.1 accuracy: " + str(score21))
#print("Test 2.2 accuracy: " + str(score22))
#print("Test 2.3 accuracy: " + str(score23))
#print("Test 2.4 accuracy: " + str(score24))

#Function to print a string of classified labels
def tell_story(labels,label_info):
    story = ""
    space = " "
    for i in xrange(labels.size):
        if (label_info[i,5]=="1"):
            story = story  + labels[i]+ space
        else: story = story + labels[i]
    return story


#Print all classified labels for test 1 and 2 as strings
#print("Test 1:")
#tell_story(classified1labels,test_boxinfo)
#print("Test 2:")
test2story = tell_story(classified2labels,test2_boxinfo)
test2storys = test2story.split()
for i in xrange(len(test2storys)):
    for x in range (enword.size):
        if test2story[i] == enword[x]:
            print("True")
        else: print(test2story[i])
