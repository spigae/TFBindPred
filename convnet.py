import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import sys
from random import shuffle
#
tf.reset_default_graph
# IMG_SIZE = 50
column_size = 16
row_size = 56
#LR = 0.005
#
# enrico 10.04.2018
#
fout1 = open('accuracy.dat', 'w')
fout2 = open('predictions.dat', 'w')
#
files ={
        "TotalEnergyDecomposition":
                {
                        "train_jasp":"for_covnet/Labelled/TotalEnergyDecomposition/train_jasp_noord.npy",
                        "test_jasp": "for_covnet/Labelled/TotalEnergyDecomposition/test_jasp_noord.npy"
                },
}

class Covnet:

        def __init__(self,table_name,LR, n_epoch, iterations, train, test):
                self.LR = LR
                self.table_name = table_name
                self.n_epoch = n_epoch
                self.iterations = iterations
                self.train = train
                self.test = test

        def runCov(self):
                iterations = int(self.iterations)
                accuracy =[]
                TP =[]
                TN =[]
                FP =[]
                FN =[]
                for i in range(iterations):
                        with tf.Graph().as_default():
                                print ("run %s") %(i)
                                model = self.Cov(i)
                                res = model.split(",")
                                accuracy.append(float(res[0]))
                                TP.append(int(res[1]))
                                TN.append(int(res[2]))
                                FP.append(int(res[3]))
                                FN.append(int(res[4]))

        def avg(self,lis):
                avgs = float(sum(lis))/len(lis)
                return avgs

        def Cov(self,num):
                print(float(self.LR))
                MODEL_NAME = 'BNvsNB-{}-{}-{}-{}-{}-{}.model'.format(str(num),str(self.n_epoch),str(self.LR),str(self.train),str(self.table_name) ,'convolution')
                convnet = input_data(shape=[None, row_size, column_size, 1], name='input')
                convnet = conv_2d(convnet,nb_filter=32,filter_size=32, strides=1,activation='relu', padding="same")
                convnet = max_pool_2d(convnet,kernel_size=[2, 2], strides=1, padding="valid")
                convnet = conv_2d(convnet,nb_filter=32,filter_size=32, strides=1, activation='relu' ,padding="same")
                convnet = max_pool_2d(convnet,kernel_size=[2, 2], strides=1, padding="valid")
                convnet = conv_2d(convnet,nb_filter=32,filter_size=32, strides=1, activation='relu' ,padding="same")
                convnet = conv_2d(convnet,nb_filter=32,filter_size=32, strides=1, activation='relu' ,padding="same")
                convnet = fully_connected(convnet, 896, activation='relu')
                convnet = dropout(convnet, 0.4)
                convnet = fully_connected(convnet, 2, activation='softmax')
                convnet = regression(convnet, optimizer="sgd", loss='categorical_crossentropy', name='targets')
                model = tflearn.DNN(convnet, tensorboard_dir ='log')
                train_data =np.load(files[str(self.table_name)][self.train])
                train = train_data
                X = np.array([i[0] for i in train]).reshape(-1, row_size,column_size,1)
                Y = [i[1] for i in train]
                model.fit({'input': X}, {'targets': Y},n_epoch = int(self.n_epoch),show_metric=True, run_id=MODEL_NAME)
                model.save(MODEL_NAME)
                test_data = np.load(files[str(self.table_name)][self.test])
                TP = 0
                FP = 0
                TN = 0
                FN = 0
                for num, data in enumerate(test_data):
                        file_num = data[1]
                        file_data = data[0]
                        data = file_data.reshape(row_size,column_size,1)
                        model_out = model.predict([data])[0]
                        if np.argmax(model_out) == 1:
                                str_label = 'Non Binding site'
                                pred = 'BB'
                        else:
                                str_label = 'Binding site'
                                pred = 'GB'

                        if str(file_num) == '[0 1]':
                                true_label = 'Non Binding'
                                real = 'BB'
                        else:
                                true_label = 'Binding'
                                real = 'GB'
                        fout2.write("%s %s \n" % (real,pred))
                        if str_label == 'Non Binding site' and true_label == 'Non Binding':
                                TN = TN + 1
                        elif str_label == 'Binding site' and true_label == 'Binding':
                                TP = TP + 1
                        elif str_label == 'Non Binding site' and true_label == 'Binding':
                                FN = FN + 1
                        elif str_label == 'Binding site' and true_label == 'Non Binding':
                                FP = FP + 1
                accuracy = float(TP + TN) / float(TP + TN + FP + FN)
                TPR = float(TP) / float(TP+FN)
                TNR = float(TN) / float(FP+TN)
                tot = float(TP + TN + FP + FN)
                print(accuracy,TP,TN,FP,FN,TPR,TNR,tot)
                final_out = "%s,%s,%s,%s,%s,%s" %(accuracy,TP,TN,FP,FN,tot)
                fout1.write("%s %s %s %s %s %s %s %s\n" % (accuracy,TP,TN,FP,FN,TPR,TNR,tot))
                return final_out

if __name__ == "__main__":
        covNet = Covnet(sys.argv[1],sys.argv[2],sys.argv[3],
                sys.argv[4],sys.argv[5],sys.argv[6])
        covNet.runCov()
