from __future__ import  division
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 13:33:51 2016

@author: rajpu
"""
import tensorflow as tf

import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from PIL import Image
import os
import input_data


__author__ = "rajpu"
__date__ = "$Nov 25, 2016 3:58:16 AM$"

def LogisticR(trainX, trainY, testX, testY, ImageX, ImageY):
    print "Logistic Regression"
    maxCorrect = 0
    maxEta = 0
    eta1List = [0.01]
    for eta in eta1List:
        #print eta
        allErrors = np.zeros(shape=(50000,1))
        w = np.random.rand(785,10)
        error = 0
        for i in range(0,50000):
            x = np.append(1,trainX[i,:])
            y = np.zeros(shape=(10,1))
            t = np.zeros(shape=(10,1))
            t[trainY[i]] = 1
            a = np.dot(np.transpose(w),x)
            den = np.sum(np.exp(a))
            for k in range(0,10):
                y[k][0] = np.exp(a[k])/den
            
            Gerror = np.zeros(shape=(785,10))
            #check this error
            error = 0
            for k in range(0,10):
                temp12 = np.multiply((y[k][0] - t[k][0]),x)
                Gerror[:,k] = np.add(Gerror[:,k], temp12)
                error =  error - (t[k][0] * math.log(y[k][0]))
            
            #add iterations for gradient descent
            allErrors[i][0] = error
            #eta = 0.0097
            for k in range(0,10):
                #temp = np.multiply(eta,Gerror[:,k])
                w[:,k] = np.subtract(w[:,k], np.multiply(eta,Gerror[:,k]))
                
        # maybe you can use validation to set training parameter NEETA
        
        predictedValues = np.zeros(shape=(50000,1))
        correct = 0
        wrong = 0
        for i in range(0,50000):
            xt = np.append(1,trainX[i,:])
            yt = np.zeros(shape=(10,1))
            at = np.dot(np.transpose(w),xt)
            dent = np.sum(np.exp(at))
            for k in range(0,10):
                yt[k][0] = np.exp(at[k])/dent
            
            preIndex = np.where(yt == yt.max())[0]  
            predictedValues[i][0] = preIndex
            if preIndex == trainY[i]:
                correct += 1
            else:
                wrong += 1        
        
        if(maxCorrect < correct):
            maxCorrect = correct
            maxEta = eta
                
        valpredictedValues = np.zeros(shape=(10000,1))
        valcorrect = 0
        valwrong = 0
        for i in range(0,10000):
            valxt = np.append(1,validX[i,:])
            valyt = np.zeros(shape=(10,1))
            valat = np.dot(np.transpose(w),valxt)
            valdent = np.sum(np.exp(valat))
            for k in range(0,10):
                valyt[k][0] = np.exp(valat[k])/valdent
            
            preIndex = np.where(valyt == valyt.max())[0]  
            valpredictedValues[i][0] = preIndex
            if preIndex == validY[i]:
                valcorrect += 1
            else:
                valwrong += 1
                
        
        testpredictedValues = np.zeros(shape=(10000,1))
        testcorrect = 0
        testwrong = 0
        for i in range(0,10000):
            testxt = np.append(1,testX[i,:])
            testyt = np.zeros(shape=(10,1))
            testat = np.dot(np.transpose(w),testxt)
            testdent = np.sum(np.exp(testat))
            for k in range(0,10):
                testyt[k][0] = np.exp(testat[k])/testdent
            
            preIndex = np.where(testyt == testyt.max())[0]  
            testpredictedValues[i][0] = preIndex
            if preIndex == testY[i]:
                testcorrect += 1
            else:
                testwrong += 1
        
        imgpredictedValues = np.zeros(shape=(20000,1))
        imgcorrect = 0
        imgwrong = 0
        for i in range(0,20000):
            imgxt = np.append(1,ImageX[i,:])
            imgyt = np.zeros(shape=(10,1))
            imgat = np.dot(np.transpose(w),imgxt)
            imgdent = np.sum(np.exp(imgat))
            for k in range(0,10):
                imgyt[k][0] = np.exp(imgat[k])/imgdent
            
            preIndex = np.where(imgyt == imgyt.max())[0]  
            imgpredictedValues[i][0] = preIndex
            if preIndex == ImageY[i][0]:
                imgcorrect += 1
            else:
                imgwrong += 1
        
        print "Accuracy of Training Data ", (correct/50000)*100
        print "Accuracy of Test Data ", (testcorrect/10000)*100
        print "Accuracy of Valid Data ", (valcorrect/10000)*100
        print "Accuracy of USPS Data ", (imgcorrect/20000)*100
           
        """     
        graphX = list(range(50000))        
        plt.figure(1)
        plt.plot(graphX,allErrors)
        plt.xlabel("data points")
        plt.ylabel("errors")
        plt.title("change in error")
        plt.show()
        
        graphX = list(range(50000))        
        plt.figure(2)
        plt.plot(graphX, predictedValues,'r--', graphX, trainY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Logistic Regression training data")
        plt.show()
        
        graphX = list(range(10000))        
        plt.figure(3)
        plt.plot(graphX, valpredictedValues,'r--', graphX, validY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Logistic Regression test data")
        plt.show()

        graphX = list(range(10000))        
        plt.figure(4)
        plt.plot(graphX, testpredictedValues,'r--', graphX, testY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Logistic Regression valid data")
        plt.show()
        
        graphX = list(range(20000))        
        plt.figure(5)
        plt.plot(graphX, imgpredictedValues,'r--', graphX, ImageY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Logistic Regression USPS data")
        plt.show()
        """
    
def NeuralNetwork(trainX, trainY, testX, testY, ImageX, ImageY):
    print "Neural Network"
    eta2List = [0.03]
    maxCorrect = 0
    maxE1 = 0
    valmaxCorrect = 0
    valmaxE1 = 0
    testmaxCorrect = 0
    testmaxE1 = 0
    imgmaxCorrect = 0
    imgmaxE1 = 0
    maxE2 = 0
    count1 = 0
    count2 = 0
    allErrors = np.zeros(shape=(50,1))
    for eta in eta2List:
        #print "count2 = ",count2
        count2 += 1
        N = 1000
        M = 1024
        D = 784
        K = 10
        iterations = 50
        w1 = np.random.randn(D,M)
        w2 = np.random.randn(M,K)
        b1 = np.ones(shape=(N,M))
        b2 = np.ones(shape=(N,K))
        for r in range(0,50):
            #print r
            for i in range(0,iterations):
                #print i
                x = trainX[(i*N):(i*N+N)]
                t = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    t[j][trainY[k]] = 1
                
                #layer1
                a1 = np.dot(x,w1) + b1
                z = 1/(1 + np.exp(-a1))
                
                #layer2
                a2 = np.dot(z,w2) + b2
                expa2 = np.exp(a2)
                y = expa2/(np.sum(expa2, axis=1).reshape(N,1))
                
                #layer2 Error
                delta2 = np.subtract(y,t)/N
                w2 = np.subtract(w2, np.multiply(eta, np.dot(np.transpose(z), delta2)))
                b2 = b2 - eta*delta2
                #layer1 Error
                delta1 = np.zeros(shape=(N,M))
                dw2 = np.dot(delta2, np.transpose(w2))
                for i in range(0,N):
                    temp = np.dot((z[i,:]),np.transpose(1-z[i,:]))
                    #temp = np.dot((a2[i,:]),np.transpose(1-a2[i,:]))
                    delta1[i,:] = np.multiply(temp,dw2[i,:])
                
                
                #allErrors[i*N:i*N+N,:] = -np.sum(np.multiply((y - t),(a2)), axis=1).reshape(N,1)
                w1 = np.subtract(w1, np.multiply(eta, np.dot(np.transpose(x), delta1)))
                b1 = b1 - eta*delta1
                
             
            
            #train predicting values
            
            yt = np.zeros(shape=(50000,10))
            for i in range(0,iterations):
                xt = trainX[(i*N):(i*N+N)]
                tt = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    tt[j][trainY[k]] = 1
                
                #layer1
                at1 = np.dot(xt,w1) + b1
                zt = 1/(1 + np.exp(-at1))
                
                #layer2
                at2 = np.dot(zt,w2) + b2
                expat2 = np.exp(at2)
                yt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
        
            predictedValues = np.zeros(shape=(50000,1))
            correct = 0
            wrong = 0
            for i in range(0,50000):
                preIndex = np.where(yt[i,:] == yt[i,:].max())[0]  
                predictedValues[i][0] = preIndex
                if preIndex == trainY[i]:
                    correct += 1
                else:
                    wrong += 1 
                    
            if(maxCorrect < correct):
                maxCorrect = correct
                maxE1 = eta
                maxE2 = eta
                
            #valid prdicting values
            valyt = np.zeros(shape=(10000,10))
            for i in range(0,10):
                xt = validX[(i*N):(i*N+N)]
                tt = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    tt[j][validY[k]] = 1
                
                #layer1
                at1 = np.dot(xt,w1) + b1
                zt = 1/(1 + np.exp(-at1))
                
                #layer2
                at2 = np.dot(zt,w2) + b2    
                expat2 = np.exp(at2)
                valyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
        
            valpredictedValues = np.zeros(shape=(10000,1))
            valcorrect = 0
            valwrong = 0
            for i in range(0,10000):
                preIndex = np.where(valyt[i,:] == valyt[i,:].max())[0]  
                valpredictedValues[i][0] = preIndex
                if preIndex == validY[i]:
                    valcorrect += 1
                else:
                    valwrong += 1 
            
            allErrors[r,0] = valwrong/10000
            if(valmaxCorrect < valcorrect):
                valmaxCorrect = valcorrect
                valmaxE1 = eta
                    
                    
            #test prdicting values
            testyt = np.zeros(shape=(10000,10))
            for i in range(0,10):
                xt = testX[(i*N):(i*N+N)]
                tt = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    tt[j][testY[k]] = 1
                
                #layer1
                at1 = np.dot(xt,w1) + b1
                zt = 1/(1 + np.exp(-at1))
                
                #layer2
                at2 = np.dot(zt,w2) + b2    
                expat2 = np.exp(at2)
                testyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
        
            testpredictedValues = np.zeros(shape=(10000,1))
            testcorrect = 0
            testwrong = 0
            for i in range(0,10000):
                preIndex = np.where(testyt[i,:] == testyt[i,:].max())[0]  
                testpredictedValues[i][0] = preIndex
                if preIndex == testY[i]:
                    testcorrect += 1
                else:
                    testwrong += 1 
                
            if(testmaxCorrect < testcorrect):
                testmaxCorrect = testcorrect
                testmaxE1 = eta 
                
            #USPS prdicting values
            imgyt = np.zeros(shape=(20000,10))
            for i in range(0,20):
                xt = ImageX[(i*N):(i*N+N)]
                tt = np.zeros(shape=(N,K))
                for j in range(0,N):
                    k = i*N + j
                    tt[j][ImageY[k][0]] = 1
                
                #layer1
                at1 = np.dot(xt,w1) + b1
                zt = 1/(1 + np.exp(-at1))
                
                #layer2
                at2 = np.dot(zt,w2) + b2    
                expat2 = np.exp(at2)
                imgyt[i*N:i*N+N,:] = expat2/(np.sum(expat2, axis=1).reshape(N,1))
        
            imgpredictedValues = np.zeros(shape=(20000,1))
            imgcorrect = 0
            imgwrong = 0
            for i in range(0,20000):
                preIndex = np.where(imgyt[i,:] == imgyt[i,:].max())[0]  
                imgpredictedValues[i][0] = preIndex
                if preIndex == ImageY[i]:
                    imgcorrect += 1
                else:
                    imgwrong += 1 
                
            if(imgmaxCorrect < imgcorrect):
                imgmaxCorrect = imgcorrect
                imgmaxE1 = eta 
                
                
        print "Accuracy of Training Data ", (maxCorrect/50000)*100
        print "Accuracy of Test Data ", (testmaxCorrect/10000)*100
        print "Accuracy of Valid Data ", (valmaxCorrect/10000)*100
        print "Accuracy of USPS Data ", (imgmaxCorrect/20000)*100
    
        """
        graphX = list(range(50))        
        plt.figure(1)
        plt.plot(graphX,allErrors)
        plt.xlabel("data points")
        plt.ylabel("errors")
        plt.title("change in error")
        plt.show()
        
        graphX = list(range(50000))        
        plt.figure(2)
        plt.plot(graphX, predictedValues,'r--', graphX, trainY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network training data")
        plt.show()
        
        graphX = list(range(10000))        
        plt.figure(3)
        plt.plot(graphX, valpredictedValues,'r--', graphX, validY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network test data")
        plt.show()
    
        graphX = list(range(10000))        
        plt.figure(4)
        plt.plot(graphX, testpredictedValues,'r--', graphX, testY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network valid data")
        plt.show()
        
        graphX = list(range(20000))        
        plt.figure(5)
        plt.plot(graphX, imgpredictedValues,'r--', graphX, ImageY, 'b--')
        plt.xlabel("data points")
        plt.ylabel("Target and Predicted values")
        plt.title("Singal Neural Network USPS data")
        plt.show()
        """
    
def ConvolutionalNeuralNetwork(ImageX, ImageY):

    print "Convolutional Neural Network"

    sess = tf.InteractiveSession()

    mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

    x = tf.placeholder("float", shape = [None, 784])
    y_ = tf.placeholder("float", shape = [None, 10])


    x_image = tf.reshape(x, [-1, 28, 28, 1])


    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    
    W_fcl = weight_variable([7 * 7 * 64, 1024])
    b_fcl = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fcl = tf.nn.relu(tf.matmul(h_pool2_flat, W_fcl) + b_fcl)
    
    
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fcl, keep_prob)
    
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.initialize_all_variables())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        if i % 1000 == 0:
            train_accuracy = 100*accuracy.eval(feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1.0})
            print "step %d, training accuracy %g" % (i, train_accuracy)
        train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    print "test accuracy %g" % 100*accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


    print "USPS accuracy %g" % 100*accuracy.eval(feed_dict = {x: ImageX, y_: ImageY, keep_prob: 1.0})

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') 
               
if __name__ == "__main__":
    #print "Hello World"
    pkl_file = open('mnist.pkl', 'rb')
    data = pickle.load(pkl_file)
    trainX = data[0][0]
    trainY = data[0][1]
    validX = data[1][0]
    validY = data[1][1]
    testX = data[2][0]
    testY = data[2][1]

    #Reading usps data images and creating ImageX and ImageY after scaling
    
    ImageX = np.zeros(shape=(20000,784))
    ImageY = np.zeros(shape=(20000,1))
    imgCount = 0
    for root, directories, filenames in os.walk('./USPSdata/Numerals'):
        for directory in directories:
            #print directory
            #print (os.path.join(root, directory))
            for root, directories, filenames in os.walk('./USPSdata/Numerals/'+directory):
                for filename in filenames: 
                    if filename == "Thumbs.db" or filename == "2.list":
                        ignore = 1
                    else:
                        #from PIL import Image
                        img = Image.open('./USPSdata/Numerals/' + directory + '/' + filename)
                        img = img.resize((28, 28))
                        #imgdata = np.array(img.getdata())
                        #imgdata = 1 - np.square(imgdata)/65536
                        imgdata = 1 - np.array(img.getdata())/256
                        ImageX[imgCount,:] = imgdata
                        ImageY[imgCount,0] = directory
                        imgCount += 1
   
                        
    LogisticR(trainX, trainY, testX, testY, ImageX, ImageY)
    
        
   
    NeuralNetwork(trainX, trainY, testX, testY, ImageX, ImageY)
    
    
    ConvolutionalNeuralNetwork(ImageX, ImageY)