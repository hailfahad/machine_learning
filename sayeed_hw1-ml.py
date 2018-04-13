import numpy as np
import scipy.spatial.distance as scdist
import scipy.io as scio
from operator import itemgetter,attrgetter
from collections import Counter
from scipy.stats import itemfreq
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from pathlib import Path

def main():
    millis_start = int(round(time.time() * 1000));
    print("Start",millis_start);
    NUMBER_OF_NEAREST_NEIGHBOR=1;
    # load data
    #file_path="C:\Python\ML\MNIST_digit_data.mat";
    file_path=input("Enter MNIST file path: ");
    if(Path(file_path).exists()==False):
        print("File does not exist");
        return;
    #print("file path",file_path);
    matdata=scio.loadmat(file_path);
    labels_test=np.array(matdata['labels_test']);
    labels_train=np.array(matdata['labels_train']);
    images_test=np.array(matdata['images_test']);
    images_train=np.array(matdata['images_train']);
    
    print("file size",images_train.shape);
    #testing_data_estimates=estimateTest(images_train,labels_train,images_test, labels_test,NUMBER_OF_NEAREST_NEIGHBOR);
    #print("testing estimates",testing_data_estimates);
            
    accuracy_arr,average_accuracy=kNN(images_train, labels_train, images_test[0:1000],labels_test[0:1000],NUMBER_OF_NEAREST_NEIGHBOR);
    print("Digit-wise accuracy >>",accuracy_arr);
    print("Average accuracy >>",average_accuracy);
    #Plot change in number of training data for performance changes
    training_start_size=30;
    training_end_size=10000;
    number_of_data_sets=10;
    k=[1];
    data_plot=plotPerformance(training_start_size,training_end_size,number_of_data_sets,images_train,labels_train,images_test[0:1000],labels_test[0:1000],k[0]);
    #Plot the chart for this 
    plt.ylabel("Accuracy");
    plt.xlabel("Training Dataset");
    plt.title("Performance across training data for k=1");
    plt.hold(True);
    plt.grid(True);
    plt.ylim((0,1));
    plt.plot(data_plot[:,0],data_plot[:,1],'xb-');
    plt.savefig("7c.png");
    #plt.show();
    print("Q 7c done and output diagram saved");
    
    #Now plot for multiple k
    k=[1,2,3,5,10];
    plt.ylabel("Accuracy");
    plt.xlabel("Training Dataset");
    plt.title("Performance across training data for different K-neighbors");
    plt.hold(True);
    plt.grid(True);
    plt.ylim((0,1));
    #fig=plt.figure(1);
    color=iter(cm.rainbow(np.linspace(0,1,len(k))));
    
    for index in k:
        #print("value of k",index);
        data_plot=plotPerformance(training_start_size,training_end_size,number_of_data_sets,images_train,labels_train,images_test[0:1000],labels_test[0:1000],index);
        c=next(color);
        #ax1=fig.add_subplot(111);
        plt.plot(data_plot[:,0],data_plot[:,1],c=c,label="k="+str(index));
    
    plt.legend();    
    plt.savefig("7d.png");
    #plt.show();
    print("Output for Q 7d saved");
    
    
    #Selection of best K for 2000 training data
    #First add the labels to the training data, then shuffle, remove the labels data and choose for training and validation
    training_data_2000,labels_data_2000=shuffle_in_unison(images_train,labels_train,2000);
    
    training_data_1000=training_data_2000[:1000];
    training_labels_1000=labels_data_2000[:1000];
    validation_data_1000=training_data_2000[1000:2000];
    validation_labels_1000=labels_data_2000[1000:2000];
    k_accuracy_array=np.empty([len(k),2]);
    count=0;
    plt.ylabel("Accuracy");
    plt.xlabel("K-nearest neighbors");
    plt.title("Best K across 1000 training data and 1000 validation data");
    plt.hold(True);
    plt.grid(True);
    plt.ylim((0.6,1));
    color=iter(cm.rainbow(np.linspace(0,1,len(k))));
    
    for index in k:
        accuracy_arr,average_accuracy=kNN(training_data_1000, training_labels_1000, validation_data_1000,validation_labels_1000,index);
        k_accuracy_array[count,0]=index;
        k_accuracy_array[count,1]=average_accuracy;
        count+=1;
        c=next(color);
        plt.scatter(index,average_accuracy,c=c,marker='x',label="k="+str(index));
            
    #print("Accuracy array",k_accuracy_array);
                
    plt.legend();    
    plt.savefig("7e.png");
    #plt.show();
    print("Output for 7e saved");
    k_accuracy_array=sorted(k_accuracy_array,key=itemgetter(1), reverse=True);
    print("Best K for these 2000 training data selected is >>",int(k_accuracy_array[0][0]));
    
    millis_end = int(round(time.time() * 1000));
    print("Time Completed",millis_end-millis_start);
    
def shuffle_in_unison(train_data,labels_data,number_of_data):
    shuff_state=np.random.get_state();
    np.random.shuffle(train_data[:number_of_data]);
    np.random.set_state(shuff_state);
    np.random.shuffle(labels_data[:number_of_data]);
    return train_data,labels_data;
    


def plotPerformance(training_start_size,training_end_size,number_of_data_sets,images_train,labels_train,images_test,labels_test,k):
    #Determine the training data sets using logspace command and then calculate average accuracy against the test images for plotting
    #Normalize and then unnormalize
    if(number_of_data_sets==0):
        #No array of training data. Choose the training end size
        train_data_sets_arr=[training_end_size];
    else:
        train_data_sets_arr=np.round(np.logspace(np.log10(training_start_size),np.log10(training_end_size),number_of_data_sets));
    
    #train_data_sets_arr=[30,57,109,208,397,756,1442,2750];
    #Assume that i got the training data set sizes
    data_plot=np.empty([len(train_data_sets_arr),2]);
    
    for index in range(len(train_data_sets_arr)):
        #call the kNN function to retrieve accuracy for each set of training data
        accuracy_arr,average_accuracy=kNN(images_train[:train_data_sets_arr[index]], labels_train[:train_data_sets_arr[index]], images_test,labels_test,k);
        data_plot[index,0]=train_data_sets_arr[index];
        data_plot[index,1]=average_accuracy;
    #Got all datapoints for plotting
    return data_plot;
    

def estimateTest(images_train,labels_train,images_test, labels_test,NUMBER_OF_NEAREST_NEIGHBOR):
    #Estimate the test data now  
    SIZE_OF_TRAINING_DATA=images_train.shape[0];
    NUMBER_OF_DIMENSION=images_train.shape[1];
    SIZE_OF_TESTING_DATA=images_test.shape[0];
    #print("will run for",len(images_test),len(images_train));
    #Start testing by calculating distance 
    distance_array=np.empty([SIZE_OF_TRAINING_DATA,2]);
    testing_data_estimates=np.empty(SIZE_OF_TESTING_DATA);
    for i in range(len(images_test)):
        for training_index in range(len(images_train)):
            #i have the row by row data from images_test and im looping through the training data to find distance
            #delta=images_test[i,:]-images_train[training_index,:];
            #dist=np.dot(delta,delta);
            dist=np.sum((images_test[i,:]-images_train[training_index,:])**2,axis=None);
            distance_array[training_index,0]=training_index;
            distance_array[training_index,1]=dist;
        #now we are done with training data .. so sort and decide the output for the testing data
        
        distance_array=sorted(distance_array,key=itemgetter(1));
        
        #Based on number of K neighbors passed, need to make a decision
        #extract the part of use from the sorted distance_array
        #Iterate K vector and determine the estimate
        #Pick nearest lables (highest number of neighbors first
        nearest_labels=np.array(distance_array)[:NUMBER_OF_NEAREST_NEIGHBOR,0];        
        count=Counter(nearest_labels);        
        chosen_label=max(count,key=count.get);
        testing_data_estimates[i]=labels_train[int(chosen_label)];
        #clean distance array
        distance_array=np.empty([SIZE_OF_TRAINING_DATA,2]);
    
    return testing_data_estimates;
    
def kNN(images_train, labels_train, images_test,labels_test,k):
    #Calculate accuracy for the provided images_test using the k that is passed.
    estimated_test=estimateTest(images_train, labels_train,images_test,labels_test,k);
    #Match the returned estimates with the label to determine accuracy
    definition_images=np.zeros((10,4));
    #np.zeros(definition_images);
    for rows in range(len(definition_images)):
        definition_images[rows,0]=rows;
    
            
    for i in range(len(estimated_test)):
        if(labels_test[i]==estimated_test[i]):
            #It is a match
            definition_images[labels_test[i],1]+=1;
            definition_images[labels_test[i],2]+=1;
            definition_images[labels_test[i],3]= (((definition_images[labels_test[i],3])*(definition_images[labels_test[i],2]-1))+1)/(definition_images[labels_test[i],2]);
        else:
            #No match
            definition_images[labels_test[i],2]+=1;
            definition_images[labels_test[i],3]= (((definition_images[labels_test[i],3])*(definition_images[labels_test[i],2]-1)))/(definition_images[labels_test[i],2]);
    
            
    #Now I have accuracy array with all the hits and misses and can calculate the accuracy based on counts of total occurrences
    #Accuracy is returned. Calculate Average accuracy
    sum_product=np.inner(definition_images[:,2],definition_images[:,3]);
    average_accuracy=sum_product/1000;
                
    return definition_images[:,3],average_accuracy;
    
    
main()