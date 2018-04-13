
import numpy as np
import scipy.spatial.distance as scdist
import scipy
from scipy import ndimage
import scipy.io as scio
from operator import itemgetter,attrgetter
from collections import Counter
from scipy.stats import itemfreq
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from pathlib import Path
from random import randint

def main():
        millis_start = int(round(time.time() * 1000));
        eta=0.8;
        c=0.5;
        epochs=5;
        figure_count=0;
	
        print("Start",millis_start);
	# load data
        file_path="/home/fahad/data/MNIST_digit_data.mat";
        #file_path=input("Enter MNIST file path: ");
        if(Path(file_path).exists()==False):
                print("File does not exist");
                return;

        matdata=scio.loadmat(file_path);
        labels_test=np.array(matdata['labels_test']);
        labels_train=np.array(matdata['labels_train']);
        images_test=np.array(matdata['images_test']);
        images_train=np.array(matdata['images_train']);
        
        print("data",labels_train.shape);

        images_train_one_six,labels_train_one_six=shuffle_in_unison(images_train, labels_train,1000,1,6);
        images_test_one_six,labels_test_one_six=shuffle_in_unison(images_test, labels_test,1000,1,6);
        altered_labels_train=alter_label(labels_train_one_six,1,6);
        altered_labels_test=alter_label(labels_test_one_six,1,6);
        
        #Now i have the training data and labels where 1 is 1 and 6 is -1
        ret_weights,err,list_weights=train_binary_svm(images_train_one_six, altered_labels_train,eta,epochs,c);
        #print("final weights",ret_weights[100:120]);
        #print("err list", err );
        #print("weight list",len(list_weights));
        #plot accuracy
        figure_count+=1;
        plt.figure(figure_count);
        plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw3-3.png","Accuracy","Iterations","Distribution across epoch for 1 and 6 using binary SVM");
        
        #4. sort for all ones before 6
        sorted_train_one_six,sorted_labels_train_one_six=sort(images_train_one_six,labels_train_one_six,1,6,1000,1000);
        sorted_altered_labels_train_one_six=alter_label(sorted_labels_train_one_six,1,6);
        ret_weights,err,list_weights=train_binary_svm(sorted_train_one_six, sorted_altered_labels_train_one_six,eta,epochs,c);
        #plot accuracy
        figure_count+=1;
        plt.figure(figure_count);
        plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw3-4.png","Accuracy","Iterations","Distribution across epoch for sorted 1 and 6 using binary SVM");
        
        #5. Multi-class SVM
        ret_class_weights,ret_err, ret_corrects=train_multiclass_svm(images_train[:4000],labels_train[:4000], eta, epochs,c);
        #print("weights",len(ret_class_weights), "len of err",ret_err);
        conf_matrix,accuracy_arr,mistakes_mat=test_multiclass_svm(images_test[:2500],labels_test[:2500],ret_class_weights);
        norm_conf_matrix,average_accuracy=normalize_conf_matrix(conf_matrix);
        print("normalized confusion matrix",norm_conf_matrix);
        print("avg accuracy",average_accuracy);
        
        #7. Display top mistakes and show images??
        mistakes_mat[~np.all(mistakes_mat==0, axis=1)];
        sort_mistake_mat=sorted(mistakes_mat, key=lambda x:x[0]);
        
        #print("shape of mistakes",np.array(sort_mistake_mat).shape);
        shape_mistake=np.array(sort_mistake_mat).shape[0];
        #select worst 20
        figure_count+=1;
        plt.figure(figure_count);
        
        list_worst=[];
        fig_num=1;
        for i in range(20):
                plt.subplot(4,5,fig_num);
                plt.axis('off');
                #print("which index interested",sort_mistake_mat[shape_mistake-i-1][3]);
                imag=images_test[sort_mistake_mat[shape_mistake-i-1][3]];
                rot_img=np.fliplr(np.array(imag).reshape((28,28)));
                final_rot_img=scipy.ndimage.rotate(rot_img,90);
                plt.imshow(final_rot_img,cmap='gray',interpolation='nearest');
                predict_string='Predicted:'+str(sort_mistake_mat[shape_mistake-i-1][1]);
                actual_string='Actual:'+str(sort_mistake_mat[shape_mistake-i-1][2]);
                #plt.set_x_label(actual_string,fontsize=8);
                #plt.set_y_label(predict_string,fontsize=8);
                plt.text(0,0, predict_string,fontsize=7);
                plt.text(0,0,actual_string,fontsize=7, horizontalalignment='right', verticalalignment='top',rotation='vertical');
                fig_num+=1;

        #list_worst.append(sort_mistake_mat[]);
        plt.savefig("worst-images.png");	



def sort(train,label,digit1,digit2,data_size1,data_size2):
	digit1_train_data=[];
	digit2_train_data=[];
	digit1_label_data=[];
	digit2_label_data=[];
	for i in range(len(train)):
		if(label[i]==digit1):
			digit1_train_data.append(train[i]);
			digit1_label_data.append(label[i]);
		elif(label[i]==digit2):
			digit2_train_data.append(train[i]);
			digit2_label_data.append(label[i]);
	
	return np.array(digit1_train_data[:data_size1-1]+digit2_train_data[:data_size2-1]),np.array(digit1_label_data[:data_size1-1]+digit2_label_data[:data_size2-1]);

def plot_accu(test_data, test_label,list_weights,fig_name,yLabel,xLabel,title):
	accuracy=[];
	rowIndex=0;
	iterations=0;
	#Plot the chart for this 
	plt.ylabel(yLabel);
	plt.xlabel(xLabel);
	plt.title(title);
	plt.hold(True);
	plt.grid(True);
	plt.ylim((0,1));
	data_plot=np.empty((len(list_weights),2));
	print(data_plot.shape,len(list_weights));
	    
	for iter_weights in list_weights:
		rowIndex=0;
		accuracy=[];
                #iter_weights_np=np.array(iter_weights);
                #print("weights",iter_weights);
		for row in test_data:
                        if (test_label[rowIndex]*np.dot(row,iter_weights[1:])<1):
                                #Misclassified
                                accuracy.append(0);
                        else:
                                accuracy.append(1);
                        rowIndex+=1;
		data_plot[iterations,0]=iterations+1;
		data_plot[iterations,1]=sum(accuracy)/len(accuracy);
		#print("accuracy variation",data_plot[iterations,1]);
		iterations+=1;
		
	print("Final Accuracy",data_plot[iterations-1,1], "iterations",data_plot.shape);
	#Smooth the fit line ??
	plt.plot(data_plot[:,0],data_plot[:,1],'xb-');
	plt.savefig(fig_name);
	#plt.show();

def test_multiclass_svm(test_data, test_label_unaltered,list_class_weights):
        #extract final weights for each digit for using
        digits=[0,1,2,3,4,5,6,7,8,9];
        weight_dict={};
        last_weights={};
        accuracy_dict={};
        accuracy_arr=np.zeros((10,3));
        confusion_matrix=np.zeros((10,10));
        #wt_x=np.zeros((test_data.shape[0],2));
        temp_dist=np.zeros((10,2));
        mistakes_mat=np.zeros((test_data.shape[0],4));
        for i in range(10):
                accuracy_arr[i][0]=i;
                temp_dist[i][0]=i;                
                
        #print("temp dist start",temp_dist);
        for digit in digits:
                temp=list_class_weights[digit];
                last_weight=temp[len(temp)-1];
                #print("sum of weights",sum(last_weight));
                last_weights[digit]=last_weight;
                last_weight=None;
        
        
        #each test data needs to be run against all the final weights 
        rowIndex=0;
        row_change_flag=False;
        for row in test_data:
                row_change_flag=True;
                for i in range(10):
                        temp_dist[i][1]=0;
                classified_flag=False;
                #print("temp dist after zeroing",temp_dist);
                for dig,iter_weights in last_weights.items():
                        ##Try simple test to check for prediction
                        if((np.dot(row,iter_weights[1:])>=1)):
                           #it predicts this digit
                           accuracy_arr[dig][1]+=1;
                           confusion_matrix[test_label_unaltered[rowIndex].item()][test_label_unaltered[rowIndex].item()]+=1;
                           classified_flag=True;
                        elif(((np.dot(row,iter_weights[1:])<1)) and (((np.dot(row,iter_weights[1:])>=0)))):
                             confusion_matrix[test_label_unaltered[rowIndex].item()][dig]+=1;
                             mistakes_mat[rowIndex][0]= np.dot(row,iter_weights[1:]);
                             mistakes_mat[rowIndex][1]= dig;
                             mistakes_mat[rowIndex][2]= test_label_unaltered[rowIndex].item();
                             mistakes_mat[rowIndex][3]=rowIndex;
                             #These are the mistakes
                       # if(mistakes_mat[rowIndex][0]>np.dot(row,iter_weights[1:])):
                        #        mistakes_mat[rowIndex][0]= np.dot(row,iter_weights[1:]);
                         #       mistakes_mat[rowIndex][1]= dig;
                          #      mistakes_mat[rowIndex][2]= test_label_unaltered[rowIndex].item();
                          #      mistakes_mat[rowIndex][3]=rowIndex;
                             #classified_flag=True;
                           #accuracy_arr[dig][2]=np.dot(row,iter_weights[1:]);

                        #print("seq of digits",dig);
                        temp_dist[dig][1]=np.dot(iter_weights[1:],row);

                        row_change_flag=False;
                        

                #Now i work with the temp_dist to determine which is the best match out of 10 training
                #print("temp_dist before",temp_dist);
                sort_temp_dist=sorted(temp_dist,key=itemgetter(1));
                #print("temp_distafter ",sort_temp_dist);
                #if(classified_flag == False):
                        #confusion_matrix[test_label_unaltered[rowIndex].item()][sort_temp_dist[9][0]]+=1;
                #confusion_matrix[test_label_unaltered[rowIndex].item()][wt_x[rowIndex][0]]+=1;
                rowIndex+=1;
        #print("conf matrix",confusion_matrix);
        #print("accu arr",accuracy_arr);
        #print("sum",np.sum(confusion_matrix));
        return confusion_matrix,accuracy_arr,mistakes_mat;

def normalize_conf_matrix(conf_matrix):
        rows=conf_matrix.shape[0];
        sum_row=1;
        average_accuracy=0.;
        for i in range(0,rows):
                sum_row=np.sum(conf_matrix[i][:]);
                #print("elements before norm",conf_matrix[i][i]);
                if(sum_row!=0):
                        conf_matrix[i][:]=conf_matrix[i][:]/sum_row;
                #print("printing elements of importance",i,"  ",conf_matrix[i][i]);
                average_accuracy+=conf_matrix[i][i];
        #print("sums norm ",sum(conf_matrix));
        return conf_matrix,average_accuracy/rows;

def train_multiclass_svm(train_data,train_label_unaltered, eta, epochs,c):
        errors={"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0};
        corrects={"0":0,"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0};
        train_digit=[0,1,2,3,4,5,6,7,8,9];
        list_classification_weights={};
            
        #Make 1 predictor digit as positive 1 and rest as -1 and accordingly, keep separating by calling binary svm each time
        for digit in train_digit:
                alt_train_label=alter_label_multi(train_label_unaltered,digit);
                working_train_label=np.copy(train_label_unaltered);
                for x in np.nditer(working_train_label,op_flags=['readwrite']):
                        if(x==digit):
                                #change it to 1 
                                x[...]=1;
                #corrects[digit]=corrects[digit]+1;
                        else:
                                #change it to -1
                                x[...]=-1;
                #Now I have a ready working train label for training the weights using binary
                ret_wts, err, list_wts=train_binary_svm(train_data,alt_train_label,eta,epochs,c);
                #print("err dict",err);
                #errors[digit]=corrects[digit]-err['1'];
                list_classification_weights[digit]=list_wts;
                
        #When i am out of loop.. I will have list_wts for each digit against the rest 
        return list_classification_weights,errors,corrects;

def train_binary_svm(train_data, train_label, eta,epochs,c):
        list_weights=[];
        weights=np.zeros(1+train_data.shape[1]);
        rowIndex=0;
        errors={"1":0,"-1":0};
        weights[0]=0; #bias term
        count_iteration=0;
        for epoch in range(0,epochs):
                rowIndex=0;
                for row in train_data:
                        count_iteration+=1;
                        #print("ever",np.dot(weights[1:],row));
                        if(train_label[rowIndex]*np.dot(weights[1:],row)<1):
                                #For hinge loss -- can modify to regression loss
                                weights[1:]+=((c*(row*train_label[rowIndex])+(-2*eta*(1/(count_iteration))*weights[1:])));
                                if(train_label[rowIndex]==1):
                                        errors["1"]+=1;
                                else:
                                        errors["-1"]+=1;
                        else:
                                weights[1:]+=eta*(-2*(1/(count_iteration))*weights[1:]);
                        rowIndex+=1;
                        list_weights.append(weights.tolist());
        return weights,errors,list_weights;




def alter_label_multi(labels,digit):
	new_label=[];
	for x in range(0,labels.shape[0]):
		if(labels[x]==digit):
			new_label.append(1);
		else:
			new_label.append(-1);
	temp_label=np.array(new_label);
	return temp_label;

def alter_label(labels,digit1, digit2):
	new_label=[];
	for x in range(0,labels.shape[0]):
		if(labels[x]==digit1):
			new_label.append(1);
		elif(labels[x]==digit2):
			new_label.append(-1);
	temp_label=np.array(new_label);
	return temp_label;

def shuffle_in_unison(data,labels,number_of_data,digit1,digit2):
	#Create 2 sets of data and then shuffle
	temp_data=[];
	temp_label=[];
	count_digit1=0;
	count_digit2=0;
	count=0;
	for labelrow in labels:
		if(labelrow==digit1 and count_digit1<number_of_data/2):
			temp_data.append(data[count]);
			temp_label.append(labelrow);
			count_digit1+=1;
		elif(labelrow==digit2 and count_digit2<number_of_data/2):
			temp_data.append(data[count]);
			temp_label.append(labelrow);
			count_digit2+=1;
		count+=1;
		
	temp_data_new=np.array(temp_data);
	temp_label_new=np.array(temp_label);
	print("size",temp_data_new.shape);
	shuff_state=np.random.get_state();
	np.random.shuffle(temp_data_new);
	np.random.set_state(shuff_state);
	np.random.shuffle(temp_label_new);
	return temp_data_new[:number_of_data], temp_label_new[:number_of_data];

main()
