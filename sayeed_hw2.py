#!/usr/bin/env python3

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
	eta=0.1;
	epochs=10;
	
	print("Start",millis_start);
	# load data
	#file_path="/home/siddiqui/umbc/ml/MNIST_digit_data.mat";
	file_path=input("Enter MNIST file path: ");
	if(Path(file_path).exists()==False):
		print("File does not exist");
		return;

	matdata=scio.loadmat(file_path);
	labels_test=np.array(matdata['labels_test']);
	labels_train=np.array(matdata['labels_train']);
	images_test=np.array(matdata['images_test']);
	images_train=np.array(matdata['images_train']);

	images_train_one_six,labels_train_one_six=shuffle_in_unison(images_train, labels_train,1000,1,6);
	images_test_one_six,labels_test_one_six=shuffle_in_unison(images_test, labels_test,1000,1,6);
	
	altered_labels_train=alter_label(labels_train_one_six,1,6);
	altered_labels_test=alter_label(labels_test_one_six,1,6);

	millis_start_5b_train = int(round(time.time() * 1000));
	ret_weights,error,list_weights=train(images_train_one_six,altered_labels_train,eta,epochs);
	train_time_5b = (int(round(time.time() * 1000)) - millis_start_5b_train);
	print("5b training time:",train_time_5b);
	millis_start_5c_testing = int(round(time.time() * 1000));
	plt.figure(1);
	plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw2-5c.png","Accuracy","Iterations","Accuracy Distribution across iterations for 1 and 6");
	test_time_5c = int(round(time.time() * 1000)) - millis_start_5c_testing;
	print("5c testing time:",test_time_5c);
	
	#5d . Visualize
	final_learned_weight=list_weights[len(list_weights)-1];
	plt.figure(2);
	visualize(final_learned_weight,"5d-1-and-6",2);
	
	#5e. Visualization of 20 best scoring and 20 worst scoring images
	best1,best6,worst1,worst6=score_test_images(list_weights[len(list_weights)-1],images_test_one_six,altered_labels_test);
	print("shapes best 1",np.array(best1).shape,"6",np.array(worst6).shape);
		
	#Iterate through the images and create the subplot
	subplotting(3,best1,"5e-20-best-1-images");
	subplotting(4,worst1,"5e-20-worst-1-images");
	subplotting(5,best6,"5e-20-best-6-images");
	subplotting(6,worst6,"5e-20-worst-1-images");
	#plt.figure(3);
		

	#5f. Flip the label for 10% of 1 and 6 data
	flipped_label_one_six=flip(altered_labels_train,0.10);
	#train a new model with flipped labels
	millis_start_5f_training = int(round(time.time() * 1000));
	ret_weights,error,list_weights=train(images_train_one_six,flipped_label_one_six,eta,epochs);
	training_time_5f = int(round(time.time() * 1000)) - millis_start_5f_training;
	print("5f training time:",training_time_5f);
	#print("size of weights",sum(list_weights[3]));		
	millis_start_5f_testing = int(round(time.time() * 1000));
	plt.figure(7);
	plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw2-5f.png","Accuracy","Iterations","Accuracy Distribution across iterations for flipped labels for 10% for 1 and 6");
	testing_time_5f = int(round(time.time() * 1000)) - millis_start_5f_testing;
	print("5f testing time:",testing_time_5f);

	#5g. Sort data for 1 and 6 and plot accuracy and note the time
	millis_start_5g_training = int(round(time.time() * 1000));
	sorted_train_one_six,sorted_labels_train_one_six=sort(images_train_one_six,labels_train_one_six,1,6,500,500);
	#Test sorted
	ret_weights,error,list_weights=train(sorted_train_one_six,sorted_labels_train_one_six,eta,epochs);
	
	training_time_5g = int(round(time.time() * 1000)) - millis_start_5g_training;
	print("5g training time:",training_time_5g);
	
	#Test the plot for sorted stuff and note time
	millis_start_5g_testing = int(round(time.time() * 1000));
	plt.figure(8);	
	plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw2-5g.png","Accuracy","Iterations","Accuracy Distribution across iterations for sorted 1 and 6");
	testing_time_5g = int(round(time.time() * 1000)) - millis_start_5g_testing;
	print("5g testing time:",testing_time_5g);
	
	#Repeat process for 2 and 8. Note time
	
	images_train_two_eight,labels_train_two_eight=shuffle_in_unison(images_train, labels_train,1000,2,8);
	images_test_two_eight,labels_test_two_eight=shuffle_in_unison(images_test, labels_test,1000,2,8);
	altered_labels_train_two_eight=alter_label(labels_train_two_eight,2,8);
	altered_labels_test_two_eight=alter_label(labels_test_two_eight,2,8);

	millis_start_5h_training = int(round(time.time() * 1000));
	ret_weights,error,list_weights=train(images_train_two_eight,altered_labels_train_two_eight,eta,epochs);
	training_time_5h = int(round(time.time() * 1000)) - millis_start_5h_training;
	print("5h training time:",training_time_5h);
	
	millis_start_5h_testing = int(round(time.time() * 1000));
	plt.figure(9);
	plot_accu(images_test_two_eight,altered_labels_test_two_eight,list_weights,"hw2-5h.png","Accuracy","Iterations","Accuracy Distribution across iterations for 2 and 8");
	testing_time_5h = int(round(time.time() * 1000)) - millis_start_5h_testing;
	print("5h testing time:",testing_time_5h);

	#Visualize
	final_learned_weight=list_weights[len(list_weights)-1];
	plt.figure(10);
	visualize(final_learned_weight,"5d-2-and-8",6);

	#5i. Repeat 1 and 6 for 10 training examples and 60K training examples
	images_train_one_six_10,labels_train_one_six_10=shuffle_in_unison(images_train, labels_train,10,1,6);
	images_train_one_six_12000,labels_train_one_six_12000=shuffle_in_unison(images_train, labels_train,12500,1,6);
	#images_test_one_six,labels_test_one_six=shuffle_in_unison(images_test, labels_test,1000,[1,6]);
	
	altered_labels_train_10=alter_label(labels_train_one_six_10,1,6);
	altered_labels_train_12000=alter_label(labels_train_one_six_12000,1,6);
	#I have the data now 
	millis_start_5i_train_10 = int(round(time.time() * 1000));
	
	ret_weights,error,list_weights=train(images_train_one_six_10,altered_labels_train_10,eta,epochs);
	train_time_10_5i = int(round(time.time() * 1000)) - millis_start_5i_train_10;
	print("5i training time for 10:",train_time_10_5i);

	millis_start_5i_testing_10 = int(round(time.time() * 1000));
	plt.figure(11);
	plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw2-5i-10.png","Accuracy","Iterations","Accuracy Distribution across 10 iterations for 1 and 6");
	testing_time_10_5i = int(round(time.time() * 1000)) - millis_start_5i_testing_10;
	print("5i testing time for 10:",testing_time_10_5i);

	millis_start_5i_train_12000 = int(round(time.time() * 1000));
	ret_weights,error,list_weights=train(images_train_one_six_12000,altered_labels_train_12000,eta,epochs);
	train_time_12000_5i = int(round(time.time() * 1000)) - millis_start_5i_train_12000;
	print("5i training time for 12000:",train_time_12000_5i);
	millis_start_5i_testing_12000 = int(round(time.time() * 1000));
	plt.figure(12);
	plot_accu(images_test_one_six,altered_labels_test,list_weights,"hw2-5i-12000.png","Accuracy","Iterations","Accuracy Distribution across 12000 iterations for 1 and 6");
	testing_time_12000_5i = int(round(time.time() * 1000)) - millis_start_5i_testing_12000;
	print("5i testing time for 12000:",testing_time_12000_5i);

def score_test_images(final_weight,images_test,labels_test):
	correct_score_list1=[];
	correct_score_list6=[];
	squared_error=0;

	#Test whether the prediction is correct. and prepare a list of wrong predictors
	rowIndex=0;
	for row in images_test:
		ret_prediction=predict(row,final_weight);
		if(ret_prediction==labels_test[rowIndex]):
			#pREDICTtion is correct. Calculate the error
				if(ret_prediction==1):
					#it is 1
					for i in range(0,len(final_weight[1:])-1):
						squared_error+=(final_weight[i]-row[i])**2;
					correct_score_list1.append([rowIndex,squared_error]);
					squared_error=0;	
				else:
					#It is 6
					for i in range(0,len(final_weight[1:])-1):
						squared_error+=(final_weight[i]-row[i])**2;
					correct_score_list6.append([rowIndex,squared_error]);
					squared_error=0;	
				
		rowIndex+=1;
	#Sort the 4 lists
	#correct_score_list1.sort();
	#correct_score_list6.sort();
	sorted_correct_score_list1=sorted(correct_score_list1,key=lambda x:x[1]);
	sorted_correct_score_list6=sorted(correct_score_list6,key=lambda x:x[1]);

	#shape_correct1=np.array(correct_score_list1).shape[0];
	#shape_correct6=np.array(correct_score_list6).shape[0];

	#sorted_correct_score_list1=sorted(correct_score_list1, key=lambda x:x[1]);
	#sorted_correct_score_list6=sorted(correct_score_list6, key=lambda x:x[1]);
	#Pick the top 20, get their rowIndex and create the top 20 data row frmom image data
	count=0;
	length1=len(sorted_correct_score_list1);
	length6=len(sorted_correct_score_list6);
	best1=[];
	best6=[];
	worst1=[];
	worst6=[];

	for item1 in sorted_correct_score_list1:
		if(count<20):
			best1.append(images_test[item1[0]]);
		if(count+20>= length1):
			worst1.append(images_test[item1[0]]);
		count+=1;			
	count=0;
	for item6 in sorted_correct_score_list6: 		
		if(count<20):
			best6.append(images_test[item6[0]]);
		if(count+20>= length6):
			worst6.append(images_test[item6[0]]);
		count+=1;			

	#print("size of best and worst",len(best1),len(worst6));
	return best1,best6,worst1,worst6;

def subplotting(figure,disp_list,output_name):
	plt.figure(figure);
	fig_num=1;
	for imag in disp_list:
		#print("shape of imag",np.array(imag).shape);
		plt.subplot(4,5,fig_num);
		plt.axis('off');
		fig_num+=1;
		
		rot_img=np.fliplr(np.array(imag).reshape((28,28)));
		final_rot_img=scipy.ndimage.rotate(rot_img,90);
		plt.imshow(final_rot_img,cmap='gray',interpolation='nearest');
	plt.savefig(output_name,bbox_inches='tight');	

    
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


def visualize(weight,fig_name,fig_num):
	subplot_id1=fig_num+210;
	subplot_id2=fig_num+210+1;

	positive_weight=[];
	negative_weight=[];
	for items in weight:
		if items==0:
			positive_weight.append(0);			
			negative_weight.append(0);
		elif items>0:
			positive_weight.append(items);			
			negative_weight.append(0);
		elif items<0:
			negative_weight.append(items);
			positive_weight.append(0);			

	#Visualize
	print("shape",np.array(positive_weight)[1:].shape);
	positive_weight=np.fliplr(np.array(positive_weight)[1:].reshape((28,28)));
	negative_weight=np.fliplr(np.array(negative_weight)[1:].reshape((28,28)));
	rot_positive_weight=scipy.ndimage.rotate(positive_weight,90);
	rot_negative_weight=scipy.ndimage.rotate(negative_weight,90);
	
	plt.subplot(1,2,1);
	plt.axis('off');
	plt.title("Positive and negative weight for "+fig_name)
	plt.imshow(rot_positive_weight,cmap='gray',interpolation='nearest');
	plt.subplot(1,2,2);
	plt.imshow(rot_negative_weight,cmap='gray',interpolation='nearest');
	plt.savefig(fig_name,bbox_inches='tight');	


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
	plt.ylim((0.4,1));
	data_plot=np.empty((len(list_weights),2));
	#print(data_plot.shape,len(list_weights));
	    
	for diff_weights in list_weights:
		rowIndex=0;
		#print("index",i);
		diff_weights_arr=np.array(diff_weights);
		#print("sum weights",sum(diff_weights_arr));
		accuracy=[];
		for row in test_data:
			ret_prediction=predict(row,diff_weights_arr);
			if(ret_prediction==test_label[rowIndex]):
				accuracy.append(1);
			else:
				accuracy.append(0);
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


def flip(labels,quantity):
	number_of_flips=len(labels)*quantity;
	counter=0;
	while counter<number_of_flips:
		rowIndex=randint(0,len(labels)-1);
		if(labels[rowIndex]==1):
			labels[rowIndex]=-1;
		else:
			labels[rowIndex]=1;
		counter+=1;
	return labels;

def train(data_train,labels,eta, epochs):
	#print("train size",data_train.shape);
	list_weights=[];
	weights=np.zeros(1+data_train.shape[1]);
	rowIndex=0;	
	errors=[];
	weights[0]=0;
	for num in range(epochs):
		rowIndex=0;
		for row in data_train:
			ret_prediction=predict(row,weights);
			if(labels[rowIndex]==ret_prediction):
				update=0;
			else:
				update=-ret_prediction;	
				errors.append(update);

			weights[0]=weights[0]+update*eta;
			#apply this new weight to all the columns
			for j in range(len(row)-1):
				if(update!=0):
					weights[j+1]=weights[j+1]+update*eta*row[j];
			rowIndex+=1;
			list_weights.append(weights.tolist());

	return weights,errors,list_weights;
	    

def alter_label(labels,digit1, digit2):
	new_label=[];
	for x in range(0,labels.shape[0]):
		if(labels[x]==digit1):
			new_label.append(1);
		elif(labels[x]==digit2):
			new_label.append(-1);
	temp_label=np.array(new_label);
	return temp_label;

def predict(row, weights):
	activation=weights[0];
	activation+=np.dot(weights[1:],row);
	if(activation>=0):
		return 1;
	else:
		return -1;
	
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

	#shuff_state=np.random.get_state();
	#np.random.shuffle(temp_data);
	#np.random.set_state(shuff_state);
	#np.random.shuffle(temp_label);
	return temp_data_new[:number_of_data], temp_label_new[:number_of_data];

main()


