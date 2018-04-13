import numpy as np
import scipy.spatial.distance as scdist
import scipy
from scipy import ndimage
import scipy.io as scio
from operator import itemgetter,attrgetter
from collections import Counter
from collections import OrderedDict
from scipy.stats import itemfreq
import time
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from pathlib import Path
from random import randint
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import sklearn.decomposition as skd
from  sklearn.neural_network import MLPClassifier
from sklearn.utils.extmath import randomized_svd


def main():
        millis_start = int(round(time.time() * 1000));
        eta=0.8;
        c=0.5;
        epochs=1;
        figure_count=0;
	
        print("Start",millis_start);
	# load data
        #file_path="/home/fahad/code/ML/MNIST_digit_data.mat";
        file_path="MNIST_digit_data.mat";
        #file_path=input("Enter MNIST file path: ");
        if(Path(file_path).exists()==False):
                print("File does not exist in current directory");
                file_path=input("Enter MNIST file path: ");
                #return;

        matdata=scio.loadmat(file_path);
        labels_test=np.array(matdata['labels_test']);
        labels_train=np.array(matdata['labels_train']);
        images_test=np.array(matdata['images_test']);
        images_train=np.array(matdata['images_train']);
        
           #These are the data sets to be used for everything
        shuff_train_images_1000, shuff_train_labels_1000=shuffle_in_unison(images_train, labels_train,1000);

        

        #3. Train SVM
        svm=LinearSVC();
        svm_model_wts=svm.fit(shuff_train_images_1000, shuff_train_labels_1000);
        #weights=svm.coef_;
        predicted=svm.predict(images_test);
        print("predicted", predicted.shape);
        #print("actual labels",labels_test);
        conf_matrix=np.zeros((10,10));
        accuracy=0;
        for i in range(predicted.shape[0]):
                conf_matrix[labels_test[i][0]][predicted[i]]+=1;
        #cnf_matrix=confusion_matrix(labels_test[0:4], predicted);
        for j in range(10):
                accuracy+=conf_matrix[j][j];

        print(np.sum(conf_matrix), " Accuracy:",accuracy/(np.sum(conf_matrix)));

        #4. Access m to get weight values
        #print("weight",svm.coef_.shape);
        confusion_matrix=np.zeros((10,10));
        #print("c matrix",confusion_matrix.shape);
        idx=0;
        count_wts=0;
        misclassify=0;
        misclassify_records=[];
        for row in images_test:
                #print("looping test",idx);
                count_wts=0;
                flag=True;
                #weights_missclassified=[];
                prev_wt=0;
                pred_idx=0;
                for weights in svm.coef_:
                        #print("looping",count_wts);
                        #if(labels_test[idx]==count_wts):
                        
                        if(np.dot(weights,row)>prev_wt):
                                prev_wt=np.dot(weights,row);
                                pred_idx=count_wts;
                        #weights_missclassified.append(np.dot(weights,row));
                        if(np.dot(weights,row)>=1 and flag):
                                        #print("labels",labels_test[idx][0]);
                                confusion_matrix[labels_test[idx][0]][count_wts]+=1;
                                flag=False;
                                #else:
                                       # confusion_matrix[labels_test[idx][0]][count_wts]+=1;
                        if(flag and count_wts==9):
                                #record is misclassified by all
                                misclassify+=1;
                                confusion_matrix[labels_test[idx][0]][pred_idx]+=1;
                                #misclassify_records.append(row);
                        count_wts+=1;
                idx+=1;
        #print(confusion_matrix);

                
        #print("total sum",sum(sum(confusion_matrix)));
        accuracy=0;
        for j in range(10):
                accuracy+=confusion_matrix[j][j];
        #print("Accuracy:",accuracy/labels_test.shape[0]);

        
        #5. PCA on training data
        pca=PCA(n_components=50);
        pca.fit(images_train);
        images_train_reduced=pca.transform(images_train);
        #print("shape of PCA",images_train_reduced.shape);

        #manually reduce the dimension using SVD
        mean=np.mean(images_train,axis=1);
        print("mean shape", mean.shape);
        mean_vec=mean.reshape((mean.shape[0],1));
        #N=images_train.shape[0];
        
        norm_images_train=images_train-mean_vec;

        mean_test=np.mean(images_test,axis=1);
        mean_vec_test=mean_test.reshape((mean_test.shape[0],1));
        norm_images_test=images_test-mean_vec_test;
        print("shape of norm data",norm_images_train.shape);
        
        svd=skd.TruncatedSVD(n_components=50);
        reduced_images_train=svd.fit_transform(norm_images_train);
        
        #reduced_images_train=svd.transform(norm_images_train);
        print("reduced",reduced_images_train.shape);
        retrans_images_train=svd.inverse_transform(reduced_images_train);
        print("unreduced",retrans_images_train.shape);
        #Now compare to original data -- How??
        #pick random idxes for 20 images and plot alongside
        indexes=random.sample(range(0,norm_images_train.shape[0]),20);
        indexes=sorted(indexes);
        #print("indexes",indexes);
        fig_num=1;
        figure_count+=1;
        plt.figure(figure_count);
        for i in range(0,2):
                #figure_count+=1;
                for idxs in indexes:
                        plt.subplot(8,5,fig_num);
                        plt.axis('off');
                        if(i==0):
                                #print("which idx",idxs);
                                imag=norm_images_train[idxs,:];
                        else:
                                #print("trans",idxs);
                                imag=retrans_images_train[idxs,:];
                        rot_img=np.fliplr(np.array(imag).reshape((28,28)));
                        final_rot_img=scipy.ndimage.rotate(rot_img,90);
                        plt.imshow(final_rot_img,cmap='gray',interpolation='nearest');
                        #plt.tight_layout();
                        fig_num+=1;
                #plt.text(0,0,actual_string,fontsize=7);

        plt.savefig("comparison-20.png");	

        vecs_eig,list_mse,list_dim=project_data(norm_images_train,500);
        #plot mses
        figure_count+=1;
        plt.figure(figure_count);
        plt.ylabel("Mean Square Errors");
        plt.xlabel("Dimensions");
        plt.title("MSE across reduced dimensions");
        plt.hold(True);
        plt.grid(True);
        #plt.ylim((0,1));
        plt.plot(list_dim,list_mse,'xb-');
        plt.savefig("hw4-5-mse.png");

        #Display eigs
        fig_num=1;
        figure_count+=1;
        plt.figure(figure_count);
        for i in range(vecs_eig.shape[0]):
                row=vecs_eig[i,:];
                plt.subplot(2,5,fig_num);
                plt.axis('off');
                rot_img=np.fliplr(vecs_eig[i,:].reshape((28,28)));
                final_rot_img=scipy.ndimage.rotate(rot_img,90);
                plt.imshow(final_rot_img,cmap='gray',interpolation='nearest');
                fig_num+=1;
        plt.savefig("hw4-5-eigs.png");
        #return;

        #6. project and train - should i be using normed data?
        dimension_list=[2,5,10,20,30,50,70,100,150,200,250,300,400,500,748];
        #dimension_list=[400,500,748];
        #print("dim length", len(dimension_list));
        dim_acc_svm=np.empty((len(dimension_list),2));
        dim_acc_mlp=np.empty((len(dimension_list),2));
        counter=0;
        for dims in dimension_list:
                accu=train_svm(norm_images_train[:10000],labels_train[:10000],norm_images_test[:3000], labels_test[:3000],dims);
                accu_mlp=train_mlp(norm_images_train[:10000],labels_train[:10000],norm_images_test[:3000], labels_test[:3000],dims);
                dim_acc_svm[counter,0]=dims;
                dim_acc_svm[counter,1]=accu;
                dim_acc_mlp[counter,0]=dims;
                dim_acc_mlp[counter,1]=accu;
                counter+=1;

        #print("dimension",dim_acc_svm);
        #print("dim mlp",dim_acc_mlp);
        figure_count+=1;
        plt.figure(figure_count);
        plt.ylabel("Accuracy");
        plt.xlabel("Dimensions");
        plt.title("Distribution of accuracy across dimensions");
        plt.hold(True);
        plt.grid(True);
        plt.ylim((0,1));
        plt.plot(dim_acc_svm[:,0],dim_acc_svm[:,1],'xb-');
        plt.savefig("hw4-7.png");
        #return;
        #8. implementaion of neural network
        figure_count+=1;
        plt.figure(figure_count);
        plt.ylabel("Accuracy");
        plt.xlabel("Dimensions");
        plt.title("Distribution of accuracy across dimensions");
        plt.hold(True);
        plt.grid(True);
        plt.ylim((0,1));
        plt.plot(dim_acc_mlp[:,0],dim_acc_mlp[:,1],'xb-');
        plt.savefig("hw4-8.png");

        

def train_mlp(train,labels_train, test, labels_test, dims):
        #svd=skd.TruncatedSVD(n_components=dims);
        pca=PCA(n_components=dims);
               
        pca.fit(train);
        pca_train=pca.transform(train);
        inv_pca_train=pca.inverse_transform(pca_train);

        #new_train=svd.fit_transform(train);
        #inv_train=svd.inverse_transform(new_train);
        #new_train=svd.transform(train);
        #new_test=svd.fit_transform(test);
        #new_test=svd.transform(test);
        clf=MLPClassifier();
        mlp_model=clf.fit(inv_pca_train,labels_train);
        #print("mlp model",mlp_model.shape);
        res=clf.predict(test);
        print("shape of res",res.shape);
        conf_matrix=np.zeros((10,10));
        accuracy=0;

        for i in range(res.shape[0]):
                conf_matrix[labels_test[i][0]][res[i]]+=1;
        
        #cnf_matrix=confusion_matrix(labels_test[0:4], predicted);
        for j in range(10):
                accuracy+=conf_matrix[j][j];

        accuracy=accuracy/(np.sum(conf_matrix));
        print(np.sum(conf_matrix), " Accuracy:",accuracy);
        return accuracy;
                
                
def train_svm(train,labels_train, test, labels_test, dims):
        if dims==0:
                #Train as is
                new_train=train;
                new_test=test;
        else:
               svd_t=skd.TruncatedSVD(n_components=dims);
               pca=PCA(n_components=dims);
               
               pca.fit(train);
               pca_train=pca.transform(train);
               inv_pca_train=pca.inverse_transform(pca_train);
               #pca.fit(test);
               #pca_test=pca.transform(test);
               svd_t.fit(train);
               new_train=svd_t.transform(train);
               
               #new_train=svd.transform(train);
               svd_t.fit(test);
               new_test=svd_t.transform(test);
               #new_test=svd.transform(test);

        svm=SVC(decision_function_shape='ovo');
       # svm=LinearSVC(max_iter=10000);
        svm.fit(inv_pca_train, labels_train);
        predicted=svm.predict(test);
        ac=svm.score(test,labels_test);
        print("Scoring ",ac);
        #print("predicted", predicted.shape);
        return ac;
                
def project_data(train_data, dimensions):
        list_mse=[];
        list_dim=[];
        dims=np.round(np.logspace(np.log10(1),np.log10(dimensions),50));
        dim_u_list=list(set(dims.tolist()));
        #print("final dim",sorted(dim_u_list));
        #for idx,i in np.ndenumerate(dims):
        #for i in range(1,dimensions+1):
        for i in sorted(dim_u_list):
                #print("dims",i);
                svd_t=skd.TruncatedSVD(n_components=i);
                t_train_data=svd_t.fit_transform(train_data);
                #U,sigma,VT=randomized_svd(train_data,n_components=i,n_iter=10,random_state=None);
                #print("U",U.shape,"sigma",sigma.shape,"VT",VT.shape);
                retrans_train_data=svd_t.inverse_transform(t_train_data);

                #t_train_data=svd_t.transform(train_data);
                #Find mean squares
                #print("shapes",train_data.shape, t_train_data.shape);
                mse=mean_squared_error(train_data,retrans_train_data);
                
                #print("mse",mse);
                list_dim.append(i);
                list_mse.append(mse);
        #Eigenvectors
        svd_eig=skd.TruncatedSVD(n_components=10);
        Utr=svd_eig.fit_transform(train_data);
        print("utr",Utr.shape," ",(svd_eig.components_).shape);
        eig_vecs_to_display=svd_eig.components_;

        
        eigs=Utr.dot(svd_eig.components_);
        #print("eigs",eigs.shape);
        #pca=PCA();
        #pca.fit(train_data);
        #eig_vecs=pca.components_;
        #eig_values=pca.singular_values_;
        #print("eig vecs",eig_vecs.shape);
        #print("eig values",eig_values.shape);
        
        return eig_vecs_to_display,list_mse,list_dim;


def shuffle_in_unison(data,labels,number_of_data):
	#Create 2 sets of data and then shuffle
	shuff_state=np.random.get_state();
	np.random.shuffle(data);
	np.random.set_state(shuff_state);
	np.random.shuffle(labels);
	return data[:number_of_data], labels[:number_of_data];


main()
