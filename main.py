import pip
import numpy as np 
import scipy.io as sio 
import matplotlib.pyplot as plt 

from sklearn import svm
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import SVR

#from keras.models import Sequential
#from keras.layers import Dense 

from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
import cmath
import numpy
import math


from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsRegressor
#matlab檔名 

matfn='/Users/yishan/desktop/project/python/shiaas_final.mat'
#matfn='C:/Users/USER/Downloads/smallcafe_1.mat'

data=sio.loadmat(matfn) 

row1 = [[1.6,2.4],[1.6,3.775],[1.6,5.15],[1.6,6.525],[1.6,7.9],[1.6,9.275]]
row2 = [[3.2,2.4],[3.2,3.775],[3.2,5.15],[3.2,6.525],[3.2,7.9],[3.2,9.275]]
row3 = [[4.8,5.15],[4.8,6.525],[4.8,7.9],[4.8,9.275]]
test_point = [[2.4,2.4],[4.0,3.775],[2.4,5.15],[4.0,6.525],[2.4,7.9],[4.0,9.275]]
row1.extend(row2)
row1.extend(row3)
row1.extend(test_point)
coordinate = row1
#print(coordinate)



matrix_no=data['x_test']
X=matrix_no[100:,:,:]
#X1=np.hstack((X[:,:,0],X[:,:,1],X[:,:,2],X[:,:,3],X[:,:,4],X[:,:,5]))
X1=np.hstack((X[:,0,:],X[:,1,:],X[:,2,:],X[:,3,:],X[:,4,:],X[:,5,:],X[:,6,:],X[:,7,:],X[:,8,:],X[:,9,:],X[:,10,:],X[:,11,:],X[:,12,:],X[:,13,:],X[:,14,:],X[:,15,:],X[:,16,:],X[:,17,:],X[:,18,:],X[:,19,:],X[:,20,:],X[:,21,:],X[:,22,:],X[:,23,:],X[:,24,:],X[:,25,:],X[:,26,:],X[:,27,:],X[:,28,:],X[:,29,:]))



#for i in range (0,180):
#    plt.plot(k,X1[i])
#plt.show()
#X=np.hstack((matrix_no[:,0,:],matrix_no[:,1,:],matrix_no[:,2,:],matrix_no[:,3,:],matrix_no[:,4,:],matrix_no[:,5,:],matrix_no[:,6,:],matrix_no[:,7,:],matrix_no[:,8,:],matrix_no[:,9,:],matrix_no[:,10,:],matrix_no[:,11,:],matrix_no[:,12,:],matrix_no[:,13,:],matrix_no[:,14,:],matrix_no[:,15,:],matrix_no[:,16,:],matrix_no[:,17,:],matrix_no[:,18,:],matrix_no[:,19,:],matrix_no[:,20,:],matrix_no[:,21,:],matrix_no[:,22,:],matrix_no[:,23,:],matrix_no[:,24,:],matrix_no[:,25,:],matrix_no[:,26,:],matrix_no[:,27,:],matrix_no[:,28,:],matrix_no[:,29,:]))

#pca=PCA(n_components=80)
#pca.fit(X1)
#X2=pca.fit_transform(X1)
X2=X1

Y= data['y_test']
Y=Y[:,100:]
Y=Y.reshape(len(Y[0]))


y=[0]*23
x=[0]*23
for i in range (0,23):
    y[i]=0
for i in range (0,len(Y)):
    #if(Y[i]-1 < 16):
    y[Y[i]-1]+=1
for i in range(1,23):
    y[i]+=y[i-1]

x[0]=X2[0:y[0]-1,:]
x[0]=x[0][500:2000,:]
for i in range(1,23):
    x[i]=X2[y[i-1]:y[i]-1,:]
    x[i]=x[i][500:2000,:]




x_train=[0] * 16
y_train=[0] * 16
x_test = [0] * 16
y_test = [0] * 16



x_train[0],x_test[0],y_train[0],y_test[0]=train_test_split(x[0],Y[0+500:0+2000],test_size=0.2,random_state=40)

for i in range(1 , 16):
    x_train[i],x_test[i],y_train[i],y_test[i]=train_test_split(x[i],Y[y[i-1]+500:y[i-1]+2000],test_size=0.2,random_state=40)


x_test_1 = x[16][500:800]
y_test_1 = [17] * len(x_test_1)

for i in range(17 , 22):
    x_test_1 = np.vstack((x_test_1 , x[i][500:800]))
    temp = [i+1] *300
    y_test_1 = np.hstack((y_test_1 , temp))




'''
clustering=DBSCAN(eps=10,min_samples=2).fit(x_train[0])
x_pred = clustering.labels_
for i in range (0,10):
    plt.plot(kkk,x_train[0][i,:])
plt.show()
#plt.scatter(x_train[0][:,0],x_train[0][:,1],c=clustering.labels_)
'''
x1_train = x_train[0]
x1_test = x_test[0]
y1_train = y_train[0]
y1_test = y_test[0]

for i in range (1,16) :
    x1_train=np.vstack((x1_train,x_train[i]))
    x1_test = np.vstack((x1_test , x_test[i]))
    y1_train = np.hstack((y1_train , y_train[i]))
    y1_test = np.hstack((y1_test , y_test[i]))
    
for i in range (16,22):
    x1_test = np.vstack((x1_test , x[i][500:800]))
    temp = [i+1] *300
    y1_test = np.hstack((y1_test , temp))


#x_train,x_test,y_train,y_test=train_test_split(X2,Y,test_size=0.2,random_state=40)

clf=SVC(kernel='rbf',probability=True)
clf.fit(x1_train,y1_train)
'''
y1_test_predict=clf.predict(x1_test)

k=abs(y1_test_predict-y1_test)

sum=0
for i in range(0,4800):
    if k[i]==0:
        sum=sum+1;
s3=sum/len(k)

#print("linear:",s1)
#print("poly",s2)
print("rbf",s3)
#plt.plot(x_train,y_train)
'''
pro_pre=clf.predict_proba(x1_test)

y_result=[0]*6600
for k in range(0,6600):
    y_result[k]=[0]*2
    

for i in range(0,6600):
    pro_sum=0
    for j in range(0,16):    
        if pro_pre[i,j]>=0.1:
            y_result[i][0]+=pro_pre[i,j]*coordinate[j][0]
            y_result[i][1]+=pro_pre[i,j]*coordinate[j][1]
            pro_sum+=pro_pre[i,j]
    y_result[i][0]=y_result[i][0]*(1/pro_sum)
    y_result[i][1]=y_result[i][1]*(1/pro_sum)


err = [0]*6600

for j in range(0,22):
    for i in range(300*j,300*(j+1)):
        re=math.sqrt(pow(y_result[i][0] - coordinate[j][0],2)+pow(y_result[i][1]-coordinate[j][1],2))
        err[i]=re
'''
for j in range(16,22):
    for i in range(4800+1500*(j-16),4800+1500*(j-15)):
        re=math.sqrt(pow(y_result[i][0] - coordinate[j][0],2)+pow(y_result[i][1]-coordinate[j][1],2))
        err[i]=re
'''


pdf = [0,0,0,0,0,0,0,0,0,0,0]    

for i in range (0,6600):
    if err[i]<0.5:
        pdf[1] += 1
    elif err[i]<1:
        pdf[2] += 1
    elif err[i]<1.5:
        pdf[3] += 1
    elif err[i]<2:
        pdf[4] += 1
    elif err[i]<2.5:
        pdf[5] += 1
    elif err[i]<3:
        pdf[6] += 1
    elif err[i]<3.5:
        pdf[7] += 1
    elif err[i]<4:
        pdf[8] += 1
    elif err[i]<4.5:
        pdf[9] += 1
    else:
        pdf[10] += 1

cdf = pdf.copy()
for i in range(1,len(pdf)):
    cdf[i] = pdf[i] + cdf[i-1]

cdf = np.array(cdf)
cdf = cdf/6600

x_label = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

plt.grid(True)
plt.step(x_label, cdf , color= "orange",label="SVC")
plt.xlabel('error distance(m)')
plt.ylabel('CDF')
################################################





y_train_location_y=[0]*19200
y_train_location_x=[0]*19200
for i in range(0,19200):
    y_train_location_y[i]=coordinate[y1_train[i]-1][1]
    y_train_location_x[i]=coordinate[y1_train[i]-1][0]
    
#linear    
regr_x = LinearRegression()
regr_y = LinearRegression()
regr_x.fit(x1_train, y_train_location_x)
regr_y.fit(x1_train, y_train_location_y)   
pre_linear_x= regr_x.predict(x1_test)
pre_linear_y= regr_y.predict(x1_test)

#svr
clf_svr=SVR(kernel='rbf')
clf_svr.fit(x1_train,y_train_location_y)
predict_y_svr=clf_svr.predict(x1_test)
clf_svr.fit(x1_train,y_train_location_x)
predict_x_svr=clf_svr.predict(x1_test)


#knn
model_knr_dis_x = KNeighborsRegressor(weights='distance')
model_knr_dis_y = KNeighborsRegressor(weights='distance')
model_knr_dis_x.fit(x1_train, y_train_location_x)
model_knr_dis_y.fit(x1_train, y_train_location_y)
pre_knr_dis_x = model_knr_dis_x.predict(x1_test)
pre_knr_dis_y = model_knr_dis_y.predict(x1_test)

################################################
#linear

err = [0]*6600

for j in range(0,22):
    for i in range(300*j,300*(j+1)):
        re=math.sqrt(pow(pre_linear_x[i]- coordinate[j][0],2)+pow(pre_linear_y[i]-coordinate[j][1],2))
        err[i]=re
'''
for j in range(16,22):
    for i in range(4800+1500*(j-16),4800+1500*(j-15)):
        re=math.sqrt(pow(pre_linear_x[i] - coordinate[j][0],2)+pow(pre_linear_y[i]-coordinate[j][1],2))
        err[i]=re
'''

pdf = [0,0,0,0,0,0,0,0,0,0,0]    

for i in range (0,6600):
    if err[i]<0.5:
        pdf[1] += 1
    elif err[i]<1:
        pdf[2] += 1
    elif err[i]<1.5:
        pdf[3] += 1
    elif err[i]<2:
        pdf[4] += 1
    elif err[i]<2.5:
        pdf[5] += 1
    elif err[i]<3:
        pdf[6] += 1
    elif err[i]<3.5:
        pdf[7] += 1
    elif err[i]<4:
        pdf[8] += 1
    elif err[i]<4.5:
        pdf[9] += 1
    else:
        pdf[10] += 1

cdf_linear = pdf.copy()
for i in range(1,len(pdf)):
    cdf_linear[i] = pdf[i] + cdf_linear[i-1]

cdf_linear = np.array(cdf_linear)
cdf_linear = cdf_linear/6600

x_label = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
plt.grid(True)
plt.step(x_label, cdf_linear , color= "brown",label='Linearing')
plt.xlabel('error distance(m)')
plt.ylabel('CDF')
################################################
#svr

err = [0]*6600

for j in range(0,22):
    for i in range(300*j,300*(j+1)):
        re=math.sqrt(pow(predict_x_svr[i]- coordinate[j][0],2)+pow(predict_y_svr[i]-coordinate[j][1],2))
        err[i]=re
'''
for j in range(16,22):
    for i in range(4800+1500*(j-16),4800+1500*(j-15)):
        re=math.sqrt(pow(predict_x_svr[i] - coordinate[j][0],2)+pow(predict_y_svr[i]-coordinate[j][1],2))
        err[i]=re
'''

pdf = [0,0,0,0,0,0,0,0,0,0,0]    

for i in range (0,6600):
    if err[i]<0.5:
        pdf[1] += 1
    elif err[i]<1:
        pdf[2] += 1
    elif err[i]<1.5:
        pdf[3] += 1
    elif err[i]<2:
        pdf[4] += 1
    elif err[i]<2.5:
        pdf[5] += 1
    elif err[i]<3:
        pdf[6] += 1
    elif err[i]<3.5:
        pdf[7] += 1
    elif err[i]<4:
        pdf[8] += 1
    elif err[i]<4.5:
        pdf[9] += 1
    else:
        pdf[10] += 1

cdf_svr = pdf.copy()
for i in range(1,len(pdf)):
    cdf_svr[i] = pdf[i] + cdf_svr[i-1]

cdf_svr = np.array(cdf_svr)
cdf_svr = cdf_svr/6600

x_label = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
plt.grid(True)
plt.step(x_label, cdf_svr , color="green",label="SVR")
plt.xlabel('error distance(m)')
plt.ylabel('CDF')
################################################
#knn

err = [0]*6600

for j in range(0,22):
    for i in range(300*j,300*(j+1)):
        re=math.sqrt(pow(pre_knr_dis_x[i]- coordinate[j][0],2)+pow(pre_knr_dis_y[i]-coordinate[j][1],2))
        err[i]=re
'''
for j in range(16,22):
    for i in range(4800+1500*(j-16),4800+1500*(j-15)):
        re=math.sqrt(pow(pre_knr_dis_x[i] - coordinate[j][0],2)+pow(pre_knr_dis_y[i]-coordinate[j][1],2))
        err[i]=re
'''

pdf = [0,0,0,0,0,0,0,0,0,0,0]    

for i in range (0,6600):
    if err[i]<0.5:
        pdf[1] += 1
    elif err[i]<1:
        pdf[2] += 1
    elif err[i]<1.5:
        pdf[3] += 1
    elif err[i]<2:
        pdf[4] += 1
    elif err[i]<2.5:
        pdf[5] += 1
    elif err[i]<3:
        pdf[6] += 1
    elif err[i]<3.5:
        pdf[7] += 1
    elif err[i]<4:
        pdf[8] += 1
    elif err[i]<4.5:
        pdf[9] += 1
    else:
        pdf[10] += 1

cdf_knn = pdf.copy()
for i in range(1,len(pdf)):
    cdf_knn[i] = pdf[i] + cdf_knn[i-1]

cdf_knn = np.array(cdf_knn)
cdf_knn = cdf_knn/6600

x_label = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
plt.grid(True)
plt.plot(x_label, cdf_knn , color= "red",label='K-nearing neighbor',marker="o")
plt.plot(x_label, cdf_svr , color="green",label="SVR",marker="o")
plt.plot(x_label, cdf_linear , color= "brown",label='Linearing',marker="o")
plt.plot(x_label, cdf , color= "orange",label="SVC",marker="o")
plt.xlabel('Error distance(m)')
plt.ylabel('Cumulative error probability(%)')
plt.legend(title='Model:')
plt.show()
################################################

'''
plt.xlabel('error distance(m)')
plt.ylabel('CDF')
plt.legend(title='Model:')
plt.show()
'''
