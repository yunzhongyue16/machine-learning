import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split   #切分数据
from sklearn.metrics import mean_squared_error         #评价指标
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# import warnings
# warnings.filterwarnings("ignore")
train_data_file = "zhengqi_train.txt"
test_data_file = "zhengqi_test.txt"
train_data = pd.read_csv(train_data_file, sep='\t', encoding='utf-8')
test_data = pd.read_csv(test_data_file, sep='\t', encoding='utf-8')


#归一化处理
features_columns=[col for col in train_data.columns if col not in ['target']]
min_max_scaler=preprocessing.MinMaxScaler()
min_max_scaler=min_max_scaler.fit(train_data[features_columns])
train_data_scaler=min_max_scaler.transform(train_data[features_columns])
test_data_scaler=min_max_scaler.transform(test_data[features_columns])
train_data_scaler=pd.DataFrame(train_data_scaler)
train_data_scaler.columns=features_columns
test_data_scaler=pd.DataFrame(test_data_scaler)
test_data_scaler.columns=features_columns
train_data_scaler['target']=train_data['target']

#PCA降维
pca=PCA(n_components=16)
new_train_pca_16=pca.fit_transform(train_data_scaler.iloc[:,0:-1])
new_test_pca_16=pca.transform(test_data_scaler)
new_train_pca_16=pd.DataFrame(new_train_pca_16)
new_test_pca_16=pd.DataFrame(new_test_pca_16)
new_train_pca_16['target']=train_data_scaler['target']


new_train_pca_16=new_train_pca_16.fillna(0)    #采用PCA保留16维特征的数据
train=new_train_pca_16[new_test_pca_16.columns]

target=new_train_pca_16['target']
#划分训练集为80%，测试集为20%
train_data,test_data,train_target,test_target=train_test_split(train,target,test_size=0.2,random_state=0)

clf=lgb.LGBMRegressor(learning_rate=0.01,max_depth=-1,n_estimators=5000,boosting_type='gbdt',random_state=2019,objective='regression',)
#训练模型
clf.fit(X=train_data,y=train_target,eval_metric='MSE',verbose=50)
score=mean_squared_error(test_target,clf.predict(test_data))
print("lightgbm:	",score)



