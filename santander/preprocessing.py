#-*- coding=utf-8 -*-
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import scale, MinMaxScaler
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
#分测试训练集
def train_test_split(df, target, percentage=0.3):
    X = df.drop([target],axis = 1)
    y = df[target]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=percentage)
    for train_idx, test_idx in sss.split(X,y):
        df_test = df.loc[test_idx]
        df_train = df.loc[train_idx]
    df_train.index = range(len(df_train))
    df_test.index = range(len(df_test))
    return df_train, df_test

def normalization(df, target, method = 'minmax'):
	if method == 'minmax':
		scaler = MinMaxScaler()
    	X = df.drop([target],axis=1)
    	y = df[target]
    	X_normalized = scaler.fit_transform(X)
    	columns = df.columns
    	df_out = pd.DataFrame(X_normalized, columns = columns)
    	df_out[target] = y
    	return X_normalized, y df_out
    elif method = 'z-score':
    	sclaer = scale()
    	X = df.drop([target],axis=1)
    	y = df[target]
    	X_normalized = scaler.fit_transform(X)
    	columns = df.columns
    	df_out = pd.DataFrame(X_normalized, columns = columns)
    	df_out[target] = y
    	return X_normalized, y df_out

def SMOTE_methods(df_train,target,method):
    '''The output data has been normalized by MinMaxScaler'''
    scaler = MinMaxScaler()
    X = df_train.drop([target],axis=1)
    y = df_train[target]
    X_normalized = scaler.fit_transform(X)
    if method == 'regular':
        X_res, y_res = SMOTE(kind='regular').fit_sample(X_normalized, y)
    elif method == 'borderline1':
        X_res, y_res = SMOTE(kind='borderline1').fit_sample(X_normalized, y)
    elif method == 'borderline2':
        X_res, y_res = SMOTE(kind='borderline2').fit_sample(X_normalized, y)
    elif method == 'svm':
        X_res, y_res = SMOET(kind='svm').fit_sample(X_normalized, y)
    elif method == 'Tomek':
        sm = SMOTETomek()
        X_res, y_res = sm().fit_sample(X_normalized, y)
    elif method == 'ENN':
        sm = SMOTEENN()
        X_res, y_res = sm().fit_sample(X_normalized, y)
    else:
        raise ValueError('输入方法有误')
    df_final = pd.DataFrame(X_res, columns = X.columns)
    df_final['target'] = y_res
    return df_final

#为方便之后测试，请务必使用函数形式

def under_sample(df_train):
    '''
    input: training dateframe including target
    output: under sampled dataframe
    '''
    t0 = df_train[df_train['target']==0]
    t1 = df_train[df_train['target']==1]


    df = t0.sample(frac=len(t1)/len(t0)).append(t1)
    
    return df

def over_sample(df_train):
    '''
    input: training dateframe including target
    output: under sampled dataframe
    '''
    t0 = df_train[df_train['target']==0]
    t1 = df_train[df_train['target']==1]


    df = t1.sample(frac=len(t0)/len(t1),replace=True).append(t0)
    
    return df