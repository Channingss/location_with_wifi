
# coding: utf-8

# In[46]:
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
# In[50]:
def shijian_chuli(train): # 处理时间字段的函数，会新生成三列数据
    # hour_sx = {0:1,1:1,2:1,3:1,4:1,5:1,6:2,7:4,8:4,9:4,10:2,11:2,12:5,13:5,14:2,15:3,16:3,17:6,18:6,19:6,20:3,21:3,22:1,23:1}
    # week_sx = {0:1,1:1,2:1,3:1,4:1,5:2,6:2}
    train.loc[:,'time'] = pd.to_datetime(train.time_stamp)
    train['weekday'] = train.time.dt.weekday
    train['hour_time'] = train.time.dt.hour
    # train['hour_sx'] = train.hour_time.map(hour_sx)
    # train['week_sx'] = train.weekday.map(week_sx)
    return(train)

def socre_top(predict,y):
    num = 0.0
    for i in range(0,y.shape[0]):
        if y[i] in predict[i,:]:
            num += 1
    return   num/y.shape[0]


def process_table(path1='shopId_wifi_count', path2='bhvIndex_wifiInfo', path3='ccf_first_round_shop_info.csv', path4='ccf_first_round_user_shop_behavior.csv'):
    shopId_wifi_count = pd.read_pickle(path1)
    index_wifi = pd.read_pickle(path2)
    s_info = pd.read_csv(path3)
    u_bhv = pd.read_csv(path4)
    pri_train = pd.merge(u_bhv, s_info[['shop_id','mall_id','longitude','latitude']], on='shop_id') # 将shop表mall_id字段和behavior表通过shop_id字段连接
    return shopId_wifi_count, index_wifi, pri_train


def final_train_one_mall(mall_id, top=35, fill_nanWifi=-110, shopId_wifi_count=None, index_wifi=None, pri_train=None):
    """

    mall_id -- 传入str类型，表示最终要得到哪个mall的特征
    top -- 表示你要选没个商店的wifi出现频次做排序之后的top几
    path1 -- 'shopId_wifi_count'这个文件的路径,默认在同一个路径就不用传新路径
    path2 -- 'bhvIndex_wifiInfo'这个文件的路径,默认在同一个路径就不用传新路径
    path3 -- '训练数据-ccf_first_round_shop_info.csv'，默认这个函数在同一个路径就不用传新路径
    path4 -- '训练数据-ccf_first_round_user_shop_behavior.csv'，默认这个函数在同一个路径就不用传新路径
    """
    shop_wifi_top = shopId_wifi_count[shopId_wifi_count['sort_id'] <= top] # 取按shop_id分组的组内前30出现次数的wifi
    final_wifi_top_index = shop_wifi_top.groupby('wifi_name').count().index # 取出最终需要的wifi_name
    final_bhvIndex_wifiInfo_top = index_wifi[index_wifi['wifi_name'].isin(final_wifi_top_index)] # 只保留筛选出来的wifi
    pri_train_mall = pri_train[pri_train['mall_id']==mall_id]
    pri_train_mall = pri_train_mall.drop(['wifi_infos', 'mall_id', 'user_id'], axis=1) # m_4828商场的记录,去掉无用的column
    # 由于有些behavior记录的所有wifi都被筛选剔除了，所以在final_bhvIndex_wifiInfo可能有对应不上，会自动加index，并且数据为none
    final_wifi_mall = final_bhvIndex_wifiInfo_top.loc[pri_train_mall.index,:]
    ### 将wifi信息变成二元行向量 ####
    final_wifi_mall = final_wifi_mall.set_index('wifi_name',append=True) # 先将wifi_name变成索引
    final_wifi_mall = pd.Series(final_wifi_mall.values.reshape(len(final_wifi_mall)),index=final_wifi_mall.index) # 然后把DataFrame变成对应的Series
    final_wifi_mall = final_wifi_mall.unstack() # 然后把所有不同的wifi_name变成column，由于有些是nan,所以会多出一列np.nan
    if np.nan in final_wifi_mall.columns:
        final_wifi_mall = final_wifi_mall.drop(np.nan, axis=1) # 去掉空列
    final_wifi_mall[final_wifi_mall.isnull()] = fill_nanWifi
    ###############################
    final_train_1 = pri_train_mall.join(final_wifi_mall) #将最终wifi信息表加入其实训练特征
    shijian_chuli(final_train_1)
    final_train_1 = final_train_1.drop(['time'], axis=1) # 得到一天的星期几和一天24小时的两个特征
    # 将星期处理成二元向量，并加入final_train_1
    weekday_proc = pd.DataFrame(final_train_1['weekday'])
    weekday_proc['yes'] = 1
    weekday_proc = weekday_proc.set_index('weekday',append=True)
    weekday_proc=pd.Series(weekday_proc.values.reshape(len(weekday_proc)),index=weekday_proc.index)
    weekday_proc = weekday_proc.unstack()
    weekday_proc[weekday_proc.isnull()] = 0.0 # 将空值设为0
    final_train_1 = final_train_1.drop('weekday', axis=1).join(weekday_proc)
    final_train_1.rename(columns={0:'Sun', 1:'Mon', 2:'Tue', 3:'Wed', 4:'Thu', 5:'Fri', 6:'Sat'}, inplace=True)
    # 将小时处理成二元向量，并加入final_train_1
    hour_proc = pd.DataFrame(final_train_1['hour_time'])
    hour_proc['yes'] = 1
    hour_proc = hour_proc.set_index('hour_time',append=True)
    hour_proc=pd.Series(hour_proc.values.reshape(len(hour_proc)),index=hour_proc.index)
    hour_proc = hour_proc.unstack()
    hour_proc[hour_proc.isnull()] = 0.0 # 一条记录只会在一个时刻，所以其他空值设为0.0
    final_train_mall = final_train_1.drop('hour_time', axis=1).join(hour_proc)
    final_train_mall.iloc[:,3:] = final_train_mall.iloc[:,3:].astype(float) # 将除了shop_id的所有列换成float
    return final_train_mall
def main():
    path1 ='./shopId_wifi_count'
    path2 ='./bhvIndex_wifiInfo'
    path3 ='./ccf_first_round_shop_info.csv'
    path_train = './ccf_first_round_user_shop_behavior.csv'
    shop_info =pd.read_csv('./ccf_first_round_shop_info.csv')
    shopId_wifi_count_train, index_wifi_train, pri_train_train = process_table(path1 = path1, path2 = path2, path3 = path3,path4 = path_train)
    mall_list = list(set(list(shop_info.mall_id)))
    coverage_DF = pd.DataFrame(index=mall_list)
    # In[53]:
    for index,mall_id in  enumerate(mall_list):

        print('index: %s, mall_id: %s'%(index,mall_id))

        final_train = final_train_one_mall(mall_id=mall_id, top=10,
                                                   shopId_wifi_count=shopId_wifi_count_train, index_wifi=index_wifi_train, pri_train=pri_train_train)
        # In[54]:
        feature= [x for x in final_train.columns if x not in ['user_id','mall_id','shop_id','longitude_x','longitude_x','longitude_y','latitude_y','time_stamp','day']]
        shop_info_mall = shop_info[shop_info['mall_id']==mall_id]
        # In[60]:
        final_train.time_stamp = pd.to_datetime(final_train.time_stamp)
        final_train['day'] = final_train.time_stamp.dt.day
        test = final_train[final_train['day']>24]
        train = final_train[final_train['day']<=24]
        # In[44]:
        labelEncode = preprocessing.LabelEncoder() # sklearn的str向量映射方法，可以将str类型的labels变成对应的数字标签
        labelEncode.fit(shop_info_mall['shop_id']) # 先用一个向量得到固定的对应方式，也就是得到固定的编码器，有125个shop，对应125个数字映射
        y_train = labelEncode.transform(train['shop_id'])
        y_test = labelEncode.transform(test['shop_id'])# 分别对train和test处理
        num_class = len(labelEncode.classes_)
        # In[ ]:

        min_max_scaler = preprocessing.MinMaxScaler() # 这里采用区间缩放归一化
        min_max_scaler.fit(final_train[feature]) # 得到归一化的特征矩阵
        X_train = min_max_scaler.transform(train[feature])
        X_test = min_max_scaler.transform(test[feature])
        params = {
                            'objective': 'multi:softprob',
                            'eta': 0.1,
                            'max_depth': 9,
                            'eval_metric': 'merror',
                            'seed': 0,
                            'missing': -999,
                            'num_class':num_class,
                            'silent' : 1
                            }
        xgbtrain = xgb.DMatrix(X_train, y_train)
        xgbtest = xgb.DMatrix(X_test)
        xgbtrain4pred= xgb.DMatrix(X_train)
        watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]
        num_rounds=60
        model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)

        test_predict_proba = model.predict(xgbtest)
        train_predict_proba = model.predict(xgbtrain4pred)

        sample_test = np.argsort(test_predict_proba,axis=1)[:,-5:]
        sample_train = np.argsort(train_predict_proba,axis=1)[:,-5:]
        socre_test = socre_top(sample_test,y_test)
        coverage_DF.loc[mall_id,'coverage'] = socre_test
        accuracy=accuracy_score(np.argmax(test_predict_proba, axis=1), y_test)
        coverage_DF.loc[mall_id,'accuracy'] = accuracy

        set_of_candidates_test  = pd.DataFrame(labelEncode.inverse_transform(sample_test),index=test.index)
        set_of_candidates_train  = pd.DataFrame(labelEncode.inverse_transform(sample_train),index=train.index)
        set_of_candidates = pd.concat([set_of_candidates_test,set_of_candidates_train])
        set_of_candidates.to_pickle('./candidates_top10_iter300_xgb/set_of_candidates_'+mall_id+'_nolc.pkl')

        train_predict_proba_topk = np.zeros(sample_train.shape)
        for i in range(0,len(sample_train)):
            train_predict_proba_topk[i,] = train_predict_proba[i,sample_train[i]]

        test_predict_proba_topk = np.zeros(sample_test.shape)
        for i in range(0,len(sample_test)):
            test_predict_proba_topk[i,] = test_predict_proba[i,sample_test[i]]

        predict_proba_candidates_test  = pd.DataFrame(test_predict_proba_topk,index=test.index)
        predict_proba_candidates_train  = pd.DataFrame(train_predict_proba_topk,index=train.index)
        predict_proba_candidates = pd.concat([predict_proba_candidates_test,predict_proba_candidates_train])
        predict_proba_candidates.to_pickle('./candidates_top10_iter300_xgb/predict_proba_candidates_'+mall_id+'_nolc.pkl')
    coverage_DF.to_pickle('./coverage_xgb/coverage_nolc.pkl')

if __name__ == "__main__":
    main()



