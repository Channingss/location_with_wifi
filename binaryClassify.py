
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
	"""
	Calculate the great circle distance between two points
	on the earth (specified in decimal degrees)
	"""
	# 将十进制度数转化为弧度
	lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

	# haversine公式
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
	c = 2 * asin(sqrt(a)) 
	r = 6371 # 地球平均半径，单位为公里
	return c * r * 1000

haversinevectorize = np.vectorize(haversine)
def shijian_chuli(train):
	hour_sx = {0:1,1:1,2:1,3:1,4:1,5:1,6:2,7:4,8:4,9:4,10:2,11:2,12:5,13:5,14:2,15:3,16:3,17:6,18:6,19:6,20:3,21:3,22:1,23:1}
	week_sx = {0:1,1:1,2:1,3:1,4:1,5:2,6:2}
	train.loc[:,'time'] = pd.to_datetime(train.time_stamp)
	train['weekday'] = train.time.dt.weekday
	train['hour_time'] = train.time.dt.hour
	train['hour_sx'] = train.hour_time.map(hour_sx)
	train['week_sx'] = train.weekday.map(week_sx)
	return(train)


def process_table(top=25, path1='shopId_wifi_count', path2='all_data', path3='all_index_wifi'):
    shopId_wifi_count = pd.read_pickle(path1)
    # 把train和test表合并做处理，不会出现字段不统一的情况,index>=1138015的全是test集（1621946）
    all_data = pd.read_pickle(path2)

    # 把train，test的合表取出一张表，index对应合表的index，每一条是对应的是wifi_name,strength
    all_index_wifi = pd.read_pickle(path3)

    shop_wifi_topk = shopId_wifi_count[shopId_wifi_count['sort_id'] <= 25] # 通过train的shop，这里的数字可选top k的wifi！！！！！
    shop_wifi_topk_index = shop_wifi_topk.groupby('wifi_name').count().index # 通过train的shop，取出最终需要的wifi的name

    final_all_wifiInfo_topk = all_index_wifi[all_index_wifi['wifi_name'].isin(shop_wifi_topk_index)] # 合表中只保留筛选得到的wifi
    return all_data, final_all_wifiInfo_topk

def final_test_data(mall_id, fill_nanWifi=-110, all_data=None, final_all_wifiInfo_topk=None):

    all_data_mall_id = all_data[all_data['mall_id']==mall_id]
    all_data_mall_id = all_data_mall_id.drop(['wifi_infos'], axis=1) # m_4828商场的记录,去掉无用的column

    # 由于有些behavior记录的所有wifi都被筛选剔除了，所以在final_bhvIndex_wifiInfo可能有对应不上，会自动加index，并且数据为none
    final_all_wifiInfo_mall_id = final_all_wifiInfo_topk.loc[all_data_mall_id.index,:] # 通过index取出当前mall对应的wifi表
    # 这里把最终index对应的wifi表，变成index对应wifi的二元向量
    final_all_wifiInfo_mall_id = final_all_wifiInfo_mall_id.set_index('wifi_name',append=True) # 先将wifi_name变成索引
    final_all_wifiInfo_mall_id = pd.Series(final_all_wifiInfo_mall_id.values.reshape(len(final_all_wifiInfo_mall_id)),index=final_all_wifiInfo_mall_id.index) # 然后把DataFrame变成对应的Series
    final_all_wifiInfo_mall_id = final_all_wifiInfo_mall_id.unstack() # 然后把所有不同的wifi_name变成column，由于有些是nan,所以会多出一列np.nan
    final_all_wifiInfo_mall_id = final_all_wifiInfo_mall_id.drop(np.nan, axis=1) # 去掉空列
    final_all_wifiInfo_mall_id[final_all_wifiInfo_mall_id.isnull()] = fill_nanWifi # 填补wifi里面的强度空值

    final_all_data_1 = all_data_mall_id.join(final_all_wifiInfo_mall_id) #将最终wifi信息表加入其他训练特征

    shijian_chuli(final_all_data_1)
    final_all_data_1 = final_all_data_1.drop(['time_stamp', 'time'], axis=1) # 得到一天的星期几和一天24小时的两个特征

    # 取出处理完成的train，test表
    final_train_data_mall = final_all_data_1[final_all_data_1['row_id'].isnull()]
    final_test_data_mall = final_all_data_1[final_all_data_1['shop_id'].isnull()]

    final_train_data_mall = final_train_data_mall.drop('row_id', axis=1)
    final_test_data_mall = final_test_data_mall.drop('shop_id', axis=1)
    return final_train_data_mall, final_test_data_mall


def main():

	path1 ='./shopId_wifi_count'

	path2 ='./all_data'

	path3 ='./all_index_wifi'

	shop_info = pd.read_csv('./ccf_first_round_shop_info.csv')

	behavior =  pd.read_csv('./ccf_first_round_user_shop_behavior.csv')

	join = pd.merge(behavior,shop_info,on='shop_id')

	times_buy_in_mall = join.groupby(by='mall_id').count()['price']

	times_buy_in_mall = times_buy_in_mall.to_frame().rename(columns={'price':'times'})

	drop_columns = ['wifi_infos', 'category_id', 'longitude_y','latitude_y','price','mall_id','level_1','shop_id','user_id']

	mall_list = list(set(list(shop_info.mall_id)))

	shop_visit_times = pd.read_pickle('./shop_visit_times')

	evaluation =  pd.read_csv('./evaluation_public.csv').set_index('row_id')

	candidate_all_false = pd.read_pickle('./candidates_all_xgb/candidate_all_false_time.pkl')

	iteration = 1

	accuracy_DF = pd.DataFrame(index=mall_list)

	submit = pd.DataFrame(columns={'row_id','shop_id'})

	all_data, final_all_wifiInfo_topk = process_table(top=15, path1=path1, path2=path2, path3=path3)

	for mall_id in mall_list:

		predict_proba_candidates = pd.read_pickle('./candidates_top10_iter300_xgb/predict_proba_candidates_'+mall_id+'_nolc.pkl')

		predict_proba_candidates_stack = predict_proba_candidates.stack(level=0).to_frame()

		predict_proba_candidates_stack_resetindex = predict_proba_candidates_stack.reset_index(level=1)

		final_train_m_wifilist, final_test_m_wifilist = final_test_data(mall_id,all_data = all_data,final_all_wifiInfo_topk = final_all_wifiInfo_topk)

		final_train_m_wifilist = final_train_m_wifilist.drop(['latitude','longitude','mall_id','shop_id','user_id'],axis=1)

		predict_proba_candidates_stack_resetindex_withWfList = predict_proba_candidates_stack_resetindex.join(final_train_m_wifilist,how='inner')

		predict_proba_candidates_std = predict_proba_candidates.stack().groupby(level=0).std().to_frame().rename(columns = {0:'std'})

		predict_prob = predict_proba_candidates_stack_resetindex_withWfList.join(predict_proba_candidates_std)

		predict_prob = predict_prob.set_index('level_1',append=True)

		candidates = pd.read_pickle('./candidates_top10_iter300_xgb/set_of_candidates_'+mall_id+'_nolc.pkl')

		candidates_stack = candidates.stack(level=0).reset_index(level=1)

		candidates_stack = candidates_stack.set_index('level_1',append=True)

		sample_with_proba = predict_prob.join(candidates_stack,lsuffix='predict_proba',rsuffix='shop_id')

		candidates_join = join.join(sample_with_proba.reset_index(level=1),how='inner').rename(columns={'0shop_id':'shop_id_candidate',})

		candidates_join_negative = candidates_join[candidates_join['shop_id_candidate']!=candidates_join['shop_id']]

		candidates_join_negative_drop = candidates_join_negative.drop(drop_columns,axis=1)

		candidates_join_negative_2 = pd.merge(candidates_join_negative_drop,shop_info,left_on='shop_id_candidate',right_on='shop_id')

		candidates_join_negative_2['label'] = 0

		candidates_join_positive = candidates_join[candidates_join['shop_id_candidate']==candidates_join['shop_id']]

		candidates_join_positive_drop = candidates_join_positive.drop(drop_columns,axis=1)

		candidates_join_positive_2 = pd.merge(candidates_join_positive_drop,shop_info,left_on='shop_id_candidate',right_on='shop_id')

		candidates_join_positive_2['label'] = 1

		candidates_4train = pd.concat([candidates_join_positive_2,candidates_join_negative_2])

		candidates_4train = shijian_chuli(candidates_4train)

		candidates_4train['category_id'] = candidates_4train['category_id'].map(lambda x :x.split('_')[1])

		candidates_4train['category_id'] = candidates_4train['category_id'].astype(int)

		candidates_4train = pd.concat([candidates_4train,candidates_4train])

		# candidates_4train = candidates_4train.merge(shop_visit_times,on='shop_id')

		# candidates_4train = candidates_4train.merge(candidate_all_false,on='shop_id')


#*****计算关门时间

		maxTime = candidates_4train.groupby(by='shop_id')['hour_time'].max().to_frame()

		minTime = candidates_4train.groupby(by='shop_id')['hour_time'].min().to_frame()

		closure= maxTime.join(minTime,lsuffix='_max',rsuffix='_min')

		closure['closeTime'] = closure.apply(lambda line : line['hour_time_min'] if line['hour_time_min']<7 else line['hour_time_max'],axis=1)

		closure = closure.reset_index()

		candidates_4train = candidates_4train.merge(closure[['shop_id','closeTime']], on = 'shop_id')

#*****删除距离过远的异常点

		candidates_4train['distance_user_shop'] = haversinevectorize(candidates_4train.longitude_x.values,candidates_4train.latitude_x.values,candidates_4train.longitude.values,candidates_4train.latitude.values)

		candidates_4train_drop_faraway = candidates_4train[candidates_4train['distance_user_shop']<10000]

#计算shop正负样本的比例
		# candidates_p = candidates_4train_drop_faraway[candidates_4train_drop_faraway['label']==1]

		# candidates_n = candidates_4train_drop_faraway[candidates_4train_drop_faraway['label']==0]

		# shop_n_times = candidates_n.groupby(by = 'shop_id').count()['time'].to_frame()

		# shop_p_times = candidates_p.groupby(by = 'shop_id').count()['time'].to_frame()

		# shop_times_p_n = shop_n_times.join(shop_p_times,lsuffix='_n',rsuffix='_p',how='outer')

		# shop_times_p_n = shop_times_p_n.fillna(0)

		# # shop_times_p_n['tooStrong'] = shop_times_p_n['time_n'] / shop_times_p_n['time_p']
		# # shop_times_p_n['tooStrong'] = shop_times_p_n.apply(lambda a : a['time_n'] if np.isinf(a['tooStrong']) else a['tooStrong'],axis=1)
		# # shop_times_p_n['tooStrong'] = shop_times_p_n.apply(lambda a : -a['time_p'] if a['time_n'] == 0.0 else a['tooStrong'],axis=1)

		# shop_times_p_n = shop_times_p_n.reset_index()

		# candidates_4train_drop_faraway = candidates_4train_drop_faraway.merge(shop_times_p_n,on='shop_id')

#******TEST候选集处理

		predict_proba_candidates_test = pd.read_pickle('./candidates_top10_iter300_xgb_test/predict_proba_candidates_'+mall_id+'.pkl')

		candidates_test = pd.read_pickle('./candidates_top10_iter300_xgb_test/set_of_candidates_'+mall_id+'.pkl')

		predict_proba_candidates_test['row_id'] = predict_proba_candidates_test['row_id'].astype(int)

		candidates_test['row_id'] = candidates_test['row_id'].astype(int)

		candidates_reindex = candidates_test.set_index('row_id')

		candidates__test_stack = candidates_reindex.stack(level=0).to_frame()

		# predict_proba_candidates_test_stack_resetindex = predict_proba_candidates_test_stack.reset_index(level=1)

		final_test_m_wifilist = final_test_m_wifilist.drop(['latitude','longitude','mall_id','user_id'],axis=1)

		final_test_m_wifilist = final_test_m_wifilist.set_index('row_id')

		predict_proba_candidates_test_reindex = predict_proba_candidates_test.set_index('row_id')

		predict_proba_candidates_test_stack = predict_proba_candidates_test_reindex.stack(level=0).to_frame()

		# predict_proba_candidates_test_stack = predict_proba_candidates_test_stack.reset_index(level='row_id')

		# predict_proba_candidates_test_stack = predict_proba_candidates_test_stack.set_index('row_id',drop=False)

		predict_proba_candidates_test_stack_resetindex_withWfList = predict_proba_candidates_test_stack.join(final_test_m_wifilist)

		predict_proba_candidates_test_std = predict_proba_candidates_test_stack.stack().groupby(level=0).std().to_frame().rename(columns = {0:'std'})

		predict_prob = predict_proba_candidates_test_stack_resetindex_withWfList.join(predict_proba_candidates_test_std)

		print('predict_prob.shape:',predict_prob.shape)

		sample_with_proba_test = predict_prob.join(candidates__test_stack,lsuffix='predict_proba',rsuffix='shop_id')

		print('sample_with_proba_test.shape:',sample_with_proba_test.shape)

		candidates_join_test = evaluation.join(sample_with_proba_test,how='inner').rename(columns={'0shop_id':'shop_id_candidate'})

		print('candidates_join_test.shape:',candidates_join_test.shape)

		candidates_join_test = candidates_join_test.reset_index()

		candidates_4test= pd.merge(candidates_join_test,shop_info,left_on='shop_id_candidate',right_on='shop_id')

		candidates_4test = shijian_chuli(candidates_4test)

		candidates_4test['category_id'] = candidates_4test['category_id'].map(lambda x :x.split('_')[1])

		candidates_4test['category_id'] = candidates_4test['category_id'].astype(int)

		# candidates_4test = candidates_4test.merge(shop_visit_times,on='shop_id')

		# candidates_4test = candidates_4test.merge(candidate_all_false,on='shop_id')

		maxTime_test = candidates_4test.groupby(by='shop_id')['hour_time'].max().to_frame()

		minTime_test = candidates_4test.groupby(by='shop_id')['hour_time'].min().to_frame()

		closure_test= maxTime_test.join(minTime_test,lsuffix='_max',rsuffix='_min')

		closure_test['closeTime'] = closure_test.apply(lambda line : line['hour_time_min'] if line['hour_time_min']<7 else line['hour_time_max'],axis=1)

		closure_test = closure_test.reset_index()

		candidates_4test = candidates_4test.merge(closure_test[['shop_id','closeTime']], on = 'shop_id')

		candidates_4test = candidates_4test.rename(columns = {'latitude_y':'latitude','longitude_y':'longitude'})

		candidates_4test['distance_user_shop'] = haversinevectorize(candidates_4test.longitude_x.values,candidates_4test.latitude_x.values,candidates_4test.longitude.values,candidates_4test.latitude.values)

		# candidates_4test = candidates_4test.merge(shop_times_p_n,on='shop_id')

		print(candidates_4test.columns)

		print('candidates_4test.shape:',candidates_4test.shape)

#******TEST候选集处理


		featrue = ['distance_user_shop','category_id','price','weekday','hour_time','0predict_proba','hour_sx','week_sx','times','time_p','time_n','ratio_false_classify','closeTime']

		feature_wifilist = [x for x in candidates_4train.columns if x not in ['shop_id_candidate','time','shop_id','time_stamp','mall_id']]

		feature_wifilist_noProba_test = [x for x in candidates_4test.columns if x not in ['shop_id_candidate','time','shop_id','time_stamp','mall_id','level_1','row_id','user_id','mall_id_x','mall_id_y','wifi_infos']]

		feature_wifilist_noProba_train = [x for x in candidates_4train.columns if x not in ['shop_id_candidate','time','shop_id','time_stamp','mall_id','label']]

		print(candidates_4test[feature_wifilist_noProba_test].columns)


		print(candidates_4train[feature_wifilist_noProba_train].columns)

		featrue_noProba= ['distance_user_shop','category_id','price','weekday','hour_time','hour_sx','week_sx','times','time_p','time_n','ratio_false_classify','closeTime']

		X_train = candidates_4train_drop_faraway[feature_wifilist_noProba_train]

		X_test = candidates_4test[feature_wifilist_noProba_test]

		y_train = candidates_4train_drop_faraway['label']

		params= {
					'max_depth': 10,
					'eta': 1,
					'silent': 1,
					'objective': 'binary:logistic',
					'eval_metric':'auc'}

		xgbtrain = xgb.DMatrix(X_train, label = y_train)

		xgbtest= xgb.DMatrix(X_test)

#	model_lstd
		watchlist__lstd = [(xgbtrain, 'eval'), (xgbtrain, 'train')]

		num_rounds_lstd =100

		model = xgb.train(params, xgbtrain, num_rounds_lstd, watchlist__lstd, early_stopping_rounds=15)

		preb_test = model.predict(xgbtest)

		candidates_4test['score'] = preb_test

		print('iteration: ',iteration)

		iteration += 1

		candidates_join_shop_info_submit = candidates_4test.iloc[candidates_4test.groupby(['row_id']).apply(lambda x: x['score'].idxmax())]

		submit = pd.concat([submit,candidates_join_shop_info_submit[['row_id','shop_id_candidate']].rename(columns={'shop_id_candidate':'shop_id'})])

	submit.to_csv('sub_1800_1117_bin_withwflist.csv',index=False)





if __name__ =='__main__':
	main()


