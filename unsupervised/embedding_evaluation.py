import numpy as np
from sklearn.utils import shuffle
import torch
import joblib
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt
import os
import matplotlib
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

# from scipy import interp

import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import time

def get_emb_y(loader, encoder, device, dtype='numpy', is_rand_label=False):
	# train_emb, train_y
	x, y = encoder.get_embeddings(loader, device, is_rand_label)

	if dtype == 'numpy':
		return x,y
	elif dtype == 'torch':
		return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
	else:
		raise NotImplementedError

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def sensitivity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative
	Condition_negative = tp_sum + fn_sum + 1e-6
	sensitivity = tp_sum / Condition_negative

	return sensitivity



def precision(y_true, y_pred):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative
	Condition_negative = tn_sum + fn_sum + 1e-6
	precision = tn_sum / Condition_negative

	return precision


def specificity(y_pred, y_true):
	CM = confusion_matrix(y_true, y_pred) 

	tn_sum = CM[0, 0] # True Negative
	fp_sum = CM[0, 1] # False Positive

	tp_sum = CM[1, 1] # True Positive
	fn_sum = CM[1, 0] # False Negative

	Condition_negative = tn_sum + fp_sum + 1e-6
	Specificity = tn_sum / Condition_negative

	return Specificity

def binary_auc(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(np.unique(y_true)) < 2:
        return np.nan

    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)

class EmbeddingEvaluation():
	def __init__(self, base_classifier, evaluator, task_type, num_tasks,
                 device, save_dir=None,
                 params_dict=None, param_search=True, is_rand_label=False):
		self.save_dir = save_dir
		self.is_rand_label = is_rand_label
		self.base_classifier = base_classifier
		self.evaluator = evaluator
		self.eval_metric = evaluator.eval_metric
		self.task_type = task_type
		self.num_tasks = num_tasks
		self.device = device
		self.param_search = param_search
		self.params_dict = params_dict
		if self.eval_metric == 'rmse':
			self.gscv_scoring_name = 'neg_root_mean_squared_error'
		elif self.eval_metric == 'mae':
			self.gscv_scoring_name = 'neg_mean_absolute_error'
		elif self.eval_metric == 'rocauc':
			self.gscv_scoring_name = 'roc_auc'
		elif self.eval_metric == 'accuracy':
			self.gscv_scoring_name = 'accuracy'
		else:
			raise ValueError('Undefined grid search scoring for metric %s ' % self.eval_metric)

		self.classifier = None
	def scorer(self, y_true, y_raw):

		input_dict = {"y_true": y_true, "y_pred": y_raw}
		score = self.evaluator.eval(input_dict)[self.eval_metric]
		return score


	def ee_binary_classification(self, epoch,train_emb, train_y, val_emb, val_y):
		if self.param_search:
			params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
			self.classifier = make_pipeline(StandardScaler(),
			                                GridSearchCV(self.base_classifier, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
			                                )
		else:
			self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

		if np.isnan(train_emb).any():
			print("Has NaNs ... ignoring them")
			train_emb = np.nan_to_num(train_emb)
		
		if np.isnan(val_emb).any():
			print("Has NaNs ... ignoring them")
			val_emb = np.nan_to_num(val_emb)
		
			
		self.classifier.fit(train_emb, np.squeeze(train_y))


		if self.eval_metric == 'accuracy':
			train_raw = self.classifier.predict(train_emb)
			val_raw = self.classifier.predict(val_emb)
			
		else:
			train_raw = self.classifier.predict_proba(train_emb)[:, 1]
			val_raw = self.classifier.predict_proba(val_emb)[:, 1]
		
		if self.save_dir is not None:
			os.makedirs(self.save_dir, exist_ok=True)
			joblib.dump(
				self.classifier,
				os.path.join(self.save_dir, f'hypergraph_MLP_ABIDEI_model_{epoch}.pkl')
			)

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1)

	def before_ee_binary_classification(self,train_emb, train_y, val_emb, val_y):
		if self.param_search:
			params_dict = {'C': [0.001, 0.01,0.1,1,10,100,1000]}
			self.classifier = make_pipeline(StandardScaler(),
			                                GridSearchCV(self.base_classifier, params_dict, cv=5, scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
			                                )
		else:
			self.classifier = make_pipeline(StandardScaler(), self.base_classifier)

		if np.isnan(train_emb).any():
			print("Has NaNs ... ignoring them")
			train_emb = np.nan_to_num(train_emb)
		
		if np.isnan(val_emb).any():
			print("Has NaNs ... ignoring them")
			val_emb = np.nan_to_num(val_emb)
		
			
		self.classifier.fit(train_emb, np.squeeze(train_y))


		if self.eval_metric == 'accuracy':
			train_raw = self.classifier.predict(train_emb)
			val_raw = self.classifier.predict(val_emb)
			
		else:
			train_raw = self.classifier.predict_proba(train_emb)[:, 1]
			val_raw = self.classifier.predict_proba(val_emb)[:, 1]
		
		#保存无监督的权重部分	
		# joblib.dump(self.classifier, '/mnt/lvlian/zhangqq/zhangqq/ABIDEI/weights_hypergraph_MLP_20250114_k15/before_hypergraph_MLP_ABIDEI_model.pkl')
		if self.save_dir is not None:
			os.makedirs(self.save_dir, exist_ok=True)
			joblib.dump(
				self.classifier,
				os.path.join(self.save_dir, 'before_hypergraph_MLP_ABIDEI_model.pkl')
			)

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1)

	
	def ee_multioutput_binary_classification(self, train_emb, train_y, val_emb, val_y):

		params_dict = {
			'multioutputclassifier__estimator__C': [1e-1, 1e0, 1e1, 1e2]}
		self.classifier = make_pipeline(StandardScaler(), MultiOutputClassifier(
			self.base_classifier, n_jobs=-1))
		
		if np.isnan(train_y).any():
			print("Has NaNs ... ignoring them")
			train_y = np.nan_to_num(train_y)
		self.classifier.fit(train_emb, train_y)

		train_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(train_emb)])
		val_raw = np.transpose([y_pred[:, 1] for y_pred in self.classifier.predict_proba(val_emb)])
		

		return train_raw, val_raw


	def ee_regression(self, train_emb, train_y, val_emb, val_y):
		if self.param_search:
			params_dict = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]}
# 			params_dict = {'alpha': [500, 50, 5, 0.5, 0.05, 0.005, 0.0005]}
			self.classifier = GridSearchCV(self.base_classifier, params_dict, cv=5,
			                          scoring=self.gscv_scoring_name, n_jobs=16, verbose=0)
		else:
			self.classifier = self.base_classifier

		self.classifier.fit(train_emb, np.squeeze(train_y))

		train_raw = self.classifier.predict(train_emb)
		val_raw = self.classifier.predict(val_emb)
		

		return np.expand_dims(train_raw, axis=1), np.expand_dims(val_raw, axis=1)

		train_auc_score = binary_auc(train_y, train_raw)
		val_auc_score = binary_auc(val_y, val_raw)
																	
		return train_score, val_score, train_f1_score, val_f1_score, train_sen_score, val_sen_score, train_spe_score, val_spe_score, train_precison_score, val_precison_score, train_auc_score, val_auc_score, running_time
	
	def before_embedding_evaluation(self,encoder, train_loader, valid_loader, flag):
		encoder.eval()
		if flag:
			val_start = time.time()
		train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		#test_emb, test_y = get_emb_y(test_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		if flag:
			val_end = time.time()
			running_time = val_end-val_start
			print('validation time cost : %.5f sec' %running_time)

		if 'classification' in self.task_type:

			if self.num_tasks == 1:
				train_raw, val_raw = self.before_ee_binary_classification(train_emb, train_y, val_emb, val_y
				                                                        )
			elif self.num_tasks > 1:
				train_raw, val_raw, test_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y
				                                                                    )
			else:
				raise NotImplementedError
		else:
			if self.num_tasks == 1:
				train_raw, val_raw, test_raw = self.ee_regression(train_emb, train_y, val_emb, val_y)
			else:
				raise NotImplementedError
		

		train_score = self.scorer(train_y, train_raw)
		val_score = self.scorer(val_y, val_raw)
		# test_score = self.scorer(test_y, test_raw)

		train_sen_score = sensitivity(train_raw, train_y)
		val_sen_score = sensitivity(val_raw, val_y)
		# test_sen_score = sensitivity(test_raw, test_y)

		train_spe_score = specificity(train_raw, train_y)
		val_spe_score = specificity(val_raw, val_y)
		# test_spe_score = specificity(test_raw, test_y)

		train_f1_score = f1_score(train_y, train_raw)
		val_f1_score = f1_score(val_y, val_raw)


		train_precison_score = precision(train_y, train_raw)
		val_precison_score = precision(val_y, val_raw)


		train_auc_score = binary_auc(train_y, train_raw)
		val_auc_score = binary_auc(val_y, val_raw)
				

		return train_score, val_score, train_f1_score, val_f1_score, train_sen_score, val_sen_score, train_spe_score, val_spe_score, train_precison_score, val_precison_score, train_auc_score, val_auc_score, running_time

	def embedding_evaluation(self, epoch, encoder, train_loader, valid_loader, flag):
		encoder.eval()
		if flag:
			val_start = time.time()

		train_emb, train_y = get_emb_y(train_loader, encoder, self.device, is_rand_label=self.is_rand_label)
		val_emb, val_y = get_emb_y(valid_loader, encoder, self.device, is_rand_label=self.is_rand_label)

		if flag:
			val_end = time.time()
			running_time = val_end - val_start
			print('validation time cost : %.5f sec' % running_time)

		if 'classification' in self.task_type:
			if self.num_tasks == 1:
				train_raw, val_raw = self.ee_binary_classification(epoch, train_emb, train_y, val_emb, val_y)
			elif self.num_tasks > 1:
				train_raw, val_raw = self.ee_multioutput_binary_classification(train_emb, train_y, val_emb, val_y)
			else:
				raise NotImplementedError
		else:
			if self.num_tasks == 1:
				train_raw, val_raw = self.ee_regression(train_emb, train_y, val_emb, val_y)
			else:
				raise NotImplementedError

		train_score = self.scorer(train_y, train_raw)
		val_score = self.scorer(val_y, val_raw)

		train_sen_score = sensitivity(train_raw, train_y)
		val_sen_score = sensitivity(val_raw, val_y)

		train_spe_score = specificity(train_raw, train_y)
		val_spe_score = specificity(val_raw, val_y)

		train_f1_score = f1_score(train_y, train_raw)
		val_f1_score = f1_score(val_y, val_raw)

		train_precison_score = precision(train_y, train_raw)
		val_precison_score = precision(val_y, val_raw)

		train_auc_score = binary_auc(train_y, train_raw)
		val_auc_score = binary_auc(val_y, val_raw)

		return (
			train_score, val_score,
			train_f1_score, val_f1_score,
			train_sen_score, val_sen_score,
			train_spe_score, val_spe_score,
			train_precison_score, val_precison_score,
			train_auc_score, val_auc_score,
			running_time
		)

	def kf_embedding_evaluation(self, epoch,encoder, dataset, folds=5, batch_size=32, flag=False):
		kf_train = []
		kf_val = []
		kf_test = []
		kf_train_f1 = []
		kf_val_f1 = []
		kf_test_f1 = []
		kf_train_sen = []
		kf_val_sen = []
		kf_test_sen = []
		kf_train_spe = []
		kf_val_spe = []
		kf_test_spe = []
		kf_train_pre = []
		kf_val_pre = []
		kf_train_auc = []
		kf_val_auc = []
		running_times = []
		
		kf = KFold(n_splits=folds, shuffle=True, random_state=None)
		#for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
		for k_id, (train_index, val_index) in enumerate(kf.split(dataset)):

			train_dataset = [dataset[int(i)] for i in list(train_index)]
			val_dataset = [dataset[int(i)] for i in list(val_index)]

			train_loader = DataLoader(train_dataset, batch_size=batch_size)
			valid_loader = DataLoader(val_dataset, batch_size=batch_size)

			train_score, val_score, train_f1, val_f1, train_sen, val_sen, train_spe, val_spe, train_precison_score, val_precison_score, train_auc_score, val_auc_score, running_time = self.embedding_evaluation(epoch, encoder, train_loader, valid_loader, flag=1)
			running_times.append(running_time)
	
			kf_train_f1.append(train_f1)
			kf_val_f1.append(val_f1)
			
	
			kf_train_spe.append(train_spe)
			kf_val_spe.append(val_spe)
		

			kf_train.append(train_score)
			kf_val.append(val_score)
			

			kf_train_sen.append(train_sen)
			kf_val_sen.append(val_sen)
			
			kf_train_pre.append(train_precison_score)
			kf_val_pre.append(val_precison_score)
			
			kf_train_auc.append(train_auc_score)
			kf_val_auc.append(val_auc_score)

		mean_time = np.array(running_times).mean()
		print("mean validation time %.5f:\n"% mean_time)

		kf_train_ms = [
			np.array(kf_train).mean(), np.array(kf_train).std(),
			np.array(kf_train_f1).mean(), np.array(kf_train_f1).std(),
			np.array(kf_train_sen).mean(), np.array(kf_train_sen).std(),
			np.array(kf_train_spe).mean(), np.array(kf_train_spe).std(),
			np.array(kf_train_pre).mean(), np.array(kf_train_pre).std(),
			np.array(kf_train_auc).mean(), np.array(kf_train_auc).std()
		]

		kf_val_ms = [
			np.array(kf_val).mean(), np.array(kf_val).std(),
			np.array(kf_val_f1).mean(), np.array(kf_val_f1).std(),
			np.array(kf_val_sen).mean(), np.array(kf_val_sen).std(),
			np.array(kf_val_spe).mean(), np.array(kf_val_spe).std(),
			np.array(kf_val_pre).mean(), np.array(kf_val_pre).std(),
			np.array(kf_val_auc).mean(), np.array(kf_val_auc).std()
		]
				

		return kf_train_ms, kf_val_ms
	


	def kf_before_embedding_evaluation(self, encoder, dataset, folds=5, batch_size=32, flag=False):
		kf_train = []
		kf_val = []
		kf_test = []
		kf_train_f1 = []
		kf_val_f1 = []
		kf_test_f1 = []
		kf_train_sen = []
		kf_val_sen = []
		kf_test_sen = []
		kf_train_spe = []
		kf_val_spe = []
		kf_test_spe = []

		kf_train_precision=[]
		kf_val_precision=[]
		kf_train_auc=[]
		kf_val_auc=[]

		running_times = []
		
		kf = KFold(n_splits=folds, shuffle=True, random_state=None)
		#for k_id, (train_val_index, test_index) in enumerate(kf.split(dataset)):
		for k_id, (train_index, val_index) in enumerate(kf.split(dataset)):

		
			#test_id.append(test_index)

			# test_dataset = [dataset[int(i)] for i in list(test_index)]
			# train_index, val_index = train_test_split(train_val_index, test_size=0.2, random_state=None)

			train_dataset = [dataset[int(i)] for i in list(train_index)]
			val_dataset = [dataset[int(i)] for i in list(val_index)]

			train_loader = DataLoader(train_dataset, batch_size=batch_size)
			valid_loader = DataLoader(val_dataset, batch_size=batch_size)
			#test_loader = DataLoader(test_dataset, batch_size=batch_size)

			# embedding_evaluation -> get_emb_y -> encoder.get_embeddings -> forward
			# train_score, val_score, test_score, train_f1, val_f1, test_f1, train_sen, val_sen, test_sen, train_spe, val_spe, test_spe, fpr, tpr, running_time= self.embedding_evaluation(encoder, 
			train_score, val_score, train_f1, val_f1, train_sen, val_sen, train_spe, val_spe, train_precison_score, val_precison_score, train_auc_score, val_auc_score, running_time = self.before_embedding_evaluation(encoder, train_loader, valid_loader, flag=1)
			running_times.append(running_time)
	
			kf_train_f1.append(train_f1)
			kf_val_f1.append(val_f1)
			
	
			kf_train_spe.append(train_spe)
			kf_val_spe.append(val_spe)
		

			kf_train.append(train_score)
			kf_val.append(val_score)
			

			kf_train_sen.append(train_sen)
			kf_val_sen.append(val_sen)

			kf_train_precision.append(train_precison_score)
			kf_val_precision.append(val_precison_score)

			kf_train_auc.append(train_auc_score)
			kf_val_auc.append(val_auc_score)
						

		mean_time = np.array(running_times).mean()
		print("mean validation time %.5f:\n"% mean_time)

		kf_train_ms = [
			np.array(kf_train).mean(), np.array(kf_train).std(),
			np.array(kf_train_f1).mean(), np.array(kf_train_f1).std(),
			np.array(kf_train_sen).mean(), np.array(kf_train_sen).std(),
			np.array(kf_train_spe).mean(), np.array(kf_train_spe).std(),
			np.array(kf_train_precision).mean(), np.array(kf_train_precision).std(),
			np.array(kf_train_auc).mean(), np.array(kf_train_auc).std()
		]

		kf_val_ms = [
			np.array(kf_val).mean(), np.array(kf_val).std(),
			np.array(kf_val_f1).mean(), np.array(kf_val_f1).std(),
			np.array(kf_val_sen).mean(), np.array(kf_val_sen).std(),
			np.array(kf_val_spe).mean(), np.array(kf_val_spe).std(),
			np.array(kf_val_precision).mean(), np.array(kf_val_precision).std(),
			np.array(kf_val_auc).mean(), np.array(kf_val_auc).std()
		]

		return kf_train_ms, kf_val_ms


	######################### 新增 ####################
	from sklearn.model_selection import train_test_split
	from torch_geometric.data import DataLoader  # 你文件里已经导了 DataLoader 就不用重复

	# ====== 放到 class EmbeddingEvaluation 里面（同级缩进）======

	def split_embedding_evaluation(self, epoch, encoder, dataset, train_ratio=0.8, batch_size=32, flag=False):
		"""
		单次随机划分（train/val），返回 (train_ms, val_ms)
		结构与 kf_embedding_evaluation 一致：10 项 list
		"""
		idx = np.arange(len(dataset))
		train_idx, val_idx = train_test_split(
			idx,
			train_size=train_ratio,
			shuffle=True,
			random_state=0  # 固定住，每次划分一致；想每次随机就改成 None
		)

		train_dataset = [dataset[int(i)] for i in train_idx]
		val_dataset   = [dataset[int(i)] for i in val_idx]

		train_loader = DataLoader(train_dataset, batch_size=batch_size)
		valid_loader = DataLoader(val_dataset, batch_size=batch_size)

		# embedding_evaluation 返回 11 个值
		(train_score, val_score,
		train_f1, val_f1,
		train_sen, val_sen,
		train_spe, val_spe,
		train_pre, val_pre,
		train_auc, val_auc,
		running_time) = self.embedding_evaluation(epoch, encoder, train_loader, valid_loader, flag=1)
		# 单次 split 没有 std，先填 0.0，保证 train.py 兼容
		train_ms = [train_score, 0.0,
					train_f1,   0.0,
					train_sen,  0.0,
					train_spe,  0.0,
					train_pre,  0.0,
					train_auc,  0.0]
		val_ms   = [val_score, 0.0,
					val_f1,   0.0,
					val_sen,  0.0,
					val_spe,  0.0,
					val_pre,  0.0,
					val_auc,  0.0]
		return train_ms, val_ms


	def split_before_embedding_evaluation(self, encoder, dataset, train_ratio=0.8, batch_size=32, flag=False):
		"""
		训练前评估版本：单次随机划分（train/val），返回 (train_ms, val_ms)
		结构与 kf_before_embedding_evaluation 一致：10 项 list
		"""
		idx = np.arange(len(dataset))
		train_idx, val_idx = train_test_split(
			idx,
			train_size=train_ratio,
			shuffle=True,
			random_state=0
		)

		train_dataset = [dataset[int(i)] for i in train_idx]
		val_dataset   = [dataset[int(i)] for i in val_idx]

		train_loader = DataLoader(train_dataset, batch_size=batch_size)
		valid_loader = DataLoader(val_dataset, batch_size=batch_size)

		# before_embedding_evaluation 返回 11 个值
		(train_score, val_score,
		train_f1, val_f1,
		train_sen, val_sen,
		train_spe, val_spe,
		train_pre, val_pre,
		train_auc, val_auc,
		running_time) = self.before_embedding_evaluation(encoder, train_loader, valid_loader, flag=1)

		train_ms = [train_score, 0.0,
					train_f1,   0.0,
					train_sen,  0.0,
					train_spe,  0.0,
					train_pre,  0.0,
					train_auc,  0.0]
		val_ms   = [val_score, 0.0,
					val_f1,   0.0,
					val_sen,  0.0,
					val_spe,  0.0,
					val_pre,  0.0,
					val_auc,  0.0]

		return train_ms, val_ms