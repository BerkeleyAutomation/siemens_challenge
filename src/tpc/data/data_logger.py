import numpy as np
import IPython
import os
import tpc.config.config_tpc as cfg

###Class created to store relevant information for learning at scale

class DataLogger():

	def __init__(self,file_path, do_evaluation=False):

		self.data = []
		self.rollout_info = {}
		self.file_path = file_path
		self.do_evaluation=do_evaluation


		if not os.path.exists(self.file_path):
			os.makedirs(self.file_path)


	def save_stat(self,stat_name,value,predicted_label,actual_label,other_data=None):
		"""
		return: the String name of the next new potential rollout
		(i.e. do not overwrite another rollout)
		"""
		i = 0

		file_path = self.file_path+'/'+stat_name

		if not os.path.exists(file_path):
			os.makedirs(file_path)

		stats_file_path = file_path+'/rollout_'+str(i)+'.npy'

		while os.path.isfile(stats_file_path):
			i += 1
			stats_file_path = file_path + '/rollout_'+str(i) +'.npy'

		print(stats_file_path)
		data = {'value':value, 'predicted_class':predicted_label, 'actual_class':actual_label, 'other_data':other_data}

		np.save(stats_file_path,data)


	def record_success(self,stat_name,class_name=None, other_data=None):

		while True:
			print("WAS "+ stat_name + "SUCCESUFL (Y/N)?")
			ans =raw_input('(y/n): ')

			actual_class=class_name

			if ans == 'y':
				self.save_stat(stat_name,1,class_name,class_name,other_data=other_data)
				return True
				# break;
			elif ans == 'n':
				if stat_name == 'object_recognition':
					print("Actual Class number is among: ")
					for k, v in cfg.net_labels.iteritems():
						print(k, v)
					class_num = raw_input('enter actual class number: ')
					actual_class = cfg.net_labels[int(class_num)]
				self.save_stat(stat_name,0,class_name, actual_class, other_data=other_data)
				return False
				# break;

		return
