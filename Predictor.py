#!/usr/bin/env python
# coding: utf-8



import numpy as np
import json
import thulac
import initializer
from initializer import get_time,get_name,init
from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing import sequence



class Predictor(object):
	def __init__(self):
		self.tokenizer=joblib.load('./predictor/model/tokenizer.h5') 
		self.law = load_model('./predictor/model/law_textcnn.h5')
		self.accu = load_model('predictor/model/accu_textcnn.h5')
		self.time = load_model('predictor/model/time_textcnn.h5')
		self.batch_size = 1
		self.cut = thulac.thulac(seg_only = True)
	def predict_law(self, vec):
		y = self.law.predict(vec)
		return y.argmax()
	
	def predict_accu(self, vec):
		y = self.accu.predict(vec)
		return y.argmax()
	
	def predict_time(self, vec):

		y = self.time.predict(vec)
		return y.argmax()
		
	def predict(self, content):
		fact = self.cut.cut(content, text = True)
		sequences = self.tokenizer.texts_to_sequences([fact])
		x=sequence.pad_sequences(sequences,maxlen=100)       
		ans = {}
		accu=self.predict_accu(x)
		accu_name=get_name(accu,'accu')        
		law=self.predict_law(x)
		law_name=get_name(law,'law')
		time=self.predict_time(x)
		time_name=get_name(time,'time')        
        
		ans['accusation'] = accu_name
		ans['articles'] = law_name
		ans['imprisonment'] = time_name

		print(ans)
		return [ans]




if __name__ == '__main__':
    content = '被害彭某甫发生纠纷后双方发生互殴,在互殴过程中被告人李1某用镰刀彭某甫头部砍伤彭某甫将被告人李某面部、头部打伤。经法医鉴定彭某甫因外伤致头皮瘢痕属轻伤二级;李1某面部及头部损失均属轻微伤。2018年1月11日,被告人李1某主动到公安机关如实供述自己的犯罪事实'
    predictor = Predictor()
    m=predictor.predict(content)






