#!/usr/bin/env python
# coding: utf-8




import json
from sklearn.externals import joblib
from keras.preprocessing import sequence,text
from keras.optimizers import RMSprop
from keras.utils import np_utils,plot_model
from keras.models import Sequential,Model
from keras.layers import Dense,Dropout,Activation,Input,Convolution1D,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten,concatenate,Embedding,GRU,Lambda, LSTM, TimeDistributed
from keras.layers import Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D,GRU, TimeDistributed, Bidirectional
from keras.layers.merge import Concatenate
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
import keras
from keras.initializers import Constant
from datetime import datetime
import gensim
import numpy as np
import os
import thulac
from gensim.models import word2vec
import codecs
import initializer
from initializer import get_time,get_name,init,get_label
from keras.utils import plot_model




def read_train_data(path,slice_size=None):
	print('reading train data...')
	fin = open(path, 'r', encoding='utf8')

	alltext = []
	accu_label = []
	law_label = []
	time_label = []

	line = fin.readline()
	while line:
		d = json.loads(line)
		alltext.append(d['fact'].strip())
		accu_label.append(get_label(d, 'accu'))
		law_label.append(get_label(d, 'law'))
		time_label.append(get_label(d, 'time'))
		line = fin.readline()
	fin.close()
    
	if slice_size is None:
		print('do not slice') 
	else:
		alltext = alltext[:slice_size]
		law_label = law_label[:slice_size]
		accu_label = accu_label[:slice_size]
		time_label = time_label[:slice_size]

	print('cut text...')
	count = 0
	cut = thulac.thulac(seg_only=True)
	train_text = []
	for text in alltext:
		count += 1
		if count % 2000 == 0:
			print(count)
		train_text.append(cut.cut(text, text=True)) #分词结果以空格间隔，每个fact一个字符串
	print(len(train_text))

	fileObject = codecs.open("./predictor/cuttext.txt", "w", "utf-8")  #必须指定utf-8否则word2vec报错
	for ip in train_text:
		fileObject.write(ip)
		fileObject.write('\n')
	fileObject.close()
	print('cut text over')
        
	return train_text,law_label, accu_label, time_label



def word2vec_train(embedding_dims=100):

	sentences = word2vec.Text8Corpus("./predictor/cuttext.txt")
	model = word2vec.Word2Vec(sentences)#size=100
	model.save('./predictor/model/word2vec.m')
	#print(model.similarity('被害人','发生'))
    #print(model.wv.similar_by_word('被害人', topn=5, restrict_vocab=None))
	# print(model.most_similar("被告人"))
	print('finished and saved!')
	return model





def cnn_gru(word_index,y,max_words=20000,filters=256,max_len=100,embedding_dims=100):
    path='./predictor/model/word2vec.m'
    if os.path.exists(path): 
        model = gensim.models.Word2Vec.load(path)
    else:
        model=word2vec_train(embedding_dims)
    # # 得到一份字典(embeddings_index)，每个词对应属于自己的100维向量
    embedding_index = {}
    word_vectors = model.wv
    for word, vocab_obj in model.wv.vocab.items():
        embedding_index[word] = word_vectors[word]
    del model, word_vectors   # 删掉gensim模型释放内存
    print('Found %s word vectors.' % len(embedding_index))
    
    num_words = min(max_words, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words,embedding_dims))  # 对比词向量字典中包含词的个数与文本数据所有词的个数，取小
    for word,i in word_index.items():# 从索引为1的词语开始，用词向量填充矩阵
        if i > max_words:
            continue
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:# 文本数据中的词在词向量字典中没有，向量为取0；如果有则取词向量中该词的向量
            embedding_matrix[i] = embedding_vector # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充
    
    embedding_layer = Embedding(num_words,
                            embedding_dims,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_len,
                            trainable=False)
    main_input = Input(shape=(max_len,), dtype='float64')
    embed = Embedding(max_words+1, filters, input_length=max_len)(main_input)
    cnn1 = Convolution1D(filters, 3, padding='same', strides = 1, activation='relu')(embed)
    cnn1 = MaxPool1D(pool_size=4)(cnn1)
    cnn2 = Convolution1D(filters, 4, padding='same', strides = 1, activation='relu')(embed)
    cnn2 = MaxPool1D(pool_size=4)(cnn2)
    cnn3 = Convolution1D(filters, 5, padding='same', strides = 1, activation='relu')(embed)
    cnn3 = MaxPool1D(pool_size=4)(cnn3)
    cnn = concatenate([cnn1,cnn2,cnn3], axis=-1)
    gru = Bidirectional(GRU(filters, dropout=0.2, recurrent_dropout=0.1))(cnn)
    main_output = Dense(y.shape[1], activation='softmax')(gru)
    model = Model(inputs = main_input, outputs = main_output)
        
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model




def runcnn(train_data,label, label_name,max_words=20000,max_len = 100,filters=256,embedding_dims=100):
    print('train_cnn label:%s'% label_name)
    
    tokenizer = Tokenizer(num_words=max_words)  # 建立一个max_features个词的字典
    tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
    joblib.dump(tokenizer, './predictor/model/tokenizer.h5')
    word_index=tokenizer.word_index
    print(len(tokenizer.word_index),"tokens has been saved.")
    sequences = tokenizer.texts_to_sequences(train_data)  # 对每个词编码之后，每个文本中的每个词就可以用对应的编码表示，即每条文本已经转变成一个向量了 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)

    x = sequence.pad_sequences(sequences, max_len)  # 将每条文本的长度设置一个固定值。
    
    y = np_utils.to_categorical(label) #多分类时，此方法将1，2，3，4，....这样的分类转化成one-hot 向量的形式，最终使用softmax做为输出
    print('input_size:',x.shape,y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    x_train = x[indices][:int(lenofdata*0.8)]
    y_train = y[indices][:int(lenofdata*0.8)]
    x_test = x[indices][int(lenofdata*0.8):]
    y_test = y[indices][int(lenofdata*0.8):]

    model = cnn_gru(word_index,y,max_words,filters,max_len,embedding_dims)
    earlyStopping = keras.callbacks.EarlyStopping(
			monitor='val_loss',
			patience=0,  # 当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
			verbose=0,
			mode='auto')

    print("training model")
    model.fit(x_train,y_train,validation_split=0.2,batch_size=32,epochs=10,verbose=2,callbacks=[earlyStopping],shuffle=True,class_weight='auto') #自动设置class weight让每类的sample对损失的贡献相等


    print("pridicting...")
    scores = model.evaluate(x_test,y_test)
    print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))

    print("saving %s_model" % label_name)
    model.save('./predictor/model/%s_textcnn.h5' % label_name)
    return model 





if __name__ == "__main__":
    #law, accu,time, lawname, accuname,timename = initializer.init()
    train_data,law_label, accu_label, time_label = read_train_data(path='./data/final_all_data/first_stage/train.json',slice_size=5000)
    law_model=runcnn(train_data,law_label, 'law') 
    accu_model=runcnn(train_data,accu_label, 'accu')
    time_model=runcnn(train_data,time_label, 'time')    






