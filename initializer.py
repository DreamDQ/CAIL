#!/usr/bin/env python
# coding: utf-8




def init():
	f = open('./data/law.txt', 'r', encoding='utf8')
	law = {}             # law{0: '184', 1: '336', 2: '314', ....}
	lawname = {}         # lawname{184:0,336:2,...}
	line = f.readline()
	while line:
		lawname[len(law)] = line.strip()
		law[line.strip()] = len(law)
		line = f.readline()
	f.close()

	f = open('./data/accu.txt', 'r', encoding='utf8')
	accu = {}
	accuname = {}
	line = f.readline()
	while line:
		accuname[len(accu)] = line.strip()  # {{0: '妨害公务', 1: '寻衅滋事', 2: '盗窃、侮辱尸体'
		accu[line.strip()] = len(accu)      # {'寻衅滋事': 1, '绑架': 8, ',...}
		line = f.readline()
	f.close()
	
	time_list=['death_penalty','life_imprisonment','10_year_imprisonment','9_year_imprisonment',
	'8_year_imprisonment','7_year_imprisonment','6_year_imprisonment','5_year_imprisonment',
	'4_year_imprisonment','3_year_imprisonment','2_year_imprisonment','1_year_imprisonment']
	time = {}
	timename={}
	for i in range(len(time_list)):
		timename[i] = time_list[i]  
		time[time_list[i]] = i   
        
   
   	


	return law, accu, time, lawname, accuname, timename

law, accu, time, lawname, accuname, timename = init()


def get_time(time):
	# 将刑期用分类模型来做
	v = int(time['imprisonment'])

	if time['death_penalty']:
		return 0
	if time['life_imprisonment']:
		return 1
	elif v > 10 * 12:
		return 2
	elif v > 7 * 12:
		return 3
	elif v > 5 * 12:
		return 4
	elif v > 3 * 12:
		return 5
	elif v > 2 * 12:
		return 6
	elif v > 1 * 12:
		return 7
	else:
		return 8


def get_name(index, kind):
	global lawname
	global accuname
	global timename     

	if kind == 'law':
		return lawname[index]

	if kind == 'accu':
		return accuname[index]

	if kind == 'time':
		return timename[index]


def get_label(d, kind):
	global law
	global accu
	global time
	if kind == 'law':
		# 返回多个类的第一个
		return law[str(d['meta']['relevant_articles'][0])]
	if kind == 'accu':
		return accu[d['meta']['accusation'][0]]

	if kind == 'time':
		return get_time(d['meta']['term_of_imprisonment'])










