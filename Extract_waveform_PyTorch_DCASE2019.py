# -*- coding: utf-8 -*-
import numpy as np
import os
import soundfile as sf

from multiprocessing import Process

def extract_waveforms(l_utt):
	for line in l_utt:
		wav ,_= sf.read(line,dtype='float32')	#wav shape: (480000, 2)
		wav= np.asarray(wav,dtype=np.float32).T 

		dir_base, fn = os.path.split(line)
		dir_base, _ = os.path.split(dir_base)
		fn, _ = os.path.splitext(fn) 
		if not os.path.exists(dir_base + _dir_name):
			os.makedirs(dir_base + _dir_name)
		np.save(dir_base+_dir_name+fn, wav)
	return 


_nb_proc = 24
_fs = 32000
_trg_ext = 'wav'
_dir_dataset = 'C:/DB/TAU-urban-acoustic-scenes-2019-development/audio/'
_dir_name = '/wave_np/'
if __name__ == '__main__':
	l_utt = []
	for r, ds, fs in os.walk(_dir_dataset):
		for f in fs:
			if os.path.splitext(f)[1] != '.'+_trg_ext: continue
			l_utt.append('/'.join([r, f.replace('\\', '/')]))

	nb_utt_per_proc = int(len(l_utt) / _nb_proc)
	l_proc = []
	for i in range(_nb_proc):
		if i == _nb_proc - 1:
			l_utt_cur = l_utt[i * nb_utt_per_proc :]
		else:
			l_utt_cur = l_utt[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]
		l_proc.append(Process(target = extract_waveforms, args = (l_utt_cur,)))
		print('%d'%i)

	for i in range(_nb_proc):
		l_proc[i].start()
		print('start %d'%i)
	for i in range(_nb_proc):
		l_proc[i].join()
