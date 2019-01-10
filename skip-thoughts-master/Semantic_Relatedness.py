# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:13:25 2018

@author: student
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 02:00:27 2018

@author: puiyan
"""
import datetime
import skipthoughts
import nltk
nltk.download('punkt')
#THEANO_FLAGS='floatX=float32,device=cuda0,gpuarray.preallocate=1' 

print("Start Time :" , datetime.datetime.now())
model = skipthoughts.load_model()

print(datetime.datetime.now())
encoder = skipthoughts.Encoder(model)

print(datetime.datetime.now())
import eval_sick
eval_sick.evaluate(encoder, evaltest=True)

print("End time: " , datetime.datetime.now())

#print(datetime.datetime.now())
#nltk.download('punkt')

#vectors = encoder.encode(['I am a boy','I am a man'])

