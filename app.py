#!/usr/bin/env python
# coding: utf-8

# In[84]:


from flask import Flask, render_template, request
import numpy as np
import cv2 as cv
import os


# In[85]:


from keras.models import load_model

mod=load_model('D:/imp_programm/monkey_breed_classifier/MONKEY_BREED_MOBILENET_MODEL.h5')


# In[86]:


app = Flask(__name__)


# In[87]:


path='D:/imp_programm/monkey_breed_classifier/upload_files/'


# In[88]:


dic=  {'n0' :   'mantled_howler',          
        'n1' : 'patas_monkey'                  , 
        'n2'  : 'bald_uakari'                   , 
        'n3' :  'japanese_macaque'              , 
        'n4'  :  'pygmy_marmoset'                , 
        'n5'  :  'white_headed_capuchin'         , 
        'n6'  :  'silvery_marmoset'              , 
        'n7'  :  'common_squirrel_monkey'        , 
        'n8'  :  'black_headed_night_monkey'     , 
       'n9'  :  'nilgiri_langur' }


# In[89]:


def preprocess(w):
    inp=cv.imread(w)
    inp=cv.resize(inp,(224,224),interpolation=cv.INTER_LINEAR)
    inp=inp/255
    inp=inp.reshape(1,224,224,3)
    

    return inp


# In[90]:


@app.route('/')
def home():
    return render_template('index.html')


# In[91]:


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file=request.files['image']
        
        if file:
            loc=os.path.join(path,file.filename)
            file.save(loc)
            inp=preprocess(loc)
            pred=np.argmax(mod.predict(inp))
            
            
            return render_template('index.html', pred=pred,img_path=loc)
       
        else:
            return render_template('index.html')


# inp=preprocess(image)       
# pred=np.argmax(mod.predict(inp))
# name= dic['n'+str(pred)]

# In[94]:


if __name__ == '__main__':
	app.run(debug=True)


# In[ ]:




