'''
This file is used to predict risk for 3 drug groups
- stimulants           
- depressants            
- hallucinogens            

The predictiv model is fitted with the data from the source:
archive.ics.uci.edu/ml/datasets/Drug+consumption+%28quantified%29
'''

import numpy as np
import pandas as pd
import pickle

# To use the prediction function, the user has to insert a DataFrame which has the following columns
ls_col_names = ['nscore', 'escore', 'oscore', 'ascore', 'cscore', 'impulsive', 'ss',
       'age__0.95197', 'age__0.07854', 'age_0.49788', 'age_1.09449',
       'age_1.82213', 'age_2.59171', 'gender__0.48246', 'gender_0.48246',
       'education__2.43591', 'education__1.7379', 'education__1.43719',
       'education__1.22751', 'education__0.61113', 'education__0.05921',
       'education_0.45468', 'education_1.16365', 'education_1.98437',
       'country__0.57009', 'country__0.46841', 'country__0.28519',
       'country__0.09765', 'country_0.21128', 'country_0.24923',
       'country_0.96082', 'ethnicity__1.10702', 'ethnicity__0.50212',
       'ethnicity__0.31685', 'ethnicity__0.22166', 'ethnicity_0.1144',
       'ethnicity_0.126', 'ethnicity_1.90725']

# This Values are just example values
val = [-2.75696, -0.43999, -1.11902, -1.47955,  0.93949, -0.21712,
         0.07987,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,
         1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,
         0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,
         0.     ,  0.     ]
val = [[val[i]] for i in range(len(val))]
dic = {k:v for k,v in zip(ls_col_names,val)}
X_example = pd.DataFrame(dic)

if __name__ == '__main__':
    try:
        def pred_stimulants(X_input):
            filename        = 'model_stimulants.sav'
            loaded_model    = pickle.load(open(filename, 'rb'))

            return loaded_model.predict(X_input)

        def pred_depressants(X_input):
            filename        = 'model_depressants.sav'
            loaded_model    = pickle.load(open(filename, 'rb'))

            return loaded_model.predict(X_input)

        def pred_hallucinogens(X_input):
            filename        = 'model_hallucinogens.sav'
            loaded_model    = pickle.load(open(filename, 'rb'))

            return loaded_model.predict(X_input)

        def pred_all(X_input):
            return {'stimulants_pred': pred_stimulants(X_input), 'depressants_pred': pred_depressants(X_input), 'hallucinogens_pred': pred_hallucinogens(X_input)}

        # Making a prediction for our example
        print('===================')
        print('=== Predictions ===')
        print('===================')
        print(pred_all(X_example))

    except:
        print("Error occurred")