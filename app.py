import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

#Title 
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('california Housing Price Prediction')

st.image('https://nycdsa-blog-files.s3.us-east-2.amazonaws.com/2021/03/chaitali-majumder/house-price-497112-KhCJQICS.jpg')

st.header('a model of housing prices to predict median house values in California',divider = True)

#st.subheader('''User Must Enter Given  Values To Predict Price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://www.reduceloans.com.au/wp-content/uploads/2020/02/House-prices-rising-in-2020.jpg')

# read_data
temp_df = pd.read_csv('california.csv')

random.seed(52)

all_values = []

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var =st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])

with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]


import time

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Price')
place = st.empty()
place.image('https://i.pinimg.com/originals/f5/27/0a/f5270acbc4b98112fcd520d2eea023de.gif',width = 200)

if price>0:

    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)

    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    # st.subheader(body)

    st.success(body)
else:
    body = 'Invalid House features Values'
    st.warning(body)
