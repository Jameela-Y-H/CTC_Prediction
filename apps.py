import numpy as np
import pandas as pd
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')
from custom_transformer import commaremover

model = joblib.load('model.pkl')

df = pd.read_csv('dataset.txt')

col_names=['College', 'Role', 'City type', 'Previous CTC',
       'Previous job changes', 'Graduation marks', 'Exp (Months)'] 

def predict_salary(College,Role,City_type,Previous_ctc,Previous_job_changes,Graduation_marks,Exp_month):
        new=[College,Role,City_type,Previous_ctc,Previous_job_changes,Graduation_marks,Exp_month]
        test=pd.DataFrame([new])
        test.columns=col_names
        predicted=model.predict(test)
        return predicted
def main():

    st.title("Salary Prediction")

    temp="""
    <div style='background-color:blue;'>
    <h2 style='color:black;text-align:centre;'> predict the salary</h2>
    </div>
    """
    st.markdown(temp,unsafe_allow_html=True)
    College=st.selectbox("Select College Tier:",pd.unique(df['College']))
    Role=st.selectbox("Select Role:",pd.unique(df['Role']))
    City_type=st.selectbox("Select City Type:",pd.unique(df['City type']))
    Previous_ctc=st.number_input('Previous CTC:',min_value=10000)
    Previous_job_changes=st.number_input('Previous Job Value:',min_value=0,max_value=25)
    Graduation_marks=st.number_input('Graduation Marks:',min_value=0,max_value=100)
    Exp_month=st.number_input('Experience in Months:',min_value=0)

    if st.button('New CTC'):
        new_sal= predict_salary(College,Role,City_type,Previous_ctc,Previous_job_changes,Graduation_marks,Exp_month)
        st.write(new_sal)



if __name__ =='__main__':
	main()
