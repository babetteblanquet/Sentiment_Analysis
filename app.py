import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#st.title("Streamlit example")
html_temp = """
    <div style="background-color:#1da1f2;padding:10px">
    <h1 style="color:white;text-align:center;"> Twitter sentiment analysis</h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)




# if __name__=='__main__':
#     main()