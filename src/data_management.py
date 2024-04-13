import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def housing_data():
    df = pd.read_csv("outputs/datasets/collection/cleaned/CleanedHousePricing.csv")
    return df

def inherited_house_data():
    in_df = pd.read_csv("outputs/datasets/collection/InheritedHouses.csv")
    return in_df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

# code copied from Code Institute's Churnornmeter Project with some adjustments