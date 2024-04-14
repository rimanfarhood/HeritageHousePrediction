import streamlit as st
import pandas as pd

from src.data_management import housing_data, load_pkl_file, inherited_house_data
from src.machine_learning.prediction_analysis import predict_sale_price



def page_prediction_body():

	# load predict sale_price files	
	pipeline = load_pkl_file("outputs/ml_pipeline/predict_SalePrice/v1/pipeline.pkl")
	sale_price_features = (pd.read_csv(f"outputs/ml_pipeline/predict_SalePrice/v1/X_train.csv").columns.to_list())
	st.write("## House Price Prediction")
	st.subheader("Business Requirement 2")
	st.write(
		"2.1 The client is interested to predict the sale price for her 4 "
		"inherited houses.\n\n"
		"2.2 As well as any other house in Ames, Iowa. "
		)
	st.write(
		"Page Summary\n\n"
		"-Summed Predicted Price For All 4 Inherited Houses.\n\n"
		"-The 4 Houses' Attributes & Their Respective Predicted Sale Price\n\n"
		"-Predict Housing Sale Prices with Real-Time Data"
	)
	st.write("---")

	# Inherited houses 
	in_df = inherited_house_data()

	house1 = in_df.iloc[[0]]
	house2 = in_df.iloc[[1]]
	house3 = in_df.iloc[[2]]
	house4 = in_df.iloc[[3]]

	features = [
		'GarageArea', 'GrLivArea', 
		'OverallQual','TotalBsmtSF', 
		'YearBuilt']

	st.subheader("The Summed Predicted Price For All 4 Inherited Houses is:\n 625 133.3$") 
	
	st.subheader(
		"\n The 4 Houses' Attributes & Their Respective Predicted Sale Price.\n"
		)
	
	st.write("### House 1 Attributes")	
	st.code(house1[features])
	predict_sale_price(house1, features, pipeline)
	
	st.write("### House 2 Attributes")	
	st.code(house2[features])
	predict_sale_price(house2, features, pipeline)
	
	st.write("### House 3 Attributes")	
	st.code(house3[features])
	predict_sale_price(house3, features, pipeline)
	
	st.write("### House 4 Attributes")	
	st.code(house4[features])
	predict_sale_price(house4, features, pipeline)

	st.write("---")
	st.subheader("Predict Housing Sale Prices")
	st.write(
		"Below You Can Enter Real-Time House Data "
		"To Get An Estimated Sales Price Prediction."
		)

	# predict on live data
	X_live = DrawInputsWidgets()

	if st.button('Predict Sale Price'):
		st.write("**The Predicted Sale Price:**")
		predict_sale_price(X_live, sale_price_features, pipeline)


def DrawInputsWidgets():

	# load dataset
	df = housing_data()

    # we create input widgets for 5 features 	
	col1, col2 = st.beta_columns(2)
	col3, col4, col5 = st.beta_columns(3)

	# We are using these features to feed the ML pipeline
		
	# create an empty DataFrame, which will be the live data
	X_live = pd.DataFrame([], index=[0]) 
	
	# from here on we draw the widget based on the variable type (numerical or categorical)
	# and set initial values

	with col1:
		feature = "OverallQual"
		st_widget = st.number_input(
			label= 'Overall Quality: Rates the overall material and finish of the house: 1 - 10',
			min_value= 1, 
			max_value= 10,
            step = 1       
			)
	X_live[feature] = st_widget

	with col2:
		feature = "GrLivArea"
		st_widget = st.number_input(
			label= 'Above grade (ground) living area square feet: 334 - 5642',
			min_value= 334,
			max_value= 5642, 
            step= 10
			)
	X_live[feature] = st_widget

	with col3:
		feature = "TotalBsmtSF"
		st_widget = st.number_input(
			label= 'Total square feet of basement area: 0 - 6110',
			min_value= 0,
			max_value= 6110, 
            step= 10
			)
	X_live[feature] = st_widget

	with col4:
		feature = "GarageArea"
		st_widget = st.number_input(
			label= "Garage Area: Size of garage in square feet: 0 - 1418",
			min_value= 0,
			max_value= 1418, 
            step= 10
			)
	X_live[feature] = st_widget

	with col5:
		feature = "YearBuilt"
		st_widget = st.number_input(
			label= "Year Built: 1872-2010", 
			min_value= 1872,
			max_value= 2010,
            step= 1
			)
	X_live[feature] = st_widget

	return X_live
