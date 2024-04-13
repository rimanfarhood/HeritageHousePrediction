import streamlit as st

def predict_sale_price(X_live, sale_price_features, sale_price_pipeline):

	# from live data, subset features related to this pipeline
	X_live_sale_price = X_live.filter(sale_price_features)

	# predict
	sale_price_prediction_proba = sale_price_pipeline.predict(X_live_sale_price)

	# create logic to display the results
	proba = sale_price_prediction_proba
	value = float(proba.round(1))
	amount = '${:,.2f}'.format(value)
	statement = (
		f'The House Sale Price Estimation For Given Values: **Sale Price {amount}**'
		)

	st.write(statement)
