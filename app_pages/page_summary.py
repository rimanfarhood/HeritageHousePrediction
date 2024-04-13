import streamlit as st


def page_summary_body():

    st.write("## Project Summary")
    # copied from README file - "Business Requirements" section
    st.subheader("Business requirements")
    st.write(
        "**Requirement 1:**\n\n"
        " - The client is interested in discovering how house attributes"
        " correlate with sale prices. Therefore, the client expects data "
        " visualizations of the correlated variables against the sale price\n\n"
        "- The client is interested in predicting the house sale prices "
        "from her 4 inherited houses, and any other house in Ames, Iowa.\n\n"
        )
    st.write(
        "**Requirement 2:**\n\n"
        " - Deliver an ML model with the capability of predicting the\n"
        "sale price of the four inherited houses and any house in Ames,"
        "Iowa with at least 75% accuracy."
        )
    st.write("---")
    # text based on README file - "Dataset Content" section
    st.subheader("Jargon\n")
    st.write(
        "- **SalePrice** is the target variable.\n"
        "- **1stFlrSF**: First floor square feet\n"
        "- **GrLivArea**: Above grade (ground) living area square feet\n"
        "- **GarageYrBlt**: Year garage was built\n"
        "- **TotalBsmtSF**: Total square feet of basement area\n"
        "- **GarageArea**: Size of garage in square feet\n"
        "- **YearBuilt**: Original construction date\n"
    )
    st.write(
        "**For better understanding and interpretation I "
        "recommend to have a look through all the metadata**"
        )
    if st.checkbox("View all metadata"):
        st.write(
            "- **2ndFlrSF**: Second floor square feet\n\n"
            "- **BedroomAbvGr**: Bedrooms above grade (does NOT include basement bedrooms)\n"
            "- **BsmtExposure**: Refers to walkout or garden level walls\n"
            "   - Gd: Good Exposure;\n"
            "   - Av: Average Exposure;\n"
            "   - Mn: Mimimum Exposure;\n"
            "   - No: No Exposure;\n"
            "   - None: No Basement\n\n"
            "- **BsmtFinType1**: Rating of basement finished area\n\n"
            "   - GLQ: Good Living Quarters;\n"
            "   - ALQ: Average Living Quarters;\n"
            "   - BLQ: Below Average Living Quarters;\n"
            "   - Rec: Average Rec Room;\n"
            "   - LwQ: Low Quality;\n"
            "   - Unf: Unfinshed;\n"
            "   - None: No Basement\n\n"
            "- **BsmtFinSF1**: Type 1 finished square feet\n"
            "- **BsmtUnfSF**: Unfinished square feet of basement area\n"
            "- **GarageFinish**: Interior finish of the garage\n"
            "   - Fin: Finished;\n"
            "   - RFn: Rough Finished;\n"
            "   - Unf: Unfinished;\n"
            "   - None: No Garage\n"
            "- **KitchenQual**: Kitchen quality\n"
            "   - Ex: Excellent\n"
            "   - Gd: Good\n"
            "   - TA: Typical/Average\n"
            "   - Fa: Fair\n"
            "   - Po: Poor\n\n"
            "- **LotArea**: Lot size in square feet\n"
            "- **LotFrontage**: Linear feet of street connected to property\n"
            "- **MasVnrArea**: Masonry veneer area in square feet\n"
            "- **OpenPorchSF**: Open porch area in square feet\n"
            "- **OverallCond**: Rates the overall condition of the house\n"
            "   - 10: Very Excellent\n"
            "   - 9: Excellent\n"
            "   - 8: Very Good\n"
            "   - 7: Good\n"
            "   - 6: Above Average\n"
            "   - 5: Average\n"
            "   - 4: Below Average\n"
            "   - 3: Fair\n"
            "   - 2: Poor\n"
            "   - 1: Very Poor\n\n"
            "- **OverallQual**: Rates the overall material and finish of the house\n"
            "   - 1-10 same as OverallCond"
            "- **YearRemodAdd**: Remodel date (same as construction date if no remodeling or additions)\n"
            "- **SalePrice**: Sale Price\n\n"
            )

    st.write(    
        "The dataset contains housing profile records for houses built "
        "between 1872 and 2010 in Ames, Iowa.\n\n"
        "The dataset have various abbreviated terms used to describe their"
        "features.\n\n"
        "To view the dataset with all the information, "
        "Click -->  **[Link To Dataset]"
        "(https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)"
        "**."
    )
    
    st.write("---")

    # Link to README file, for users to access the full project documentation
    st.write(
        "* For additional information, please visit and **read** the "
        "[Project README file]"
        "(https://github.com/rimanfarhood/HeritageHousePrediction/blob/main/"
        "README.md)."
        )
    
