import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_management import housing_data


def page_study_body():
    # load data
    df = housing_data()

    # Spearman and pearson correlation plots
    spearman = plt.imread('outputs/study_plots/spearman.png')
    pearson = plt.imread('outputs/study_plots/pearson.png')
    # Eda Plots
    scatterplots = plt.imread('outputs/study_plots/scatterplots.png')
    barplots = plt.imread('outputs/study_plots/barplots.png')

    st.write("## Sale Price Correlation Study")
    st.write("Business Requirement 1")
    st.write(
        "- The client is interested in discovering how house attributes"
        " correlate with sale prices. Therefore, the client expects data "
        "visualizations Showing the most relevant variables correlated"
        " to sale price"
        )
    st.write(
        "The data have been studied by conducting correlation studies both" 
        " Spearman and Pearson. The findings has been further explored and"
        " plotted against the target to visualizing insights.\n\n"
        )
    
    # inspect data
    if st.checkbox("Inspect House Database"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"printed below are the first 10 rows.")
        st.write(df.head(10))
    st.write("---")
    st.write("**Metadata**")
    st.write(
        "- **1stFlrSF**: First floor square feet\n"
        "- **GrLivArea**: Above grade (ground) living area square feet\n"
        "- **GarageYrBlt**: Year garage was built\n"
        "- **TotalBsmtSF**: Total square feet of basement area\n"
        "- **GarageArea**: Size of garage in square feet\n"
        "- **YearBuilt**: Original construction date\n"
        "- **2ndFlrSF**: Second floor square feet\n\n"
        "- **BedroomAbvGr**: Bedrooms above grade (does NOT include"
        " basement bedrooms)\n"
    )
    st.write(
        "**For better understanding and interpretation I "
        "recommend to have a look through the metadata**"
        )
    if st.checkbox("View all metadata"):
        st.write(
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
            "   - 1-10 same as OverallCond\n"
            "- **YearRemodAdd**: Remodel date (same as construction date if no remodeling or additions)\n"
            "- **SalePrice**: Sale Price\n\n"
            )

    # hard copied from study notebook
    vars_to_study = [
        '1stFlrSF: First floor square feet', 
        'GarageArea', 
        'GrLivArea: Above grade (ground) living area square feet',
        'OverallQual: Overall Quality',  
        'TotalBsmtSF: Total square feet of basement area', 
        'YearBuilt',
        ]
    
    features = [
        'KitchenQual_TA: Kitchen Quality Typical/Average', 
        'GarageFinish_Unf: Garage is Unfinished',
        'MasVnrArea_0.0: Masonry veneer area 0.0 in square feet',
        'GarageYrBlt_Missing: Garage Year Built Missing',
        'GarageFinish_None: Garage is Not finished']

    st.write("---")

    # Correlation Pots
    st.write("### Spearman & Pearson Correlation Plots")
    # Spearman
    st.write("**Spearman Correlation**")
    st.image(spearman)
    # Pearson
    st.write("**Pearson Correlation**")
    st.image(pearson)

    st.write("---")

    # Target vs Feature Plots
    # Top positive features
    st.write("**Positive Correlated Features Investigated:**")
    st.write("Expand the image to get a clearer view of the plots")
    st.write(vars_to_study)
    st.image(scatterplots)
    
    # The correlations and plots interpretation converge. 
    st.write(
        "*  Houses with larger garages **(GarageArea)** are likely to have a "
        "increased Sale Price, indicating that buyers values"
        " spacious garages.\n"
        "* An increase in total basement square footage **(TotalBsmtSF)** often "
        "leads to an increase in the Sale Price, which indicates that basement"
        " area is an important factor in house valuation.\n"
        "* The Sale Price tends to rise with the size of the first floor "
        "**(1stFlrSF)**, which shows the significance of main-level living"
        " space in the housing market.\n"
        "* The Sale Price tends to be higher for houses with better Overall "
        "Quality **(OverallQual)**, affirming that quality is a crucial "
        "determinant of property value.\n"
        "* An increase in above-grade living area **(GrLivArea)** leads to a"
        " rise in the Sale Price, which reflects the market's valuation"
        " of living space.\n"
        "* The sale price tends to increase the more up to date the year"
        " that they were built **(YearBuilt)**.\n\n"
        )

    st.write("---")
    # Top negative correlated features
    st.write("**Negative Correlated Features Investigated:**")
    st.write(features)
    st.write("Expand the image to get a clearer view of the plots")
    st.image(barplots)
    
    #**The correlations and plots interpretation converge.** 
    st.write(
            "* **(KitchenQual_TA )** indicates that the Sale Price of houses with"
            " Typical/Average kitchen quality tends to decrease.\n"
            "* The Sale Price is typically lower when the garage is not"
            "  finished, as shown by **(GarageFinish_Unf)**.\n"
            "* Houses without any masonry veneer area tend to have a lower"
            " Sale Price, indicated by **(MasVnrArea_0.0)**. \n"
            "* Houses missing records of the year the garage was built tend to"
            " to have a decrease on the Sale Price, "
            "as shown by **(GarageYrBlt_Missing)**.\n"
            "* The Sale Price usually decreases on houses without garages,"
            " based on **(GarageFinish_None)**.\n\n"
            "**Buyers are willing to pay premiums for more space and higher "
            "quality in homes, as these patterns demonstrate the significance"
            " of size and quality in valuation.**"
            )
