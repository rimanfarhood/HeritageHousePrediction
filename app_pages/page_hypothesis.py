import streamlit as st


def page_hypothesis_body():

    st.write("### Hypothesis and Validation")
    st.write(
        "After some researching in the field of real "
        "estate and my prior knowledge I have come to this hypothesis."
    )
    st.write(
        "- **YearBuilt**: Original construction date\n"
        "- **TotalBsmtSF**: Total square feet of basement area\n"
        "- **GrLivArea**: Above grade (ground) living area square feet\n"
        "- **OverallQual**: Rates the overall material and finish of the house\n"
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
        "- **KitchenQual**: Kitchen quality\n"
        "   - Ex: Excellent\n"
        "   - Gd: Good\n"
        "   - TA: Typical/Average\n"
        "   - Fa: Fair\n"
        "   - Po: Poor\n\n"
    )
    st.write("---")
    st.write("**Hypothesis**")
    st.write(
        "Beside from external factors, these are the factors that have the "
        "greatest impact on the sale price according to my hypothesis:\n\n"
        "- The overall quality.\n\n"
        "- The size of the house, total square feet.\n\n"
        "- Number of bedrooms and bathrooms.\n\n"
        "- The condition of kitchen and bathroom, newly renovated has"
        " a great impact on sale price.\n\n" 
        "- The Year it was built.\n\n"
    )
    st.write("---")

    st.write("**Validation**")
    st.write("We validate the hypothesis by conducting a Correlation Study.")
    st.write(
        "**The Correlation study validates that:**\n\n"
        "- The Sale Price tends to be higher for houses with better **Overall "
        "Quality 'OverallQual'**, affirming the first point of the hypothesis."
        "\n\n"
        "- An increase in **above-grade living area 'GrLivArea'** leads to a"
        " increase in the Sale Price. As well as an increase in **total"
        " basement square footage 'TotalBsmtSF'**  tends increase "
        " the Sale Price Which validates the second point of the"
        " hypothesis.\n\n"
        "- The Correlation study did **not** validate the **third** point"
        " in my hypothesis.\n\n"
        "- **KitchenQual_TA** has a high negative correlation to the sale "
        "price which indicates that houses with **'Typical/average' kitchen"
        " quality** tends to decrees the sale price. Which validates a part of"
        "the fourth point in the hypothesis, the condition of the kitchen\n\n"
        "- The sale price tends to increase the more up to date the year that"
        " they were built **YearBuilt**, validating the last point in my"
        " hypothesis"
        )
