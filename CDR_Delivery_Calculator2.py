import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

st.title("CDR Weathering Rate Calculator")

# File uploader
uploaded_file = st.file_uploader("Upload your geochemical dataset (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Dropdown to choose analysis column
    analysis_column = st.selectbox(
        "Choose the column for weathering rate analysis:",
        options=["Total_Ca_Mg_moles", "Ca_moles"]
    )

    # Field filter
    field_list = sorted(data["Field ID"].dropna().unique())
    selected_fields = st.multiselect("Select Field IDs for analysis:", field_list, default=field_list)

    if selected_fields:
        data = data[data["Field ID"].isin(selected_fields)]

        # Clean and coerce to numeric
        data[analysis_column] = pd.to_numeric(data[analysis_column], errors='coerce')

        field_weathering = []
        for field in selected_fields:
            subfield = data[data['Field ID'] == field]

            bl = np.nanmedian(subfield[subfield['Sample Type'] == 'BL'][analysis_column])
            blp = np.nanmedian(subfield[subfield['Sample Type'] == 'BLP'][analysis_column])
            r1 = np.nanmedian(subfield[subfield['Sample Type'] == 'R1'][analysis_column])

            if not any(pd.isnull([bl, blp, r1])) and (blp - bl) != 0:
                fw = (blp - r1) / (blp - bl) * 100
                field_weathering.append((field, fw))

        if field_weathering:
            df_results = pd.DataFrame(field_weathering, columns=['Field ID', 'Fw (%)'])
            st.subheader("Weathering Rate Summary")
            st.dataframe(df_results)

            # Plotting
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df_results['Fw (%)'])
            plt.title("Weathering Rate (Fw %) Distribution")
            plt.ylabel("Fw (%)")
            st.pyplot(plt)
        else:
            st.warning("Not enough valid data across BL, BLP, and R1 for selected fields.")
