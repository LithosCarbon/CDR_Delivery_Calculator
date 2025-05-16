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

    # Show columns in uploaded file
    st.write("📄 Columns in uploaded file:", list(data.columns))

    # Dropdown to choose which column to use for Fw calculation
    analysis_column = st.selectbox(
        "Choose the column for weathering rate analysis:",
        options=["Total_Ca_Mg_moles", "Ca_moles"]
    )

    # Validate selected column exists
    if analysis_column not in data.columns:
        st.error(f"❌ Column '{analysis_column}' not found in uploaded data. Please check column names.")
        st.stop()

    # Convert column to numeric
    data[analysis_column] = pd.to_numeric(data[analysis_column], errors='coerce')

    # Dropdown to choose fields
    field_list = sorted(data["Field ID"].dropna().unique())
    selected_fields = st.multiselect("Select Field IDs for analysis:", field_list, default=field_list)

    if selected_fields:
        data = data[data["Field ID"].isin(selected_fields)]

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

            # Boxplot of Fw
            plt.figure(figsize=(10, 6))
            sns.boxplot(y=df_results['Fw (%)'])
            plt.title(f"Weathering Rate (Fw %) using {analysis_column}")
            plt.ylabel("Fw (%)")
            st.pyplot(plt)

        else:
            st.warning("⚠️ Not enough valid data across BL, BLP, and R1 for selected fields.")
    else:
        st.warning("⚠️ Please select at least one field.")
else:
    st.info("⬆️ Upload a CSV file to begin.")
