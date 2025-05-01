import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit setup
st.set_page_config(layout="wide")
st.title("Lithos Carbon tCDR Delivery Calculator")

uploaded_file = st.file_uploader("Upload your Geochem CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("CSV uploaded!")
        st.write(data.head())

        grouping_options = ["Field ID", "Farm ID", "Deal ID", "No Grouping (All Data)"]
        grouping_choice = st.selectbox("Group analysis by:", options=grouping_options, key="grouping_choice_main")

        if 'Grower, Deal ID' not in data.columns:
            st.error("The uploaded CSV is missing the 'Grower, Deal ID' column.")
            st.stop()

        data[['Grower', 'Deal ID']] = data['Grower, Deal ID'].str.split(', ', expand=True)
        data.drop(columns=['Grower, Deal ID'], inplace=True)

        treatments = data[data['Sample Type'].isin(['BL', 'BLP', 'R1'])].copy()
        treatments = treatments[(treatments['CaO'].notnull()) & (treatments['MgO'].notnull())]

        conversion_factors = {
            'SiO2': 0.4674, 'Al2O3': 0.5293, 'Fe2O3': 0.6994, 'CaO': 0.7147,
            'MgO': 0.6030, 'Na2O': 0.7419, 'K2O': 0.8301, 'Cr2O3': 0.6842,
            'TiO2': 0.5995, 'MnO': 0.7745, 'P2O5': 0.4364, 'SrO': 0.8456, 'BaO': 0.8957
        }

        for oxide, factor in conversion_factors.items():
            if oxide in treatments.columns:
                treatments[oxide] = pd.to_numeric(treatments[oxide], errors='coerce')
                treatments[oxide + '_elemental'] = treatments[oxide] * factor

        for element in ['Ni', 'Cr', 'LOI', 'Eu', 'Sm', 'Gd', 'V', 'Sc']:
            if element in treatments.columns:
                treatments[element] = pd.to_numeric(treatments[element], errors='coerce')

        treatments['Ca_moles'] = treatments['CaO_elemental'] / 100 * 1000 / 40.078
        treatments['Mg_moles'] = treatments['MgO_elemental'] / 100 * 1000 / 24.305
        treatments['Total_Ca_Mg_moles'] = treatments['Ca_moles'] + treatments['Mg_moles']
        data = treatments

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        if grouping_choice == "No Grouping (All Data)":
            allowed_pairs = [
                ("Adam Wilbourne", "965"), ("Colin Garrett", "907"), ("Daniel Jones", "997"), ("Danny Williams", "891"),
                ("Jay Foushee Chip Stone", "910"), ("Joe Overby", "861"), ("John Tyndall", "964"), ("Jordan Blaylock", "946"),
                ("Jordan Mitchell", "860"), ("MD Capps", "961"), ("Merlin Brougher", "894"), ("Robert Elliot", "960"),
                ("Rodger Overby", "975"), ("Stephen Sizemore", "962"), ("Will Sandling", "1006")
            ]

            data['Grower_DealID'] = data['Grower'] + ', ' + data['Deal ID'].astype(str)
            allowed_keys = [f"{g}, {d}" for g, d in allowed_pairs]
            data = data[data['Grower_DealID'].isin(allowed_keys)]

            selected_keys = st.multiselect("Select Grower, Deal ID pairs:", options=sorted(allowed_keys), default=allowed_keys)
            data = data[data['Grower_DealID'].isin(selected_keys)]

            data['__all__'] = "All Samples"
            grouping_choice = '__all__'

            filtered_data = data.copy()
            filtered_data['Grower, Deal ID'] = filtered_data['Grower'] + ', ' + filtered_data['Deal ID'].astype(str)
            cols = ['Grower, Deal ID'] + [col for col in filtered_data.columns if col not in ['Grower, Deal ID', 'Grower', 'Deal ID']]
            filtered_data = filtered_data[cols]

            st.markdown("### \U0001F4E5 Download Raw Filtered Data")
            filtered_csv = convert_df(filtered_data)
            st.download_button(
                label="Download CSV of Selected Growers",
                data=filtered_csv,
                file_name='filtered_raw_data.csv',
                mime='text/csv',
            )

        full_datasets = []
        for group in data[grouping_choice].unique():
            subset = data[data[grouping_choice] == group]
            types_present = subset['Sample Type'].unique()
            if all(x in types_present for x in ['BLP', 'R1']):
                full_datasets.append(group)

        csv = convert_df(data)

        st.markdown(f"✅ **{len(full_datasets)} unique groups** with complete datasets")
        st.markdown(f"\U0001F469‍\U0001F33E Across **{data['Grower'].nunique()} unique growers**")

        results = []
        n_bootstrap = 20000

        for group in full_datasets:
            subset = data[data[grouping_choice] == group]
            regional = data[data['Farm ID'] == 'Regional']
            bl = pd.to_numeric(regional[regional['Sample Type'] == 'BL']['Total_Ca_Mg_moles'], errors='coerce').dropna().values
            blp = pd.to_numeric(subset[subset['Sample Type'] == 'BLP']['Total_Ca_Mg_moles'], errors='coerce').dropna().values
            r1 = pd.to_numeric(subset[subset['Sample Type'] == 'R1']['Total_Ca_Mg_moles'], errors='coerce').dropna().values
            r2 = pd.to_numeric(subset[subset['Sample Type'] == 'R2']['Total_Ca_Mg_moles'], errors='coerce').dropna().values if 'R2' in subset['Sample Type'].values else None

            if min(len(bl), len(blp), len(r1)) == 0:
                continue

            bl_samples = np.random.choice(bl, (n_bootstrap, len(bl)), replace=True)
            blp_samples = np.random.choice(blp, (n_bootstrap, len(blp)), replace=True)
            r1_samples = np.random.choice(r1, (n_bootstrap, len(r1)), replace=True)
            r2_samples = np.random.choice(r2, (n_bootstrap, len(r2)), replace=True) if r2 is not None and len(r2) > 0 else None

            bl_means = np.nanmean(bl_samples, axis=1)
            blp_means = np.nanmean(blp_samples, axis=1)
            r1_means = np.nanmean(r1_samples, axis=1)
            r2_means = np.nanmean(r2_samples, axis=1) if r2_samples is not None else None

            denominator = blp_means - bl_means
            Fw_vals = np.where(denominator != 0, (blp_means - r1_means) / denominator * 100, np.nan)
            Fw_mean = np.nanmean(Fw_vals)
            Fw_std = np.nanstd(Fw_vals)

            Fw_2_vals = None
            Fw_2_mean = None
            Fw_2_std = None
            if r2_means is not None:
                Fw_2_vals = np.where(denominator != 0, (blp_means - r2_means) / denominator * 100, np.nan)
                Fw_2_mean = np.nanmean(Fw_2_vals)
                Fw_2_std = np.nanstd(Fw_2_vals)

            results.append({
                grouping_choice: group,
                "deal_id": subset['Deal ID'].iloc[0],
                "grower": subset['Grower'].iloc[0],
                "Fw_mean": Fw_mean,
                "Fw_std": Fw_std,
                "Fw_2_mean": Fw_2_mean,
                "Fw_2_std": Fw_2_std,
                "bl_list": bl_means,
                "blp_list": blp_means,
                "r1_list": r1_means,
                "r2_list": r2_means,
                "bl_count": len(bl),
                "blp_count": len(blp),
                "r1_count": len(r1),
                "r2_count": len(r2) if r2 is not None else 0,
            })

        positive_results = sorted([r for r in results if r["Fw_mean"] > 0], key=lambda x: -x["Fw_mean"])
        negative_results = sorted([r for r in results if r["Fw_mean"] <= 0], key=lambda x: x["Fw_mean"])

        st.markdown(f"\U0001F331 **{len(positive_results)} groups** with positive weathering rates")
        st.markdown(f"\U0001FAA8 **{len(negative_results)} groups** with negative or zero weathering rates")

        summary_df = pd.DataFrame([
            {
                "Deal ID": r["deal_id"],
                grouping_choice: r[grouping_choice],
                "Grower": r["grower"],
                "BL Count": r["bl_count"],
                "BLP Count": r["blp_count"],
                "R1 Count": r["r1_count"],
                "R2 Count": r["r2_count"],
                "Fw mean (%)": round(r["Fw_mean"], 2),
                "Fw Std Dev": round(r["Fw_std"], 2),
                "Fw₂ mean (%)": round(r["Fw_2_mean"], 2) if r["Fw_2_mean"] is not None else None,
                "Fw₂ Std Dev": round(r["Fw_2_std"], 2) if r["Fw_2_std"] is not None else None,
            }
            for r in results
        ])

        st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

        summary_csv = convert_df(summary_df)

        st.download_button(
            label="Download Summary Table",
            data=summary_csv,
            file_name='summary_weathering_rates.csv',
            mime='text/csv'
        )

        st.markdown("---")
        st.header("\U0001F52C Bootstrap Distribution Viewer")

        numeric_columns = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        selected_element = st.selectbox("Select element to bootstrap:", numeric_columns)

        sample_types = st.multiselect("Select Sample Type(s):", options=['BL', 'BLP', 'R1', 'R2'], default=['BL', 'BLP', 'R1'])

        fig, ax = plt.subplots(figsize=(8, 4))

        for sample_type in sample_types:
            subset = data[data['Sample Type'] == sample_type][selected_element].dropna().values
            if len(subset) == 0:
                st.warning(f"No data for {selected_element} in {sample_type}")
                continue

            boot_samples = np.random.choice(subset, (n_bootstrap, len(subset)), replace=True)
            boot_means = np.nanmean(boot_samples, axis=1)

            sns.histplot(boot_means, bins=50, kde=True, stat='density', label=sample_type, ax=ax, alpha=0.4)

        ax.set_title(f"Bootstrap Mean Distribution for {selected_element}")
        ax.set_xlabel(f"{selected_element} Mean Value")
        ax.set_ylabel("Density")
        ax.legend(title="Sample Type")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"\u26A0\uFE0F Error during processing: {e}")
else:
    st.info("\U0001F4C4 Upload a CSV file to begin.")
