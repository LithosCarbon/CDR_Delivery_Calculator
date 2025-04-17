import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import Image
import os

def check_password():
    def password_entered():
        if st.session_state["password"] == "Lithos1":
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("\U0001F510 Password Protected")
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.stop()


check_password()

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit setup
st.set_page_config(layout="wide")
st.title("Lithos tCDR Delivery Calculator")
logo = Image.open("lithos_logo.png")

# Put this near top of your sidebar or app layout
st.image(logo)


uploaded_file = st.file_uploader("Upload your Geochem CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("CSV uploaded!")
        st.write(data.head())

        # Grouping level selector
        grouping_options = ["Field ID", "Farm ID", "Grower", "Deal ID"]
        base_grouping_options = grouping_options.copy()

        # # Check if chemical binning is available
        # chemical_binning_enabled = False
        # farm_valid = []
        # if "Farm ID" in data.columns and "Sample Type" in data.columns:
        #     for farm in data['Farm ID'].unique():
        #         farm_data = data[data['Farm ID'] == farm]
        #         counts = farm_data['Sample Type'].value_counts()
        #         if all(sample in counts for sample in ['BL', 'BLP', 'R1']) and \
        #            counts['BL'] == counts['BLP'] == counts['R1']:
        #             farm_valid.append(farm)

        #     if farm_valid:
        #         chemical_binning_enabled = True
        #         grouping_options.append("Chemical Binning")

        grouping_choice = st.selectbox("Group analysis by:", options=grouping_options, key="grouping_choice_main")

        if 'Grower, Deal ID' not in data.columns:
            st.error("The uploaded CSV is missing the 'Grower, Deal ID' column.")
            st.stop()

        data[['Grower', 'Deal ID']] = data['Grower, Deal ID'].str.split(', ', expand=True)
        data.drop(columns=['Grower, Deal ID'], inplace=True)

        treatments = data[data['Sample Type'].isin(['BL', 'BLP', 'R1', 'R2'])].copy()
        treatments[grouping_choice] = treatments[grouping_choice].fillna('')
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

        for element in ['Ni', 'Cr', 'LOI']:
            if element in treatments.columns:
                treatments[element] = pd.to_numeric(treatments[element], errors='coerce')

        treatments['Ca_moles'] = treatments['CaO_elemental'] / 40.078
        treatments['Mg_moles'] = treatments['MgO_elemental'] / 24.305
        treatments['Total_Ca_Mg_moles'] = treatments['Ca_moles'] + treatments['Mg_moles']

        data = treatments

        # if grouping_choice == "Chemical Binning":
        #     st.markdown("### Chemical Binning Mode")

        #     compiled_df = data[data['Farm ID'].isin(farm_valid)].copy()

        #     csv = compiled_df.to_csv(index=False).encode('utf-8')
        #     st.download_button("Download Compiled Dataset", data=csv, file_name="compiled_data.csv")

        #     bin_metric = st.selectbox("Select metric to bin by:", ["SiO2/Al2O3", "LOI"], key="bin_metric")

        #     compiled_df['SiO2'] = pd.to_numeric(compiled_df.get('SiO2_elemental', np.nan), errors='coerce')
        #     compiled_df['Al2O3'] = pd.to_numeric(compiled_df.get('Al2O3_elemental', np.nan), errors='coerce')
        #     compiled_df['LOI'] = pd.to_numeric(compiled_df.get('LOI', np.nan), errors='coerce')

        #     if bin_metric == "SiO2/Al2O3":
        #         compiled_df['bin_metric'] = compiled_df['SiO2'] / compiled_df['Al2O3']
        #     else:
        #         compiled_df['bin_metric'] = compiled_df['LOI']

        #     compiled_df = compiled_df[compiled_df['bin_metric'].notnull()]
        #     compiled_df['Chemical Bin'] = pd.qcut(compiled_df['bin_metric'], 5, labels=[f"Bin {i+1}" for i in range(5)])

        #     data = compiled_df
        #     grouping_choice = "Chemical Bin"

        full_datasets = []
        for group in data[grouping_choice].unique():
            subset = data[data[grouping_choice] == group]
            types_present = subset['Sample Type'].unique()
            if all(x in types_present for x in ['BL', 'BLP', 'R1']):
                full_datasets.append(group)

        st.markdown(f"✅ **{len(full_datasets)} unique groups** with complete datasets")
        st.markdown(f"👩‍🌾 Across **{data['Grower'].nunique()} unique growers**")

        results = []
        n_bootstrap = 10000

        for group in full_datasets:
            subset = data[data[grouping_choice] == group]
            bl = pd.to_numeric(subset[subset['Sample Type'] == 'BL']['Total_Ca_Mg_moles'], errors='coerce').dropna().values
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

            Fw_2_vals = None
            Fw_2_mean = None
            if r2_means is not None:
                Fw_2_vals = np.where(denominator != 0, (blp_means - r2_means) / denominator * 100, np.nan)
                Fw_2_mean = np.nanmean(Fw_2_vals)

            results.append({
                grouping_choice: group,
                "deal_id": subset['Deal ID'].iloc[0],
                "grower": subset['Grower'].iloc[0],
                "Fw_mean": Fw_mean,
                "Fw_2_mean": Fw_2_mean,
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
                "Weathering Rate (Fw%)": round(r["Fw_mean"], 2)
            }
            for r in results
        ]).sort_values(by="Weathering Rate (Fw%)", ascending=False)

        st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

        def plot_distribution(r, ax):
            sns.histplot(r["bl_list"], bins=50, color="blue", label="BL", stat="density", alpha=0.5, ax=ax)
            sns.histplot(r["blp_list"], bins=50, color="purple", label="BLP", stat="density", alpha=0.5, ax=ax)
            sns.histplot(r["r1_list"], bins=50, color="green", label="R1", stat="density", alpha=0.5, ax=ax)

            sns.kdeplot(r["bl_list"], color="blue", lw=2.5, ax=ax)
            sns.kdeplot(r["blp_list"], color="purple", lw=2.5, ax=ax)
            sns.kdeplot(r["r1_list"], color="green", lw=2.5, ax=ax)

            if r["r2_list"] is not None:
                sns.histplot(r["r2_list"], bins=50, color="orange", label="R2", stat="density", alpha=0.5, ax=ax)
                sns.kdeplot(r["r2_list"], color="orange", lw=2.5, ax=ax)

            title = (
                f"{grouping_choice}: {r[grouping_choice]} | Deal ID: {r['deal_id']}\n"
                f"Fw (R1): {r['Fw_mean']:.2f}%"
            )
            if r.get("Fw_2_mean") is not None:
                title += f" | Fw_2 (R2): {r['Fw_2_mean']:.2f}%"
            title += (
                f"\nBL: {r['bl_count']}, BLP: {r['blp_count']}, "
                f"R1: {r['r1_count']}, R2: {r['r2_count']}"
            )
            ax.set_title(title)
            ax.set_xlabel("Ca + Mg (moles)")
            ax.set_ylabel("Density")
            ax.legend()

        #st.markdown("**-" * 65)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Positive Weathering Rates \U0001F33F")
            for r in positive_results:
                fig, ax = plt.subplots()
                plot_distribution(r, ax)
                st.pyplot(fig)

        with col2:
            st.subheader("Negative Weathering Rates ⛏️")
            for r in negative_results:
                fig, ax = plt.subplots()
                plot_distribution(r, ax)
                st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")
else:
    st.info("📄 Upload a CSV file to begin.")
