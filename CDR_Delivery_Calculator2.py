import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

def check_password():
    def password_entered():
        if st.session_state["password"] == "yourpassword123":  # <-- change this!
            st.session_state["authenticated"] = True
            del st.session_state["password"]
        else:
            st.session_state["authenticated"] = False

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        st.title("🔐 Password Protected")
        st.text_input("Enter password:", type="password", on_change=password_entered, key="password")
        st.stop()

check_password()






# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit setup
st.set_page_config(layout="wide")
st.title("Lithos tCDR Delivery Calculator")

uploaded_file = st.file_uploader("Upload your Geochem CSV", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("CSV uploaded!")
        st.write(data.head())

        # Step 1: Cleanup
        if 'Grower, Deal ID' not in data.columns:
            st.error("The uploaded CSV is missing the 'Grower, Deal ID' column.")
            st.stop()

        data[['Grower', 'Deal ID']] = data['Grower, Deal ID'].str.split(', ', expand=True)
        data.drop(columns=['Grower, Deal ID'], inplace=True)

        treatments = data[data['Sample Type'].isin(['BL', 'BLP', 'R1','R2'])].copy()
        treatments['Field ID'] = treatments['Field ID'].fillna('')
        treatments = treatments[(treatments['CaO'].notnull()) & (treatments['MgO'].notnull())]

        # Oxide to elemental conversions
        conversion_factors = {
            'SiO2': 0.4674, 'Al2O3': 0.5293, 'Fe2O3': 0.6994, 'CaO': 0.7147,
            'MgO': 0.6030, 'Na2O': 0.7419, 'K2O': 0.8301, 'Cr2O3': 0.6842,
            'TiO2': 0.5995, 'MnO': 0.7745, 'P2O5': 0.4364, 'SrO': 0.8456, 'BaO': 0.8957
        }

        for oxide, factor in conversion_factors.items():
            if oxide in treatments.columns:
                treatments[oxide] = pd.to_numeric(treatments[oxide], errors='coerce')
                treatments[oxide + '_elemental'] = treatments[oxide] * factor

        for element in ['Ni', 'Cr','LOI']:
            if element in treatments.columns:
                treatments[element] = pd.to_numeric(treatments[element], errors='coerce')

        # Moles calculation
        treatments['Ca_moles'] = treatments['CaO_elemental'] / 40.078
        treatments['Mg_moles'] = treatments['MgO_elemental'] / 24.305
        treatments['Total_Ca_Mg_moles'] = treatments['Ca_moles'] + treatments['Mg_moles']
        data = treatments

        # Identify complete datasets
        full_datasets = []
        for field in data['Field ID'].unique():
            subset = data[data['Field ID'] == field]
            types_present = subset['Sample Type'].unique()
            if all(x in types_present for x in ['BL', 'BLP', 'R1']):
                full_datasets.append(field)

        st.markdown(f"✅ **{len(full_datasets)} unique fields** with complete datasets")
        st.markdown(f"👩‍🌾 Across **{data['Grower'].nunique()} unique growers**")


         # ... (keep your code up to this point unchanged)

        # Step 2: Bootstrapping
        results = []
        n_bootstrap = 100000

        for field in full_datasets:
            subset = data[data['Field ID'] == field]
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

            results.append({
                "field": field,
                "deal_id": subset['Deal ID'].iloc[0],
                "grower": subset['Grower'].iloc[0],
                "Fw_mean": Fw_mean,
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

        st.markdown(f"🌱 **{len(positive_results)} fields** with positive weathering rates")
        st.markdown(f"🪨 **{len(negative_results)} fields** with negative or zero weathering rates")


        # Step 3: Summary Table
        summary_df = pd.DataFrame([
            {
                "Deal ID": r["deal_id"],
                "Field ID": r["field"],
                "Grower": r["grower"],
                "Weathering Rate (Fw%)": round(r["Fw_mean"], 2)
            }
            for r in results
        ]).sort_values(by="Weathering Rate (Fw%)", ascending=False)

        st.dataframe(summary_df.reset_index(drop=True), use_container_width=True)

        # Step 5: Plotting function
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

            ax.set_title(
                f"Farm: {r['field']} | Deal ID: {r['deal_id']}\n"
                f"Fw: {r['Fw_mean']:.2f}%\n"
                f"BL: {r['bl_count']}, BLP: {r['blp_count']}, R1: {r['r1_count']}, R2: {r['r2_count']}"
            )
            ax.set_xlabel("Ca + Mg (moles)")
            ax.set_ylabel("Density")
            ax.legend()









        st.markdown("**---------------------------------------------------------------------------------------------------------------------------------**")
        st.markdown("**---------------------------------------------------------------------------------------------------------------------------------**")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Positive Weathering Rates 🌿")
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
