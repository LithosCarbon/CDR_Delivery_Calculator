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

        for element in ['Ni', 'Cr', 'LOI']:
            if element in treatments.columns:
                treatments[element] = pd.to_numeric(treatments[element], errors='coerce')

        treatments['Ca_moles'] = treatments['CaO_elemental'] / 40.078
        treatments['Mg_moles'] = treatments['MgO_elemental'] / 24.305
        treatments['Total_Ca_Mg_moles'] = treatments['Ca_moles'] + treatments['Mg_moles']
        data = treatments

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        if grouping_choice == "No Grouping (All Data)":
            all_growers = sorted(data['Grower'].dropna().unique())
            selected_growers = st.multiselect("Select growers to include:", options=all_growers, default=all_growers)
            data = data[data['Grower'].isin(selected_growers)]
            data['__all__'] = "All Samples"
            grouping_choice = '__all__'

            # First: make a copy of your filtered data
            filtered_data = data.copy()

            # Then: create a new "Grower, Deal ID" column
            filtered_data['Grower, Deal ID'] = filtered_data['Grower'] + ', ' + filtered_data['Deal ID'].astype(str)

            # Optional: move "Grower, Deal ID" to the first column
            cols = ['Grower, Deal ID'] + [col for col in filtered_data.columns if col not in ['Grower, Deal ID', 'Grower', 'Deal ID']]
            filtered_data = filtered_data[cols]

            # Now: offer it for download
            st.markdown("### 📥 Download Raw Filtered Data")
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

        # Allow download of selected growers subset
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data)


        st.markdown(f"✅ **{len(full_datasets)} unique groups** with complete datasets")
        st.markdown(f"👩‍🌾 Across **{data['Grower'].nunique()} unique growers**")

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

            bl_medians = np.nanmedian(bl_samples, axis=1)
            blp_medians = np.nanmedian(blp_samples, axis=1)
            r1_medians = np.nanmedian(r1_samples, axis=1)
            r2_medians = np.nanmedian(r2_samples, axis=1) if r2_samples is not None else None

            denominator = blp_medians - bl_medians
            Fw_vals = np.where(denominator != 0, (blp_medians - r1_medians) / denominator * 100, np.nan)
            Fw_median = np.nanmedian(Fw_vals)
            Fw_std = np.nanstd(Fw_vals)

            Fw_2_vals = None
            Fw_2_median = None
            Fw_2_std = None
            if r2_medians is not None:
                Fw_2_vals = np.where(denominator != 0, (blp_medians - r2_medians) / denominator * 100, np.nan)
                Fw_2_median = np.nanmedian(Fw_2_vals)
                Fw_2_std = np.nanstd(Fw_2_vals)

            results.append({
                grouping_choice: group,
                "deal_id": subset['Deal ID'].iloc[0],
                "grower": subset['Grower'].iloc[0],
                "Fw_median": Fw_median,
                "Fw_std": Fw_std,
                "Fw_2_median": Fw_2_median,
                "Fw_2_std": Fw_2_std,
                "bl_list": bl_medians,
                "blp_list": blp_medians,
                "r1_list": r1_medians,
                "r2_list": r2_medians,
                "bl_count": len(bl),
                "blp_count": len(blp),
                "r1_count": len(r1),
                "r2_count": len(r2) if r2 is not None else 0,
            })

        positive_results = sorted([r for r in results if r["Fw_median"] > 0], key=lambda x: -x["Fw_median"])
        negative_results = sorted([r for r in results if r["Fw_median"] <= 0], key=lambda x: x["Fw_median"])

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
                "Fw Median (%)": round(r["Fw_median"], 2),
                "Fw Std Dev": round(r["Fw_std"], 2),
                "Fw₂ Median (%)": round(r["Fw_2_median"], 2) if r["Fw_2_median"] is not None else None,
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


        def plot_distribution(r, ax):
            sns.histplot(r["bl_list"], bins=50, color="blue", label="Regional BL", stat="density", alpha=0.5, ax=ax)
            sns.histplot(r["blp_list"], bins=50, color="purple", label="BLP", stat="density", alpha=0.5, ax=ax)
            sns.histplot(r["r1_list"], bins=50, color="green", label="R1", stat="density", alpha=0.5, ax=ax)

            sns.kdeplot(r["bl_list"], color="blue", lw=2.5, ax=ax)
            sns.kdeplot(r["blp_list"], color="purple", lw=2.5, ax=ax)
            sns.kdeplot(r["r1_list"], color="green", lw=2.5, ax=ax)

            if r["r2_list"] is not None:
                sns.histplot(r["r2_list"], bins=50, color="orange", label="R2", stat="density", alpha=0.5, ax=ax)
                sns.kdeplot(r["r2_list"], color="orange", lw=2.5, ax=ax)

            title = (
                #f"{grouping_choice}: {r[grouping_choice]} | Deal ID: {r['deal_id']}\n"
                f"Fw (R1): {r['Fw_median']:.2f} ± {r['Fw_std']:.2f}%"
            )
            if r.get("Fw_2_median") is not None:
                title += f" | Fw_2 (R2): {r['Fw_2_median']:.2f} ± {r['Fw_2_std']:.2f}%%"
            title += (
                f"\nRegional BL: {r['bl_count']}, BLP: {r['blp_count']}, "
                f"R1: {r['r1_count']}, R2: {r['r2_count']}"
            )
            ax.set_title(title)
            ax.set_xlabel("Ca + Mg (moles)")
            ax.set_ylabel("Density")
            ax.legend()

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
