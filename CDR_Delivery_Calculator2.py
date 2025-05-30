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
st.title("Lithos Carbon tCDR Delivery Calculator (Median, MgCaNa)")

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

        #treatments = data[data['Sample Type'].isin(['BL', 'BLP', 'R1'])].copy()
        treatments = data[data['Sample Type'].isin(['BL','Regional BL', 'BLP', 'R1',"R2","BLC","BLPC","R1C","Recvd Wt."])].copy()
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
        treatments['Na_moles'] = treatments['Na2O_elemental'] / 22.989769  
        treatments['Total_Ca_Mg_Na_moles'] = (treatments['Ca_moles']*2 + treatments['Mg_moles']*2 + treatments['Na_moles']*1)*10
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

        # Checkbox to activate Regional BL usage
        use_regional_bl = st.checkbox("✅ Use Regional BL for all Fw calculations", value=False)

        st.markdown(f"✅ **{len(full_datasets)} unique groups** with complete datasets")
        st.markdown(f"👩‍🌾 Across **{data['Grower'].nunique()} unique growers**")

        # Extract and validate Regional BL data if requested
        if use_regional_bl:
            regional_bl_df = data[data['Sample Type'].str.contains("Regional BL", na=False, case=False)]
            if regional_bl_df.empty:
                st.error("Regional BL selected but no 'Regional BL' samples found in 'Sample Type'.")
                st.stop()

            regional_bl_values = pd.to_numeric(regional_bl_df['Total_Ca_Mg_Na_moles'], errors='coerce').dropna().values
            if len(regional_bl_values) == 0:
                st.error("'Regional BL' samples found, but all Ca+Mg values are missing or invalid.")
                st.stop()

        results = []
        n_bootstrap = 20000



        BD_change = st.number_input(
            "Bulk density change (as decimal)", 
            min_value=0.0, max_value=0.5, value=0.0, step=0.001, format="%.2f"
        )

        additive = st.number_input(
            "Correction for Ag application (mol/kg)", 
            min_value=0.0, max_value=0.1, value=0.0, step=0.0001, format="%.3f"
        )

        control_corr = st.number_input(
            "Correction for controls", 
            min_value=0.0, max_value=0.1, value=0.0, step=0.00001, format="%.4f"
        )

        for group in full_datasets:
            subset = data[data[grouping_choice] == group]
            #regional = data[data['Farm ID'] == 'Regional']
            #bl = pd.to_numeric(subset[subset['Sample Type'] == 'BL']['Total_Ca_Mg_moles'], errors='coerce').dropna().values
            bl = regional_bl_values if use_regional_bl else pd.to_numeric(subset[subset['Sample Type'] == 'BL']['Total_Ca_Mg_Na_moles'], errors='coerce').dropna().values
            blp = pd.to_numeric(subset[subset['Sample Type'] == 'BLP']['Total_Ca_Mg_Na_moles'], errors='coerce').dropna().values
            r1 = pd.to_numeric(subset[subset['Sample Type'] == 'R1']['Total_Ca_Mg_Na_moles'], errors='coerce').dropna().values
            r2 = pd.to_numeric(subset[subset['Sample Type'] == 'R2']['Total_Ca_Mg_Na_moles'], errors='coerce').dropna().values if 'R2' in subset['Sample Type'].values else None

            r1_corrected = ((r1-additive)*(1-BD_change))+control_corr

            if min(len(bl), len(blp), len(r1_corrected)) == 0:
                continue

            bl_samples = np.random.choice(bl, (n_bootstrap, len(bl)), replace=True)
            blp_samples = np.random.choice(blp, (n_bootstrap, len(blp)), replace=True)
            r1_samples = np.random.choice(r1, (n_bootstrap, len(r1)), replace=True)
            r2_samples = np.random.choice(r2, (n_bootstrap, len(r2)), replace=True) if r2 is not None and len(r2) > 0 else None

            r1_samples_corrected = np.random.choice(r1_corrected, (n_bootstrap, len(r1_corrected)), replace=True)


            bl_medians = np.nanmedian(bl_samples, axis=1)
            blp_medians = np.nanmedian(blp_samples, axis=1)
            r1_medians = np.nanmedian(r1_samples_corrected, axis=1)
            r2_medians = np.nanmedian(r2_samples, axis=1) if r2_samples is not None else None

            denominator = blp_medians - bl_medians
            Fw_vals = np.where(denominator != 0, (blp_medians - r1_medians) / denominator * 100, np.nan)
            Fw_median = np.nanmedian(Fw_vals)

            Fw_2_vals = None
            Fw_2_median = None
            if r2_medians is not None:
                Fw_2_vals = np.where(denominator != 0, (blp_medians - r2_medians) / denominator * 100, np.nan)
                Fw_2_median = np.nanmedian(Fw_2_vals)

            bl_median = np.nanmedian(bl_samples)
            blp_median = np.nanmedian(blp_samples)
            r1_median = np.nanmedian(r1_samples_corrected)
            r2_median = np.nanmedian(r2_medians) if r2_samples is not None else None


            results.append({
                grouping_choice: group,
                "deal_id": subset['Deal ID'].iloc[0],
                "grower": subset['Grower'].iloc[0],
                "Fw_median": Fw_median,
                "Fw_2_median": Fw_2_median,
                "bl_median": bl_median,
                "blp_median": blp_median,
                "r1_median": r1_median,
                "r2_median": r2_median,
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

                "bl_median": r["bl_median"],
                "blp_median": r["blp_median"],
                "r1_median": r["r1_median"],
                "r2_median": r["r2_median"],
                "Fw Median (%)": round(r["Fw_median"], 2),
                "Fw2 Median (%)": round(r["Fw_2_median"], 2) if r["Fw_2_median"] is not None else None,
                
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
                f"Fw (R1): {r['Fw_median']:.2f}"
            )
            if r.get("Fw_2_median") is not None:
                title += f" | Fw_2 (R2): {r['Fw_2_median']:.2f} "
            title += (
                f"\nBL: {r['bl_count']}, BLP: {r['blp_count']}, "
                f"R1: {r['r1_count']}, R2: {r['r2_count']}"
            )
            ax.set_title(title)
            ax.set_xlabel("Ca + Mg + Na (moles/kg)")
            ax.set_ylabel("Density")
            ax.legend()

        col1, col2 = st.columns(2)

        show_plots = st.checkbox("📊 Show distribution plots", value=False)

        if show_plots:
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


    st.markdown("---")
    st.markdown("## 🔁 Control Bootstrap: R1C vs BLC/BLPC Comparison")

    # Fill missing IDs
    data["Farm ID"] = data["Farm ID"].fillna("Unknown Farm")
    data["Field ID"] = data["Field ID"].fillna("Unknown Field")
    data["Grower, Deal ID"] = data["Grower"] + ", " + data["Deal ID"].astype(str)

    # Grouping dropdown and BLPC exclusion toggle
    grouping_choice = st.selectbox("Select grouping for control analysis:", ["Grower, Deal ID", "Farm ID", "Field ID"])
    exclude_blpc = st.checkbox("🧮 Exclude BLPC from control calculation", value=False)

    # Determine grouping columns
    if grouping_choice == "Farm ID":
        group_cols = ["Grower, Deal ID", "Farm ID"]
    elif grouping_choice == "Field ID":
        group_cols = ["Grower, Deal ID", "Farm ID", "Field ID"]
    else:
        group_cols = ["Grower, Deal ID"]

    # Bootstrapping setup
    n_bootstrap = 10000
    control_results = []

    # Group and compute
    for name, subset in data.groupby(group_cols):
        types_present = subset["Sample Type"].unique()
        if "R1C" not in types_present:
            continue
        if exclude_blpc and "BLC" not in types_present:
            continue
        if not exclude_blpc and not any(x in types_present for x in ["BLC", "BLPC"]):
            continue

        blc = pd.to_numeric(subset[subset["Sample Type"] == "BLC"]["Total_Ca_Mg_Na_moles"], errors="coerce").dropna().values
        blpc = pd.to_numeric(subset[subset["Sample Type"] == "BLPC"]["Total_Ca_Mg_Na_moles"], errors="coerce").dropna().values
        r1c = pd.to_numeric(subset[subset["Sample Type"] == "R1C"]["Total_Ca_Mg_Na_moles"], errors="coerce").dropna().values

        if len(r1c) == 0:
            continue
        if exclude_blpc and len(blc) == 0:
            continue
        if not exclude_blpc and (len(blc) == 0 and len(blpc) == 0):
            continue

        # Combine BLC + BLPC or BLC only
        if exclude_blpc:
            combined_input = blc
        else:
            combined_input = np.concatenate([blc, blpc])

        combined_samples = np.random.choice(combined_input, (n_bootstrap, len(combined_input)), replace=True)
        combined_medians = np.nanmedian(combined_samples, axis=1)

        r1c_samples = np.random.choice(r1c, (n_bootstrap, len(r1c)), replace=True)
        r1c_medians = np.nanmedian(r1c_samples, axis=1)

        blc_median = np.nanmedian(blc) if len(blc) > 0 else None
        blpc_median = np.nanmedian(blpc) if len(blpc) > 0 else None
        Delta_R1 = np.nanmedian(r1c_medians - combined_medians)

        # Create readable group label
        group_label = " | ".join(name if isinstance(name, tuple) else [name])

        control_results.append({
            "Grower, Deal ID": name[0] if isinstance(name, tuple) else name,
            "Farm ID": name[1] if len(name) > 1 else None,
            "Field ID": name[2] if len(name) > 2 else None,
            "BLC_count": len(blc),
            "BLPC_count": len(blpc),
            "R1C_count": len(r1c),
            "BLC_Median": blc_median,
            "BLPC_Median": blpc_median,
            "Combined_Median": np.nanmedian(combined_medians),
            "R1C_Median": np.nanmedian(r1c_medians),
            "Delta_R1": Delta_R1
        })

    # Show table
    if control_results:
        st.markdown(f"**Method**: {'BLC only' if exclude_blpc else 'BLC + BLPC combined'}")
        control_df = pd.DataFrame(control_results)
        st.dataframe(control_df, use_container_width=True)

        control_csv = control_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Control Summary CSV",
            data=control_csv,
            file_name=f"control_summary_{grouping_choice.replace(' ', '_')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No valid groups found with R1C and eligible control samples.")


else:
    st.info("📄 Upload a CSV file to begin.")
