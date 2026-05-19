import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Child Health Trend Forecasting",
    page_icon="📈",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────────────────────
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']

COLUMN_LABELS = {
    'Avg_BMI':                 'Average BMI',
    'Avg_Weight_kg':           'Average Weight (kg)',
    'Avg_Height_cm':           'Average Height (cm)',
    'Pct_Dewormed':            '% Dewormed',
    'Pct_Vitamins_Intake':     '% Vitamins Intake',
    'Pct_Immunization':        '% Immunization',
    'Pct_Vaccination':         '% Vaccination',
    'Pct_Improved_Next_Month': '% Predicted to Improve',
}

COLORS = {
    'Avg_BMI':                 '#4C9BE8',
    'Avg_Weight_kg':           '#E88A4C',
    'Avg_Height_cm':           '#4CE8A0',
    'Pct_Dewormed':            '#A04CE8',
    'Pct_Vitamins_Intake':     '#E84C6A',
    'Pct_Immunization':        '#4CE8D8',
    'Pct_Vaccination':         '#E8D44C',
    'Pct_Improved_Next_Month': '#E84C4C',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def next_month_labels(last_label, n):
    m, y   = last_label.split('-')
    mi, yr = MONTH_NAMES.index(m), int(y)
    labels = []
    for _ in range(n):
        mi += 1
        if mi >= 12:
            mi, yr = 0, yr + 1
        labels.append(f"{MONTH_NAMES[mi]}-{yr}")
    return labels

@st.cache_resource
def load_bundle(path='arima_models.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

def run_forecast(bundle, n):
    models        = bundle['models']
    last_label    = bundle['last_month_label']
    last_no       = bundle['last_month_no']
    targets       = bundle['targets']
    future_labels = next_month_labels(last_label, n)
    result = pd.DataFrame({
        'Month_No':    range(last_no + 1, last_no + n + 1),
        'Month_Label': future_labels
    })
    for col in targets:
        fc = models[col].predict(n_periods=n)
        result[col] = np.round(fc, 2)
    return result, future_labels


# ── Month-by-Month Recommendation Engine ─────────────────────────────────────
def get_status_and_recommendation(month_no, month_label, pct_improve,
                                   bmi, weight, height,
                                   dewormed, vitamins, immunize, vacc,
                                   prev_improve):
    """
    Generate a status label and detailed recommendation for a single month
    based on its forecasted indicator values using many if-else conditions.
    """

    change = pct_improve - prev_improve

    # ── Determine Status ──────────────────────────────────────────────────────
    if pct_improve >= 95.0:
        if change >= 1.0:
            status = "🟢 Excellent ↑"
        elif change >= 0:
            status = "🟢 Excellent →"
        else:
            status = "🟢 Excellent ↓"
    elif pct_improve >= 85.0:
        if change >= 1.0:
            status = "🟢 Good ↑"
        elif change >= 0:
            status = "🟢 Good →"
        else:
            status = "🟡 Good ↓"
    elif pct_improve >= 75.0:
        if change >= 1.0:
            status = "🟡 Moderate ↑"
        elif change >= 0:
            status = "🟡 Moderate →"
        else:
            status = "🟡 Moderate ↓"
    elif pct_improve >= 60.0:
        if change >= 0:
            status = "🟠 Below Target ↑"
        else:
            status = "🟠 Below Target ↓"
    elif pct_improve >= 45.0:
        if change >= 0:
            status = "🔴 Poor ↑"
        else:
            status = "🔴 Poor ↓"
    else:
        status = "🔴 Critical ↓"

    # ── Build Recommendation ──────────────────────────────────────────────────
    rec_parts = []

    # 1. Improvement rate level
    if pct_improve >= 95.0:
        rec_parts.append("Outstanding improvement rate. Maintain all current health programs.")
    elif pct_improve >= 90.0:
        rec_parts.append("Excellent improvement rate. Continue programs and document best practices.")
    elif pct_improve >= 85.0:
        rec_parts.append("Good improvement rate. Identify remaining gaps and target lagging subgroups.")
    elif pct_improve >= 80.0:
        rec_parts.append("Improvement rate is good but more effort is needed to reach higher targets.")
    elif pct_improve >= 75.0:
        rec_parts.append("Moderate improvement. Reinforce nutrition counseling and feeding programs.")
    elif pct_improve >= 70.0:
        rec_parts.append("Improvement is below expectation. Review program effectiveness and coverage.")
    elif pct_improve >= 65.0:
        rec_parts.append("Low improvement rate. Increase health worker visits and caregiver training.")
    elif pct_improve >= 60.0:
        rec_parts.append("Improvement rate is concerning. Launch targeted interventions immediately.")
    elif pct_improve >= 50.0:
        rec_parts.append("Critically low improvement. Mobilize multi-sector response with LGU and DOH.")
    else:
        rec_parts.append("Improvement rate is at emergency level. Declare nutrition crisis and activate all resources.")

    # 2. Trend-based recommendation
    if change >= 3.0:
        rec_parts.append("Strong upward trend — programs are highly effective this month.")
    elif change >= 1.5:
        rec_parts.append("Positive trend — keep reinforcing current strategies.")
    elif change >= 0.5:
        rec_parts.append("Slight improvement — maintain consistency in program delivery.")
    elif change >= -0.5:
        rec_parts.append("Flat trend — assess if interventions need adjustment.")
    elif change >= -1.5:
        rec_parts.append("Slight decline detected — investigate causes and strengthen outreach.")
    elif change >= -3.0:
        rec_parts.append("Declining trend — review and intensify all health interventions immediately.")
    else:
        rec_parts.append("Sharp decline — urgent review required; convene emergency health meeting.")

    # 3. BMI-based recommendation
    if bmi < 12.0:
        rec_parts.append("BMI critically low: provide therapeutic feeding with RUTF and refer to pediatrician.")
    elif bmi < 13.5:
        rec_parts.append("BMI underweight: increase caloric intake with energy-dense foods (banana, avocado, egg).")
    elif bmi < 14.5:
        rec_parts.append("BMI slightly low: ensure 5–6 balanced meals daily with protein and healthy fats.")
    elif bmi <= 17.0:
        rec_parts.append("BMI normal: maintain diverse diet with rice, fish, vegetables, and fruits.")
    elif bmi <= 19.0:
        rec_parts.append("BMI slightly high: reduce sugary snacks and encourage 60 min outdoor activity daily.")
    else:
        rec_parts.append("BMI obese: strictly limit junk food; consult pediatric nutritionist for meal plan.")

    # 4. Weight-based recommendation
    if weight < 10.0:
        rec_parts.append("Weight critically low: coordinate emergency supplemental feeding with barangay health center.")
    elif weight < 12.0:
        rec_parts.append("Weight below average: enrich meals with iron-rich foods (liver, kangkong, fortified cereals).")
    elif weight <= 16.0:
        rec_parts.append("Weight normal: maintain feeding schedules and food diversity programs.")
    else:
        rec_parts.append("Weight above average: audit feeding program caloric content; reduce fried and processed foods.")

    # 5. Height-based recommendation
    if height < 82.0:
        rec_parts.append("Height stunted: increase calcium (milk, tofu, small fish) and ensure vitamin D through sunlight.")
    elif height < 87.0:
        rec_parts.append("Height slightly low: prioritize zinc and calcium supplementation; serve fish and legumes regularly.")
    elif height <= 92.0:
        rec_parts.append("Height normal: continue calcium-rich foods and physical activity for healthy bone development.")
    else:
        rec_parts.append("Height above average: positive growth indicator — maintain current nutritional support.")

    # 6. Deworming-based recommendation
    if dewormed < 50.0:
        rec_parts.append("Deworming critically low: launch emergency deworming drive with albendazole distribution.")
    elif dewormed < 65.0:
        rec_parts.append("Deworming low: schedule community deworming every 6 months; educate on hygiene.")
    elif dewormed < 80.0:
        rec_parts.append("Deworming moderate: increase outreach to remote areas; use barangay health centers as hubs.")
    elif dewormed < 90.0:
        rec_parts.append("Deworming good: sustain schedules and reach remaining uncovered children via home visits.")
    else:
        rec_parts.append("Deworming excellent: maintain program and document approach as best practice.")

    # 7. Vitamins-based recommendation
    if vitamins < 50.0:
        rec_parts.append("Vitamins critically low: immediately distribute Vitamin A capsules and iron syrup; integrate into barangay health days.")
    elif vitamins < 65.0:
        rec_parts.append("Vitamins low: prioritize Vitamin A, iron, and iodine; train BHWs to identify deficient children.")
    elif vitamins < 80.0:
        rec_parts.append("Vitamins moderate: strengthen micronutrient programs; promote malunggay, squash, and papaya intake.")
    elif vitamins < 90.0:
        rec_parts.append("Vitamins good: continue promoting food diversification alongside supplementation.")
    else:
        rec_parts.append("Vitamins excellent: maintain distribution and educate caregivers on vitamin-preserving food preparation.")

    # 8. Immunization-based recommendation
    if immunize < 50.0:
        rec_parts.append("Immunization critically low: deploy mobile vaccination teams for emergency catch-up campaign.")
    elif immunize < 65.0:
        rec_parts.append("Immunization low: schedule regular immunization days; conduct house-to-house follow-up.")
    elif immunize < 80.0:
        rec_parts.append("Immunization moderate: identify missed-dose children; use SMS reminders for scheduled vaccinations.")
    elif immunize < 95.0:
        rec_parts.append("Immunization good: target remaining unimmunized children to reach 95% herd immunity threshold.")
    else:
        rec_parts.append("Immunization excellent: sustain program and share best practices with neighboring communities.")

    # 9. Vaccination-based recommendation
    if vacc < 50.0:
        rec_parts.append("Vaccination critically low: deploy outreach teams; partner with LGU for BCG, DPT, OPV, MMR, HepB coverage.")
    elif vacc < 65.0:
        rec_parts.append("Vaccination low: address vaccine hesitancy through education; set up satellite vaccination posts.")
    elif vacc < 80.0:
        rec_parts.append("Vaccination moderate: track defaulters; integrate vaccination with feeding program visits.")
    elif vacc < 95.0:
        rec_parts.append("Vaccination good: maintain schedule and ensure cold chain integrity for vaccine storage.")
    else:
        rec_parts.append("Vaccination excellent: exceptional coverage — replicate strategies in other barangays.")

    # 10. Combined deworming + vitamins
    if dewormed < 65.0 and vitamins < 65.0:
        rec_parts.append("COMBINED RISK: Both deworming and vitamins are low — launch integrated health day combining both interventions.")
    elif dewormed < 80.0 and vitamins < 80.0:
        rec_parts.append("Combine deworming and vitamin distribution in one health day for efficiency and higher compliance.")

    # 11. Combined immunization + vaccination
    if immunize < 65.0 and vacc < 65.0:
        rec_parts.append("COMBINED RISK: Both immunization and vaccination are critically low — request regional DOH mobile unit support.")
    elif immunize < 80.0 and vacc < 80.0:
        rec_parts.append("Combine immunization and vaccination schedules to reduce missed appointments and caregiver burden.")

    # 12. All excellent
    if dewormed >= 90.0 and vitamins >= 90.0 and immunize >= 90.0 and vacc >= 90.0:
        rec_parts.append("All interventions at 90%+ — model barangay performance. Submit as case study for regional replication.")

    # Join all parts
    recommendation = " | ".join(rec_parts)
    return status, recommendation


# ── Load model ────────────────────────────────────────────────────────────────
try:
    bundle = load_bundle('arima_models.pkl')
except FileNotFoundError:
    st.error("arima_models.pkl not found. Please make sure it is in the same folder as this app.")
    st.stop()

history    = bundle['history']
last_label = bundle['last_month_label']
targets    = bundle['targets']

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📈 Child Health Trend Forecasting")
st.markdown(
    "**Model:** ARIMA (Auto-selected orders) &nbsp;|&nbsp; "
    "**Age Group:** 0–6 years &nbsp;|&nbsp; "
    "**Historical Data:** Jan 2020 – Dec 2025"
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Forecast Settings")
    n_months = st.slider("Months to Forecast", min_value=1, max_value=60, value=12, step=1)
    future_start = next_month_labels(last_label, 1)[0]
    future_end   = next_month_labels(last_label, n_months)[-1]
    st.caption(f"Forecasting: **{future_start}** → **{future_end}**")
    st.divider()
    selected_cols = st.multiselect(
        "Indicators to Display in Charts",
        options=targets,
        default=targets,
        format_func=lambda x: COLUMN_LABELS.get(x, x)
    )
    st.divider()
    st.info(f"📅 Last historical month: **{last_label}**\n\n📊 Training rows: **{len(history)}**")

# ── Run forecast ──────────────────────────────────────────────────────────────
forecast_df, future_labels = run_forecast(bundle, n_months)

# ── Summary metric cards ──────────────────────────────────────────────────────
imp_col   = 'Pct_Improved_Next_Month'
cur_imp   = history[imp_col].iloc[-1]
mo1_imp   = forecast_df[imp_col].iloc[0]
final_imp = forecast_df[imp_col].iloc[-1]
delta_imp = final_imp - cur_imp

c1, c2, c3, c4 = st.columns(4)
c1.metric("📅 Forecast Horizon",            f"{n_months} months")
c2.metric("📌 Current Improvement Rate",    f"{cur_imp:.1f}%")
c3.metric("🔮 Month +1 Forecast",           f"{mo1_imp:.1f}%", delta=f"{mo1_imp - cur_imp:+.1f}%")
c4.metric(f"🏁 Month +{n_months} Forecast", f"{final_imp:.1f}%", delta=f"{delta_imp:+.1f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Charts",
    "📋 Forecast Table",
    "📈 Improvement Focus",
    "💡 Nutrition Recommendations"
])

# ── TAB 1: Charts ─────────────────────────────────────────────────────────────
with tab1:
    if not selected_cols:
        st.warning("Please select at least one indicator from the sidebar.")
    else:
        last_no = bundle['last_month_no']
        n_cols  = 2
        n_rows  = (len(selected_cols) + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))
        if n_rows == 1 and len(selected_cols) == 1:
            axes_flat = [axes]
        elif n_rows == 1:
            axes_flat = list(axes)
        else:
            axes_flat = list(axes.flatten())

        for i, col in enumerate(selected_cols):
            ax    = axes_flat[i]
            color = COLORS.get(col, '#4C9BE8')
            std   = history[col].std() * 0.3
            fc_x  = forecast_df['Month_No']
            fc_y  = forecast_df[col]

            ax.plot(history['Month_No'], history[col],
                    color=color, linewidth=2, label='Historical')
            ax.plot(fc_x, fc_y,
                    color='tomato', linestyle='--', linewidth=2,
                    label=f'Forecast ({n_months}mo)')
            ax.fill_between(fc_x, fc_y - std, fc_y + std,
                            color='tomato', alpha=0.12, label='Confidence')
            ax.axvline(x=last_no, color='gray', linestyle=':', alpha=0.6)
            ax.set_title(COLUMN_LABELS.get(col, col), fontsize=11, fontweight='bold')
            ax.set_xlabel('Month No.')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25)

        for j in range(len(selected_cols), len(axes_flat)):
            axes_flat[j].set_visible(False)

        plt.suptitle(
            f'ARIMA Forecast — Next {n_months} Months (Age 0–6)',
            fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ── TAB 2: Forecast Table ─────────────────────────────────────────────────────
with tab2:
    display_df = forecast_df[['Month_Label'] + targets].copy()
    display_df.columns = ['Month'] + [COLUMN_LABELS.get(c, c) for c in targets]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=csv,
        file_name=f"forecast_{n_months}months.csv",
        mime="text/csv"
    )

# ── TAB 3: Improvement Focus ──────────────────────────────────────────────────
with tab3:
    col  = 'Pct_Improved_Next_Month'
    fc_x = forecast_df['Month_No']
    fc_y = forecast_df[col]
    std  = history[col].std() * 0.3
    last_no = bundle['last_month_no']

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(history['Month_No'], history[col],
             color='steelblue', linewidth=2.2, label='Historical')
    ax2.plot(fc_x, fc_y,
             color='tomato', linestyle='--', linewidth=2.2,
             label=f'Forecast ({n_months} months)')
    ax2.fill_between(fc_x, fc_y - std, fc_y + std,
                     color='tomato', alpha=0.15, label='Confidence Band')
    ax2.axvline(x=last_no, color='gray', linestyle=':', alpha=0.7, label='Forecast Start')
    ax2.annotate(
        f"+1mo\n{fc_y.iloc[0]:.1f}%",
        xy=(fc_x.iloc[0], fc_y.iloc[0]),
        xytext=(fc_x.iloc[0] + 0.8, fc_y.iloc[0] + 1.5),
        fontsize=9, color='tomato',
        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2)
    )
    ax2.annotate(
        f"+{n_months}mo\n{fc_y.iloc[-1]:.1f}%",
        xy=(fc_x.iloc[-1], fc_y.iloc[-1]),
        xytext=(fc_x.iloc[-1] - 5, fc_y.iloc[-1] + 1.5),
        fontsize=9, color='tomato',
        arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2)
    )
    ax2.set_title('% Children Predicted to Improve — Historical + Forecast',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Month No.')
    ax2.set_ylabel('% Improved')
    ax2.legend()
    ax2.grid(True, alpha=0.25)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ── TAB 4: Nutrition Recommendations ─────────────────────────────────────────
with tab4:
    st.subheader("💡 Month-by-Month Nutrition Recommendations")
    st.markdown(
        "Each row shows the forecasted **improvement rate**, its **status**, "
        "and a **tailored recommendation** based on all health indicators for that month."
    )

    # Build month-by-month recommendation table
    rows = []
    prev_improve = history['Pct_Improved_Next_Month'].iloc[-1]

    for _, row in forecast_df.iterrows():
        month_no    = int(row['Month_No'])
        month_label = row['Month_Label']
        pct_improve = row['Pct_Improved_Next_Month']
        bmi         = row['Avg_BMI']
        weight      = row['Avg_Weight_kg']
        height      = row['Avg_Height_cm']
        dewormed    = row['Pct_Dewormed']
        vitamins    = row['Pct_Vitamins_Intake']
        immunize    = row['Pct_Immunization']
        vacc        = row['Pct_Vaccination']

        status, recommendation = get_status_and_recommendation(
            month_no, month_label, pct_improve,
            bmi, weight, height,
            dewormed, vitamins, immunize, vacc,
            prev_improve
        )

        rows.append({
            'Month No.':        month_no,
            'Month':            month_label,
            '% Improvement':    pct_improve,
            'Status':           status,
            'Recommendation':   recommendation
        })

        prev_improve = pct_improve

    rec_df = pd.DataFrame(rows)

    # Filter by status
    status_options = ['All'] + sorted(rec_df['Status'].unique().tolist())
    status_filter  = st.selectbox("Filter by Status", options=status_options, index=0)

    if status_filter != 'All':
        display_rec = rec_df[rec_df['Status'] == status_filter]
    else:
        display_rec = rec_df

    st.dataframe(
        display_rec,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Month No.':      st.column_config.NumberColumn('Month No.', width='small'),
            'Month':          st.column_config.TextColumn('Month', width='small'),
            '% Improvement':  st.column_config.NumberColumn('% Improvement', format="%.2f%%", width='medium'),
            'Status':         st.column_config.TextColumn('Status', width='medium'),
            'Recommendation': st.column_config.TextColumn('Recommendation', width='large'),
        }
    )

    # Download
    csv_rec = rec_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Recommendations as CSV",
        data=csv_rec,
        file_name=f"recommendations_{n_months}months.csv",
        mime="text/csv"
    )
