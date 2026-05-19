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
    history       = bundle['history']
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
c1.metric("📅 Forecast Horizon",        f"{n_months} months")
c2.metric("📌 Current Improvement Rate", f"{cur_imp:.1f}%")
c3.metric("🔮 Month +1 Forecast",        f"{mo1_imp:.1f}%",  delta=f"{mo1_imp - cur_imp:+.1f}%")
c4.metric(f"🏁 Month +{n_months} Forecast", f"{final_imp:.1f}%", delta=f"{delta_imp:+.1f}%")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Charts", "📋 Forecast Table", "📈 Improvement Focus"])

# ── TAB 1: Charts ─────────────────────────────────────────────────────────────
with tab1:
    if not selected_cols:
        st.warning("Please select at least one indicator from the sidebar.")
    else:
        last_no = bundle['last_month_no']
        n_cols  = 2
        n_rows  = (len(selected_cols) + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows))

        # Normalize axes to always be a flat list
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

    st.subheader("Month-by-Month Breakdown")
    imp_rows = []
    prev = history[col].iloc[-1]
    for lbl, val in zip(future_labels, fc_y):
        chg   = val - prev
        arrow = '↑' if chg > 0.01 else ('↓' if chg < -0.01 else '→')
        imp_rows.append({
            'Month': lbl,
            '% Predicted to Improve': val,
            'Change': round(chg, 2),
            'Trend':  arrow
        })
        prev = val

    st.dataframe(pd.DataFrame(imp_rows), use_container_width=True, hide_index=True)
