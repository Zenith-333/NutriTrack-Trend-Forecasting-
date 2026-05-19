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


# ── Nutrition Recommendation Engine ──────────────────────────────────────────
def get_recommendations(forecast_df):
    """
    Generate detailed nutrition & health recommendations based on
    forecasted 12-month average values using many if-else conditions.
    """
    recs = []

    # Compute 12-month averages and trends
    avg_bmi        = forecast_df['Avg_BMI'].mean()
    avg_weight     = forecast_df['Avg_Weight_kg'].mean()
    avg_height     = forecast_df['Avg_Height_cm'].mean()
    avg_dewormed   = forecast_df['Pct_Dewormed'].mean()
    avg_vitamins   = forecast_df['Pct_Vitamins_Intake'].mean()
    avg_immunize   = forecast_df['Pct_Immunization'].mean()
    avg_vacc       = forecast_df['Pct_Vaccination'].mean()
    avg_improve    = forecast_df['Pct_Improved_Next_Month'].mean()

    trend_bmi      = forecast_df['Avg_BMI'].iloc[-1] - forecast_df['Avg_BMI'].iloc[0]
    trend_weight   = forecast_df['Avg_Weight_kg'].iloc[-1] - forecast_df['Avg_Weight_kg'].iloc[0]
    trend_improve  = forecast_df['Pct_Improved_Next_Month'].iloc[-1] - forecast_df['Pct_Improved_Next_Month'].iloc[0]
    trend_dewormed = forecast_df['Pct_Dewormed'].iloc[-1] - forecast_df['Pct_Dewormed'].iloc[0]
    trend_vitamins = forecast_df['Pct_Vitamins_Intake'].iloc[-1] - forecast_df['Pct_Vitamins_Intake'].iloc[0]
    trend_immunize = forecast_df['Pct_Immunization'].iloc[-1] - forecast_df['Pct_Immunization'].iloc[0]
    trend_vacc     = forecast_df['Pct_Vaccination'].iloc[-1] - forecast_df['Pct_Vaccination'].iloc[0]

    # ── 1. BMI-based Nutrition Recommendations ────────────────────────────────
    if avg_bmi < 12.0:
        recs.append({
            'category': '🔴 BMI — Severely Underweight',
            'priority': 'CRITICAL',
            'detail': (
                "The forecasted average BMI is critically low (< 12.0). "
                "Immediate intervention required. Provide therapeutic feeding with high-energy, "
                "high-protein foods such as peanut butter, milk, eggs, and legumes. "
                "Refer to a nutritionist or pediatrician immediately. "
                "Implement Ready-to-Use Therapeutic Food (RUTF) if available."
            )
        })
    elif avg_bmi < 13.5:
        recs.append({
            'category': '🔴 BMI — Underweight',
            'priority': 'HIGH',
            'detail': (
                "The forecasted average BMI is below normal (12.0–13.5). "
                "Children are at risk of undernutrition. "
                "Increase caloric intake through energy-dense foods: sweet potato, banana, avocado, and full-fat dairy. "
                "Provide 5–6 small meals per day instead of 3 large meals. "
                "Consider supplementary feeding programs for the barangay."
            )
        })
    elif avg_bmi < 14.5:
        recs.append({
            'category': '🟡 BMI — Slightly Below Normal',
            'priority': 'MODERATE',
            'detail': (
                "The forecasted average BMI is slightly below the healthy range (13.5–14.5). "
                "Encourage balanced meals with adequate protein and healthy fats. "
                "Include fish, chicken, mongo, and vegetables in daily diet. "
                "Monitor weight monthly and track growth using WHO growth charts."
            )
        })
    elif avg_bmi <= 17.0:
        recs.append({
            'category': '🟢 BMI — Normal',
            'priority': 'LOW',
            'detail': (
                "The forecasted average BMI is within the healthy range (14.5–17.0). "
                "Maintain current dietary practices. "
                "Continue serving diverse meals with rice, fish, vegetables, and fruits. "
                "Ensure children drink 6–8 glasses of water daily."
            )
        })
    elif avg_bmi <= 19.0:
        recs.append({
            'category': '🟡 BMI — Slightly Overweight',
            'priority': 'MODERATE',
            'detail': (
                "The forecasted average BMI is slightly above normal (17.0–19.0). "
                "Reduce intake of sugary snacks, processed foods, and sweetened beverages. "
                "Encourage outdoor physical activities for at least 60 minutes daily. "
                "Serve more vegetables and fruits; reduce rice and fried food portions."
            )
        })
    else:
        recs.append({
            'category': '🔴 BMI — Obese / Severely Overweight',
            'priority': 'HIGH',
            'detail': (
                "The forecasted average BMI exceeds 19.0, indicating risk of childhood obesity. "
                "Strictly limit junk food, softdrinks, and high-sugar snacks. "
                "Introduce structured physical activity programs in school and community. "
                "Consult a pediatric nutritionist for a personalized meal plan."
            )
        })

    # ── 2. BMI Trend ──────────────────────────────────────────────────────────
    if trend_bmi > 1.5:
        recs.append({
            'category': '📈 BMI Trend — Rapidly Increasing',
            'priority': 'MODERATE',
            'detail': (
                "BMI is forecasted to rise significantly over the next 12 months (Δ > 1.5). "
                "Monitor for signs of overnutrition or rapid weight gain. "
                "Audit snack and meal content in feeding programs — reduce high-calorie items."
            )
        })
    elif trend_bmi > 0.5:
        recs.append({
            'category': '📈 BMI Trend — Gradually Increasing',
            'priority': 'LOW',
            'detail': (
                "BMI shows a gradual upward trend (Δ 0.5–1.5). "
                "This is acceptable if within normal range. "
                "Continue regular monitoring to ensure it stays within healthy limits."
            )
        })
    elif trend_bmi < -1.0:
        recs.append({
            'category': '📉 BMI Trend — Declining',
            'priority': 'HIGH',
            'detail': (
                "BMI is forecasted to decline significantly (Δ < -1.0). "
                "This may indicate worsening nutritional status. "
                "Immediately review feeding program meal plans and increase caloric density. "
                "Investigate possible causes: illness, food insecurity, or parasitic infection."
            )
        })

    # ── 3. Weight Recommendations ─────────────────────────────────────────────
    if avg_weight < 10.0:
        recs.append({
            'category': '🔴 Weight — Critically Low',
            'priority': 'CRITICAL',
            'detail': (
                "Forecasted average weight is critically low (< 10 kg) for children aged 0–6. "
                "Provide high-protein therapeutic foods immediately. "
                "Coordinate with the barangay health center for supplemental feeding. "
                "Prioritize children under 2 years for early intervention."
            )
        })
    elif avg_weight < 12.0:
        recs.append({
            'category': '🟡 Weight — Below Average',
            'priority': 'MODERATE',
            'detail': (
                "Forecasted average weight is below expected range (10–12 kg). "
                "Enrich meals with iron-rich foods such as liver, dark green leafy vegetables, and fortified cereals. "
                "Distribute iron and zinc supplements through health programs."
            )
        })
    elif avg_weight <= 16.0:
        recs.append({
            'category': '🟢 Weight — Normal',
            'priority': 'LOW',
            'detail': (
                "Forecasted average weight is within normal range (12–16 kg). "
                "Maintain feeding schedules and food diversity. "
                "Provide continued nutrition education to caregivers."
            )
        })
    else:
        recs.append({
            'category': '🟡 Weight — Above Average',
            'priority': 'MODERATE',
            'detail': (
                "Forecasted average weight exceeds 16 kg. "
                "While this may reflect growth, assess if linked to excess caloric intake. "
                "Reduce high-fat and high-sugar food offerings in school and feeding programs."
            )
        })

    # ── 4. Height Recommendations ─────────────────────────────────────────────
    if avg_height < 82.0:
        recs.append({
            'category': '🔴 Height — Stunted Growth',
            'priority': 'HIGH',
            'detail': (
                "Forecasted average height indicates stunting risk (< 82 cm). "
                "Stunting is associated with chronic undernutrition. "
                "Increase intake of calcium-rich foods: milk, yogurt, tofu, and small fish with bones. "
                "Ensure adequate vitamin D through sunlight exposure and fortified foods. "
                "Implement catch-up growth programs in partnership with DOH."
            )
        })
    elif avg_height < 87.0:
        recs.append({
            'category': '🟡 Height — Below Expected',
            'priority': 'MODERATE',
            'detail': (
                "Forecasted average height is slightly below expected growth curve (82–87 cm). "
                "Prioritize zinc and calcium supplementation. "
                "Serve fish, milk, and legumes regularly in feeding programs. "
                "Monitor height growth quarterly using WHO growth standards."
            )
        })
    elif avg_height <= 92.0:
        recs.append({
            'category': '🟢 Height — Normal Growth',
            'priority': 'LOW',
            'detail': (
                "Forecasted average height is within normal range (87–92 cm). "
                "Continue providing calcium and vitamin D-rich foods. "
                "Encourage physical activity to support healthy bone development."
            )
        })
    else:
        recs.append({
            'category': '🟢 Height — Above Average Growth',
            'priority': 'LOW',
            'detail': (
                "Forecasted average height is above expected range (> 92 cm). "
                "This is a positive indicator of good nutrition and health. "
                "Maintain current nutritional support and document as a best practice."
            )
        })

    # ── 5. Deworming Recommendations ─────────────────────────────────────────
    if avg_dewormed < 50.0:
        recs.append({
            'category': '🔴 Deworming — Critically Low Coverage',
            'priority': 'CRITICAL',
            'detail': (
                "Deworming coverage is critically low (< 50%). "
                "Intestinal parasites severely impair nutrient absorption, causing malnutrition even with adequate food intake. "
                "Launch an emergency deworming drive targeting all children aged 1–6. "
                "Coordinate with DOH for albendazole or mebendazole distribution. "
                "Pair deworming with hygiene education: handwashing, clean water, proper sanitation."
            )
        })
    elif avg_dewormed < 65.0:
        recs.append({
            'category': '🟠 Deworming — Low Coverage',
            'priority': 'HIGH',
            'detail': (
                "Deworming coverage is low (50–65%). "
                "Significant portion of children remain at risk of parasitic infection. "
                "Schedule community deworming every 6 months. "
                "Provide caregiver education on symptoms of worm infestation."
            )
        })
    elif avg_dewormed < 80.0:
        recs.append({
            'category': '🟡 Deworming — Moderate Coverage',
            'priority': 'MODERATE',
            'detail': (
                "Deworming coverage is moderate (65–80%). "
                "Increase outreach to reach uncovered children, especially in remote areas. "
                "Use school and barangay health centers as deworming distribution points."
            )
        })
    elif avg_dewormed < 90.0:
        recs.append({
            'category': '🟢 Deworming — Good Coverage',
            'priority': 'LOW',
            'detail': (
                "Deworming coverage is good (80–90%). "
                "Sustain current deworming schedules. "
                "Target the remaining uncovered children through home visits."
            )
        })
    else:
        recs.append({
            'category': '🟢 Deworming — Excellent Coverage',
            'priority': 'LOW',
            'detail': (
                "Deworming coverage exceeds 90%. Excellent community health practice. "
                "Maintain the current program and document the approach as a model for other barangays."
            )
        })

    # ── 6. Vitamins Recommendations ───────────────────────────────────────────
    if avg_vitamins < 50.0:
        recs.append({
            'category': '🔴 Vitamins — Critically Low Intake',
            'priority': 'CRITICAL',
            'detail': (
                "Vitamin supplementation coverage is critically low (< 50%). "
                "Vitamin A deficiency increases risk of blindness and immune suppression. "
                "Iron deficiency causes anemia, fatigue, and impaired learning. "
                "Immediately distribute Vitamin A capsules (every 6 months) and iron syrup for children under 5. "
                "Integrate vitamin distribution into barangay health days."
            )
        })
    elif avg_vitamins < 65.0:
        recs.append({
            'category': '🟠 Vitamins — Low Intake',
            'priority': 'HIGH',
            'detail': (
                "Vitamin supplementation coverage is low (50–65%). "
                "Prioritize Vitamin A, iron, and iodine supplementation. "
                "Train barangay health workers to identify and refer vitamin-deficient children. "
                "Introduce vitamin-rich foods in feeding programs: malunggay, squash, papaya, liver."
            )
        })
    elif avg_vitamins < 80.0:
        recs.append({
            'category': '🟡 Vitamins — Moderate Intake',
            'priority': 'MODERATE',
            'detail': (
                "Vitamin intake coverage is moderate (65–80%). "
                "Strengthen micronutrient supplementation programs. "
                "Promote consumption of locally available vitamin-rich foods: camote tops, kangkong, ampalaya."
            )
        })
    elif avg_vitamins < 90.0:
        recs.append({
            'category': '🟢 Vitamins — Good Intake',
            'priority': 'LOW',
            'detail': (
                "Vitamin supplementation coverage is good (80–90%). "
                "Continue promoting food diversification alongside supplementation. "
                "Educate caregivers on food preparation methods that preserve vitamin content."
            )
        })
    else:
        recs.append({
            'category': '🟢 Vitamins — Excellent Intake',
            'priority': 'LOW',
            'detail': (
                "Vitamin supplementation coverage exceeds 90%. "
                "Outstanding performance. Maintain distribution schedules and diversify food sources."
            )
        })

    # ── 7. Immunization Recommendations ──────────────────────────────────────
    if avg_immunize < 50.0:
        recs.append({
            'category': '🔴 Immunization — Critically Low',
            'priority': 'CRITICAL',
            'detail': (
                "Immunization coverage is critically low (< 50%). "
                "Children are highly vulnerable to vaccine-preventable diseases like measles, polio, and diphtheria. "
                "Conduct emergency catch-up immunization campaigns immediately. "
                "Coordinate with DOH for mobile vaccination teams to reach all barangays."
            )
        })
    elif avg_immunize < 65.0:
        recs.append({
            'category': '🟠 Immunization — Low Coverage',
            'priority': 'HIGH',
            'detail': (
                "Immunization coverage is low (50–65%). "
                "Risk of disease outbreaks in the community remains high. "
                "Schedule regular immunization days at barangay health centers. "
                "Conduct house-to-house follow-up for missed children."
            )
        })
    elif avg_immunize < 80.0:
        recs.append({
            'category': '🟡 Immunization — Moderate Coverage',
            'priority': 'MODERATE',
            'detail': (
                "Immunization coverage is moderate (65–80%). "
                "Identify and prioritize children who have missed doses. "
                "Use reminder systems (text alerts, community health workers) for scheduled vaccinations."
            )
        })
    elif avg_immunize < 95.0:
        recs.append({
            'category': '🟢 Immunization — Good Coverage',
            'priority': 'LOW',
            'detail': (
                "Immunization coverage is good (80–95%). "
                "Continue current immunization programs. "
                "Aim for 95%+ herd immunity threshold by reaching remaining unimmunized children."
            )
        })
    else:
        recs.append({
            'category': '🟢 Immunization — Excellent Coverage',
            'priority': 'LOW',
            'detail': (
                "Immunization coverage exceeds 95%. Herd immunity threshold achieved. "
                "Sustain the program and share best practices with neighboring communities."
            )
        })

    # ── 8. Vaccination Recommendations ───────────────────────────────────────
    if avg_vacc < 50.0:
        recs.append({
            'category': '🔴 Vaccination — Critically Low',
            'priority': 'CRITICAL',
            'detail': (
                "Vaccination coverage is critically low (< 50%). "
                "Community is at serious risk of outbreaks. "
                "Immediately deploy vaccination outreach and partner with LGU for logistics support. "
                "Prioritize BCG, DPT, OPV, MMR, and Hepatitis B vaccines for all children 0–6."
            )
        })
    elif avg_vacc < 65.0:
        recs.append({
            'category': '🟠 Vaccination — Low Coverage',
            'priority': 'HIGH',
            'detail': (
                "Vaccination coverage is low (50–65%). "
                "Address vaccine hesitancy through community education campaigns. "
                "Set up satellite vaccination posts in accessible community areas."
            )
        })
    elif avg_vacc < 80.0:
        recs.append({
            'category': '🟡 Vaccination — Moderate Coverage',
            'priority': 'MODERATE',
            'detail': (
                "Vaccination coverage is moderate (65–80%). "
                "Track defaulters using health records and conduct follow-up visits. "
                "Integrate vaccination schedules with feeding program visits for convenience."
            )
        })
    elif avg_vacc < 95.0:
        recs.append({
            'category': '🟢 Vaccination — Good Coverage',
            'priority': 'LOW',
            'detail': (
                "Vaccination coverage is good (80–95%). "
                "Maintain current schedule and continue caregiver education. "
                "Ensure cold chain integrity for vaccine storage."
            )
        })
    else:
        recs.append({
            'category': '🟢 Vaccination — Excellent Coverage',
            'priority': 'LOW',
            'detail': (
                "Vaccination coverage exceeds 95%. Exceptional community health achievement. "
                "Document the strategies used and replicate in other areas."
            )
        })

    # ── 9. Overall Improvement Trend ─────────────────────────────────────────
    if avg_improve >= 90.0 and trend_improve >= 0:
        recs.append({
            'category': '🌟 Overall Outlook — Excellent',
            'priority': 'LOW',
            'detail': (
                "The overall child health improvement rate is forecasted at ≥ 90% and trending upward. "
                "The community's health interventions are highly effective. "
                "Continue current programs, document best practices, and consider scaling to other areas. "
                "Celebrate progress with community stakeholders to maintain motivation."
            )
        })
    elif avg_improve >= 75.0 and trend_improve >= 0:
        recs.append({
            'category': '🟢 Overall Outlook — Good',
            'priority': 'LOW',
            'detail': (
                "Child health improvement rate is forecasted at 75–90% with an upward trend. "
                "Programs are working well. Identify remaining gaps and target vulnerable subgroups. "
                "Maintain consistent monthly monitoring and reporting."
            )
        })
    elif avg_improve >= 75.0 and trend_improve < 0:
        recs.append({
            'category': '🟡 Overall Outlook — Good but Declining',
            'priority': 'MODERATE',
            'detail': (
                "Child health improvement rate is currently good (75–90%) but trending downward. "
                "Investigate causes of decline: reduced program funding, caregiver non-compliance, or seasonal factors. "
                "Reinforce health worker training and strengthen community engagement activities."
            )
        })
    elif avg_improve >= 60.0:
        recs.append({
            'category': '🟡 Overall Outlook — Moderate',
            'priority': 'MODERATE',
            'detail': (
                "Child health improvement rate is moderate (60–75%). "
                "Significant room for improvement remains. "
                "Conduct a program audit to identify which interventions are underperforming. "
                "Increase frequency of home visits and caregiver counseling sessions."
            )
        })
    elif avg_improve >= 45.0:
        recs.append({
            'category': '🟠 Overall Outlook — Below Target',
            'priority': 'HIGH',
            'detail': (
                "Child health improvement rate is below target (45–60%). "
                "Programs need significant strengthening. "
                "Convene a multi-sectoral meeting with LGU, DOH, DSWD, and DepEd to align interventions. "
                "Prioritize the most vulnerable children using community health mapping."
            )
        })
    else:
        recs.append({
            'category': '🔴 Overall Outlook — Critical',
            'priority': 'CRITICAL',
            'detail': (
                "Child health improvement rate is critically low (< 45%). "
                "Immediate and comprehensive action is required. "
                "Declare a nutrition emergency if applicable and mobilize all available resources. "
                "Implement emergency feeding, supplementation, and health interventions simultaneously. "
                "Request support from national agencies: DOH, DSWD, DOST-FNRI."
            )
        })

    # ── 10. Deworming + Vitamins Combined ─────────────────────────────────────
    if avg_dewormed < 65.0 and avg_vitamins < 65.0:
        recs.append({
            'category': '⚠️ Combined Risk — Deworming & Vitamins Both Low',
            'priority': 'CRITICAL',
            'detail': (
                "Both deworming and vitamin intake are forecasted to be low simultaneously. "
                "This combination severely increases risk of severe malnutrition and stunting. "
                "Launch an integrated health campaign combining deworming, vitamin distribution, and nutrition counseling in a single community event. "
                "Partner with local schools and churches for maximum reach."
            )
        })
    elif avg_dewormed < 80.0 and avg_vitamins < 80.0:
        recs.append({
            'category': '⚠️ Combined Risk — Deworming & Vitamins Moderate',
            'priority': 'MODERATE',
            'detail': (
                "Both deworming and vitamin coverage are forecasted below 80%. "
                "Combine deworming and vitamin distribution in a single health day to improve compliance and efficiency. "
                "Provide caregivers with take-home educational materials on nutrition and hygiene."
            )
        })

    # ── 11. Immunization + Vaccination Combined ───────────────────────────────
    if avg_immunize < 65.0 and avg_vacc < 65.0:
        recs.append({
            'category': '⚠️ Combined Risk — Immunization & Vaccination Both Low',
            'priority': 'CRITICAL',
            'detail': (
                "Both immunization and vaccination coverage are critically low. "
                "The community is at high risk for disease outbreaks. "
                "Immediately coordinate with regional DOH for a catch-up vaccination drive. "
                "Deploy mobile health units to reach isolated communities."
            )
        })
    elif avg_immunize < 80.0 and avg_vacc < 80.0:
        recs.append({
            'category': '⚠️ Combined Risk — Immunization & Vaccination Moderate',
            'priority': 'MODERATE',
            'detail': (
                "Both immunization and vaccination coverage are below 80%. "
                "Combine immunization and vaccination schedules to reduce missed appointments. "
                "Use SMS reminders and barangay announcements to notify caregivers."
            )
        })

    # ── 12. All Interventions Excellent ───────────────────────────────────────
    if (avg_dewormed >= 90.0 and avg_vitamins >= 90.0 and
            avg_immunize >= 90.0 and avg_vacc >= 90.0):
        recs.append({
            'category': '🏆 All Health Interventions — Excellent',
            'priority': 'LOW',
            'detail': (
                "All four health interventions (deworming, vitamins, immunization, vaccination) "
                "are forecasted at ≥ 90% coverage. "
                "This is outstanding. The community demonstrates exemplary health program implementation. "
                "Submit this barangay's health data as a model case study for regional replication."
            )
        })

    return recs, {
        'avg_bmi': avg_bmi, 'avg_weight': avg_weight, 'avg_height': avg_height,
        'avg_dewormed': avg_dewormed, 'avg_vitamins': avg_vitamins,
        'avg_immunize': avg_immunize, 'avg_vacc': avg_vacc,
        'avg_improve': avg_improve, 'trend_bmi': trend_bmi,
        'trend_improve': trend_improve
    }


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
c1.metric("📅 Forecast Horizon",           f"{n_months} months")
c2.metric("📌 Current Improvement Rate",   f"{cur_imp:.1f}%")
c3.metric("🔮 Month +1 Forecast",          f"{mo1_imp:.1f}%",  delta=f"{mo1_imp - cur_imp:+.1f}%")
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

# ── TAB 4: Nutrition Recommendations ─────────────────────────────────────────
with tab4:
    st.subheader("💡 Nutrition & Health Recommendations")
    st.markdown(
        f"Based on **{n_months}-month ARIMA forecast** averages. "
        "Recommendations are generated automatically using forecasted health indicator values."
    )

    # Only generate recommendations for 12-month forecast for accuracy
    rec_df, stats = get_recommendations(
        forecast_df if n_months <= 12 else forecast_df.head(12)
    )

    # Show forecast averages used
    with st.expander("📊 Forecasted Averages Used for Recommendations", expanded=False):
        avg_data = {
            'Indicator': [
                'Average BMI', 'Average Weight (kg)', 'Average Height (cm)',
                '% Dewormed', '% Vitamins Intake',
                '% Immunization', '% Vaccination', '% Predicted to Improve'
            ],
            'Forecasted Average': [
                round(stats['avg_bmi'], 2), round(stats['avg_weight'], 2),
                round(stats['avg_height'], 2), round(stats['avg_dewormed'], 2),
                round(stats['avg_vitamins'], 2), round(stats['avg_immunize'], 2),
                round(stats['avg_vacc'], 2), round(stats['avg_improve'], 2)
            ]
        }
        st.dataframe(pd.DataFrame(avg_data), use_container_width=True, hide_index=True)

    # Priority filter
    priority_filter = st.selectbox(
        "Filter by Priority",
        options=['All', 'CRITICAL', 'HIGH', 'MODERATE', 'LOW'],
        index=0
    )

    priority_colors = {
        'CRITICAL': '🔴',
        'HIGH':     '🟠',
        'MODERATE': '🟡',
        'LOW':      '🟢',
    }

    filtered = rec_df if priority_filter == 'All' else [r for r in rec_df if r['priority'] == priority_filter]

    if not filtered:
        st.info("No recommendations match the selected priority filter.")
    else:
        st.markdown(f"**Showing {len(filtered)} recommendation(s)**")
        st.divider()
        for rec in filtered:
            badge = priority_colors.get(rec['priority'], '')
            with st.expander(f"{badge} [{rec['priority']}] {rec['category']}", expanded=(rec['priority'] in ['CRITICAL', 'HIGH'])):
                st.write(rec['detail'])
