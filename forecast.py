"""
forecast.py — Child Health Trend Forecasting
Uses trained ARIMA models to forecast child health indicators.

Forecast mode: Next N Months (user chooses 1–60, default 12)

Usage:
    python forecast.py
"""

import pickle
import numpy as np
import pandas as pd


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


# ── Helpers ───────────────────────────────────────────────────────────────────

def next_month_labels(last_label, n):
    """Return n month label strings after last_label (e.g. 'Dec-2025')."""
    m, y   = last_label.split('-')
    mi, yr = MONTH_NAMES.index(m), int(y)
    labels = []
    for _ in range(n):
        mi += 1
        if mi >= 12:
            mi, yr = 0, yr + 1
        labels.append(f"{MONTH_NAMES[mi]}-{yr}")
    return labels


def load_bundle(path='arima_models.pkl'):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"\n  Error: '{path}' not found.")
        print("  Please run the training notebook first to generate the pickle.")
        return None


def get_int(prompt, lo, hi):
    while True:
        try:
            v = int(input(prompt))
            if lo <= v <= hi:
                return v
            print(f"  Enter a number between {lo} and {hi}.")
        except ValueError:
            print("  Please enter a whole number.")


def trend_arrow(change):
    if change >  0.01: return '↑  (+)'
    if change < -0.01: return '↓  (-)'
    return                     '→  (=)'


# ── Core Forecast ─────────────────────────────────────────────────────────────

def forecast_n_months(bundle, n):
    """
    Forecast all target columns n months ahead.
    Returns a DataFrame and prints a formatted summary table.
    """
    models        = bundle['models']
    history       = bundle['history']
    last_label    = bundle['last_month_label']
    last_no       = bundle['last_month_no']
    targets       = bundle['targets']
    future_labels = next_month_labels(last_label, n)

    # Build result DataFrame
    result = pd.DataFrame({
        'Month_No':    range(last_no + 1, last_no + n + 1),
        'Month_Label': future_labels
    })
    for col in targets:
        fc = models[col].predict(n_periods=n)
        result[col] = np.round(fc, 2)

    # ── Print per-column tables ───────────────────────────────────────────────
    print()
    print('=' * 62)
    print(f'  ARIMA FORECAST — NEXT {n} MONTHS')
    print(f'  From {future_labels[0]}  to  {future_labels[-1]}')
    print(f'  Age Group: 0–6 years')
    print('=' * 62)

    for col in targets:
        label    = COLUMN_LABELS.get(col, col)
        fc_vals  = result[col].tolist()
        prev_val = history[col].iloc[-1]

        print(f"\n  {'★ ' if col == 'Pct_Improved_Next_Month' else '  '}{label}")
        print(f"  {'Month':<12} {'Forecast':>9}  {'Change':>8}  Trend")
        print(f"  {'-'*46}")

        prev = prev_val
        for lbl, val in zip(future_labels, fc_vals):
            chg = val - prev
            print(f"  {lbl:<12} {val:>9.2f}  {chg:>+8.2f}  {trend_arrow(chg)}")
            prev = val

    # ── Summary ───────────────────────────────────────────────────────────────
    imp_col   = 'Pct_Improved_Next_Month'
    cur_imp   = history[imp_col].iloc[-1]
    mo1_imp   = result[imp_col].iloc[0]
    final_imp = result[imp_col].iloc[-1]
    direction = 'IMPROVE' if final_imp > cur_imp else 'DECLINE'

    print()
    print('=' * 62)
    print('  SUMMARY — % Children Predicted to Improve')
    print(f'  Current   ({last_label})  : {cur_imp:.2f}%')
    print(f'  Month +1  ({future_labels[0]}) : {mo1_imp:.2f}%')
    print(f'  Month +{n:<2} ({future_labels[-1]}): {final_imp:.2f}%')
    print(f'  Overall trend             : {direction}')
    print('=' * 62)
    print()

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print('=' * 62)
    print('  CHILD HEALTH TREND FORECASTING SYSTEM')
    print('  Powered by ARIMA  |  Age Group: 0–6 years')
    print('=' * 62)

    bundle = load_bundle('arima_models.pkl')
    if bundle is None:
        return

    history    = bundle['history']
    last_label = bundle['last_month_label']
    last_no    = bundle['last_month_no']

    print(f"\n  Historical data : Month 1 – Month {last_no}")
    print(f"  Period          : {history['Month_Label'].iloc[0]} – {last_label}")
    print(f"  Trained columns : {len(bundle['targets'])}")

    while True:
        print()
        n = get_int('  How many months to forecast? (1–60, default 12): ', 1, 60)

        result = forecast_n_months(bundle, n)

        export = input('  Export forecast to CSV? (Yes/No): ').strip().capitalize()
        if export in ['Yes', 'Y']:
            fname = f'forecast_{n}months.csv'
            result.to_csv(fname, index=False)
            print(f'  Saved → {fname}')

        again = input('\n  Run another forecast? (Yes/No): ').strip().capitalize()
        if again not in ['Yes', 'Y']:
            print('\n  Thank you for using the Child Health Forecasting System.')
            break


if __name__ == '__main__':
    main()
