import pandas as pd

# === 1. Load data utama ===
app = pd.read_csv("application_train.csv")
bureau = pd.read_csv("bureau.csv")
prev = pd.read_csv("previous_application.csv")
instal = pd.read_csv("installments_payments.csv")

print("Original shape of application_train:", app.shape)

# === 2. Agregasi bureau.csv ===
bureau_agg = bureau.groupby("SK_ID_CURR").agg({
    "AMT_CREDIT_SUM": "sum",
    "AMT_CREDIT_SUM_DEBT": "sum",
    "CREDIT_ACTIVE": lambda x: (x == "Active").sum(),
    "CREDIT_TYPE": pd.Series.nunique,
    "DAYS_CREDIT": "mean",
    "AMT_ANNUITY": "mean",
    "CNT_CREDIT_PROLONG": "sum",
    "AMT_CREDIT_SUM_OVERDUE": "sum"
}).rename(columns={
    "AMT_CREDIT_SUM": "TOTAL_BUREAU_CREDIT",
    "AMT_CREDIT_SUM_DEBT": "TOTAL_BUREAU_DEBT",
    "CREDIT_ACTIVE": "NUM_ACTIVE_LOANS",
    "CREDIT_TYPE": "NUM_CREDIT_TYPES",
    "DAYS_CREDIT": "AVG_DAYS_CREDIT",
    "AMT_ANNUITY": "AVG_BUREAU_ANNUITY",
    "CNT_CREDIT_PROLONG": "TOTAL_PROLONG_COUNT",
    "AMT_CREDIT_SUM_OVERDUE": "TOTAL_OVERDUE_BUREAU"
}).reset_index()

print("bureau_agg shape:", bureau_agg.shape)

# === 3. Agregasi previous_application.csv ===
prev_agg = prev.groupby("SK_ID_CURR").agg({
    "SK_ID_PREV": "count",
    "AMT_APPLICATION": "mean",
    "AMT_CREDIT": "mean",
    "AMT_ANNUITY": "mean",
    "NAME_CONTRACT_STATUS": lambda x: (x == "Refused").sum(),
    "DAYS_DECISION": "mean",
    "CNT_PAYMENT": "mean",
    "DAYS_LAST_DUE": "mean"
}).rename(columns={
    "SK_ID_PREV": "NUM_PREV_APPLICATIONS",
    "AMT_APPLICATION": "AVG_AMT_APPLICATION",
    "AMT_CREDIT": "AVG_AMT_CREDIT",
    "AMT_ANNUITY": "AVG_PREV_ANNUITY",
    "NAME_CONTRACT_STATUS": "NUM_PREV_REFUSED",
    "DAYS_DECISION": "AVG_DECISION_DAYS",
    "CNT_PAYMENT": "AVG_CNT_PAYMENT",
    "DAYS_LAST_DUE": "AVG_DAYS_LAST_DUE"
}).reset_index()

print("prev_agg shape:", prev_agg.shape)

# === 4. Agregasi installments_payments.csv ===
# Hitung keterlambatan dalam hari (positif saja)
instal["LATE_DAYS"] = instal["DAYS_ENTRY_PAYMENT"] - instal["DAYS_INSTALMENT"]
instal["LATE_DAYS"] = instal["LATE_DAYS"].apply(lambda x: x if x > 0 else 0)

# Agregasi per nasabah
instal_agg = instal.groupby("SK_ID_CURR").agg({
    "LATE_DAYS": ["mean", "max", "sum"],
    "AMT_INSTALMENT": "mean",
    "AMT_PAYMENT": "mean"
})

# Rename kolom multiindex
instal_agg.columns = [
    "AVG_LATE_DAYS", "MAX_LATE_DAYS", "TOTAL_LATE_DAYS",
    "AVG_AMT_INSTALMENT", "AVG_AMT_PAYMENT"
]
instal_agg = instal_agg.reset_index()

print("installments_agg shape:", instal_agg.shape)

# === 5. Merge semua agregasi ke application_train ===
app_merged = app.merge(bureau_agg, how="left", on="SK_ID_CURR")
app_merged = app_merged.merge(prev_agg, how="left", on="SK_ID_CURR")
app_merged = app_merged.merge(instal_agg, how="left", on="SK_ID_CURR")

print("Final merged shape:", app_merged.shape)

# === 6. Simpan hasil akhir ===
app_merged.to_csv("application_merged_final.csv", index=False)
print("✅ Data lengkap disimpan sebagai 'application_merged_final.csv'")
