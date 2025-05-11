import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load data
df = pd.read_csv("application_merged_final.csv")

# Pastikan data valid
df = df[df["OCCUPATION_TYPE"].notna()]
df["LATE_GROUP"] = df["AVG_LATE_DAYS"].apply(lambda x: ">5 Hari" if x > 5 else "≤5 Hari")

# Group berdasarkan pekerjaan & grup keterlambatan
grouped = df.groupby(["OCCUPATION_TYPE", "LATE_GROUP"])["TARGET"].agg(['mean', 'count']).reset_index()
grouped = grouped.rename(columns={"mean": "Gagal_Bayar_Rate", "count": "Jumlah_Nasabah"})

# Filter untuk jumlah nasabah yang cukup
grouped_filtered = grouped[grouped["Jumlah_Nasabah"] > 50]

# Sort berdasarkan gagal bayar tertinggi
grouped_sorted = grouped_filtered.sort_values(by="Gagal_Bayar_Rate", ascending=False)

print("\n📊 Top Kombinasi Pekerjaan + Keterlambatan yang Berisiko:")
print(grouped_sorted.head(10))

# Visualisasi
plt.figure(figsize=(12, 8))
sns.barplot(
    data=grouped_sorted.head(10),
    y="OCCUPATION_TYPE",
    x="Gagal_Bayar_Rate",
    hue="LATE_GROUP",
    palette="Reds"
)
plt.title("Top 10 Kombinasi Occupation + Keterlambatan vs Rasio Gagal Bayar")
plt.xlabel("Rasio Gagal Bayar")
plt.ylabel("Jenis Pekerjaan")
plt.tight_layout()
plt.savefig("insight/insight_gabungan_occupation_late_days.png")
plt.show()

# Grouping default rate berdasarkan jenis penghasilan
income_risk = df.groupby("NAME_INCOME_TYPE")["TARGET"].agg(['mean', 'count']).sort_values(by='mean', ascending=False)
income_risk = income_risk.rename(columns={"mean": "Default_Rate", "count": "Jumlah_Nasabah"})
print("\n📌 Risiko Gagal Bayar berdasarkan Income Type:")
print(income_risk)

# Visualisasi
income_risk = income_risk.reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=income_risk, y="NAME_INCOME_TYPE", x="Default_Rate", hue="NAME_INCOME_TYPE", palette="rocket", dodge=False)
plt.xlabel("Rasio Gagal Bayar")
plt.title("Default Rate Berdasarkan Income Type")
plt.legend([],[], frameon=False)  # Hilangkan legenda
plt.tight_layout()
plt.savefig("insight/insight2_income_type_default_rate.png")
plt.show()
