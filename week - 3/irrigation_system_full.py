
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import joblib

# -------------------------------
# STEP 1: LOAD AND CLEAN DATA
# -------------------------------
df = pd.read_csv("C:\\Users\\yamin\\Desktop\\week - 2\\irrigation_machine (3).csv", index_col=0)

print("Data Shape:", df.shape)
print(df.head())
print(df.info())

# -------------------------------
# STEP 2: FEATURES & LABELS
# -------------------------------
X = df.iloc[:, 0:20]    # First 20 columns: Sensors
y = df.iloc[:, 20:]     # Remaining columns: Parcels

print("Feature shape:", X.shape, "Label shape:", y.shape)

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# STEP 3: TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# STEP 4: TRAIN MODEL
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

model = MultiOutputClassifier(rf)
model.fit(X_train, y_train)

# -------------------------------
# STEP 5: EVALUATE MODEL
# -------------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=y.columns))

# -------------------------------
# STEP 6: VISUALIZATION 1 - Parcel Conditions
# -------------------------------
conditions = {
    "Parcel 0 ON": df['parcel_0'],
    "Parcel 1 ON": df['parcel_1'],
    "Parcel 2 ON": df['parcel_2'],
    "Parcel 0 & 1 ON": df['parcel_0'] & df['parcel_1'],
    "Parcel 0 & 2 ON": df['parcel_0'] & df['parcel_2'],
    "Parcel 1 & 2 ON": df['parcel_1'] & df['parcel_2'],
    "All Parcels ON": df['parcel_0'] & df['parcel_1'] & df['parcel_2'],
}

fig, axs = plt.subplots(nrows=len(conditions), figsize=(10, 15), sharex=True)

for ax, (title, condition) in zip(axs, conditions.items()):
    ax.step(df.index, condition.astype(int), where='post', linewidth=1, color='teal')
    ax.set_title(f"Sprinkler - {title}")
    ax.set_ylabel("Status")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['OFF', 'ON'])

axs[-1].set_xlabel("Time Index (Row Number)")
plt.tight_layout()
plt.show()

# -------------------------------
# STEP 7: VISUALIZATION 2 - Combined Pump Activity
# -------------------------------
plt.figure(figsize=(15, 5))
plt.step(df.index, df['parcel_0'], where='post', linewidth=2, label='Parcel 0 Pump', color='blue')
plt.step(df.index, df['parcel_1'], where='post', linewidth=2, label='Parcel 1 Pump', color='orange')
plt.step(df.index, df['parcel_2'], where='post', linewidth=2, label='Parcel 2 Pump', color='green')

plt.title("Pump Activity and Combined Farm Coverage")
plt.xlabel("Time Index (Row Number)")
plt.ylabel("Status")
plt.yticks([0, 1], ['OFF', 'ON'])
plt.legend(loc='upper right')
plt.show()

# -------------------------------
# STEP 8: SAVE MODEL & SCALER
# -------------------------------
joblib.dump({
    'model': model,
    'scaler': scaler
}, "Farm_Irrigation_System.pkl")

print("\n✅ Model and Scaler saved as Farm_Irrigation_System.pkl")
