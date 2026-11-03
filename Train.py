import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Load data di komputer lokal
df = pd.read_csv('DATA (1).csv') # Sesuaikan nama file CSV Anda

X = df.drop(columns=['GRADE']) # Sesuaikan kolom target
y = df['GRADE']

# 2. Encoding data teks
X_encoded = pd.get_dummies(X)

# 3. Train model
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# 4. Simpan model baru
joblib.dump(tree_model, 'model_decision_tree.pkl')
print("Model baru berhasil dibuat dan disimpan di komputer lokal!")