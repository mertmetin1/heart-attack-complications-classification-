import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt




# Veri setini yükle
df = pd.read_excel('/home/hp-laserjet-wrt8070/Desktop/MI.xlsm')

# Veri setinin genel bilgilerini kontrol et
print("Veri Seti Başlıkları ve İlk 3 Satır:")
print(df.head(3))
print("\nVeri Seti Bilgileri:")
print(df.info())
print("\nVeri Seti İstatistikleri:")
print(df.describe())
print("\nEksik Değerlerin Sayısı:")
print(df.isnull().sum())
print("\n'Sutun2' Benzersiz Değerleri:")
print(df['Sutun2'].unique())

# Eksik değerleri ortalama ile doldur
df.fillna(df.mean(), inplace=True)

# 'Sutun1' sütununu veri setinden çıkar
df.drop(columns=['Sutun1'], inplace=True)

# 'Sutun2' sütunundaki sayısal değerleri harflere dönüştür
df['Sutun2'] = df['Sutun2'].map({0: 'a', 1: 'b', 2: 'c', 3: 'd'})

# Özellik ve hedef değişkenlerini ayır
X = df.drop(columns=['Sutun2'])
y = df['Sutun2']

# Min-max normalizasyon uygula
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RFE ile özellik seçimi (10 özellik)
num_features_to_select =50
rfe = RFE(estimator=LinearDiscriminantAnalysis(), n_features_to_select=num_features_to_select)
X_rfe = rfe.fit_transform(X_scaled, y)

# Seçilen özellikleri kullanarak veriyi güncelle
selected_features = rfe.support_
df_rfe = pd.DataFrame(X_rfe)

# Özellik veri dağılımı haritası oluştur
#sns.pairplot(df_rfe)
#plt.show()

"""# Korelasyon matrisi ile seçilen özellikleri görüntüle
df_rfe_cm = df_rfe.corr()
print("\nKorelasyon Matrisi:")
print(df_rfe_cm)

# Korelasyon matrisini görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(df_rfe_cm, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Özellikler Arasındaki Korelasyon Matrisi')
plt.xlabel('Özellikler')
plt.ylabel('Özellikler')
plt.show()
"""


# Veriyi eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(df_rfe, y, test_size=0.3, random_state=42)



# Sınıflandırma modellerini tanımla
models = {
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": SVC(kernel='rbf', C=1.0, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=4, random_state=42)
}

# Her model için eğitim ve değerlendirme yap
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{name} Sınıflandırma Performansı:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy:.2f}")

