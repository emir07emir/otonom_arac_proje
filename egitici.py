import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Dosya Adları
CSV_DOSYA_ADI = "otonom_veri_seti.csv"
MODEL_DOSYA_ADI = "knn_model.pkl"

def modeli_egit_ve_raporla():
    print("\n" + "="*50)
    print("   OTONOM ARAÇ YAPAY ZEKA EĞİTİM MODÜLÜ")
    print("="*50 + "\n")
    
    # 1. Veri Kontrolü
    if not os.path.exists(CSV_DOSYA_ADI):
        print(f"HATA: '{CSV_DOSYA_ADI}' bulunamadı! Önce veri toplayın.")
        return

    try:
        df = pd.read_csv(CSV_DOSYA_ADI)
        print(f"[BİLGİ] Veri seti yüklendi. Toplam Satır: {len(df)}")
    except Exception as e:
        print("Okuma hatası:", e)
        return

    if len(df) < 20:
        print("UYARI: Veri sayısı çok az. Doğru sonuç için en az 100 satır veri önerilir.")
        return

    # 2. Veri Hazırlığı
    X = df.iloc[:, :-1].values  # Girdiler (Lidar + Hız + İvme)
    y = df.iloc[:, -1].values   # Çıktı (Aksiyon)

    # %80 Eğitim, %20 Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. KNN İŞLEMLERİ
    print("[İŞLEM] En iyi 'K' (Komşu Sayısı) değeri aranıyor...")
    best_k = 1
    best_acc = 0
    
    # 1'den 15'e kadar dene
    for k in range(1, 15):
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train, y_train)
        acc = knn_temp.score(X_test, y_test)
        if acc > best_acc:
            best_k = k
            best_acc = acc
            
    print(f"[SONUÇ] Optimum K Değeri: {best_k} (Başarı: %{best_acc*100:.1f})")

    # 4. Modeli En İyi K ile Eğitme
    print(f"\n[EĞİTİM] Model K={best_k} ile eğitiliyor...")
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    
    # 5. Performans Raporu
    y_pred = knn.predict(X_test)
    
    print("\n" + "-"*30)
    print("   PERFORMANS RAPORU")
    print("-"*30)
    print(f"Genel Doğruluk (Accuracy): %{accuracy_score(y_test, y_pred)*100:.2f}")
    
    print("\n--- Detaylı Metrikler (Precision / Recall / F1) ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Karmaşıklık Matrisi (Confusion Matrix) ---")
    print("(Satırlar: Gerçek, Sütunlar: Tahmin)")
    print(confusion_matrix(y_test, y_pred))
    
    # 6. Kayıt
    joblib.dump(knn, MODEL_DOSYA_ADI)
    print("\n" + "="*50)
    print(f" Model başarıyla kaydedildi: {MODEL_DOSYA_ADI}")
    print(" Simülasyonu çalıştırabilirsiniz.")
    print("="*50 + "\n")

if __name__ == "__main__":
    modeli_egit_ve_raporla()