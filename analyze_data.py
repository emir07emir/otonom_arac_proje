import pandas as pd
import numpy as np

try:
    df = pd.read_csv('veri_seti.csv', header=None)
    
    # sayıya çeviriyoruz ve hata olursa 300 yap çünkü max mesafe 300
    X = df.iloc[:, :-1].apply(pd.to_numeric, errors='coerce').fillna(300)
    y = df.iloc[:, -1]
    
    # tüm girdiler 300 (veya 300'e yakın) olan satırları filtrele
    mask = (X == 300).all(axis=1)
    
    empty_road_data = y[mask]
    
    print(f"Toplam satır: {len(df)}")
    print(f"Tüm girdiler 300 (veya 300'e yakın) olan satır sayısı: {len(empty_road_data)}")
    print("Etiket dağılımı:")
    print(empty_road_data.value_counts())
    
except Exception as e:
    print(f"Hata: {e}")
