# otomobilin sensörlerine göre veri seti oluşturur

import csv
import numpy as np

CSV_DOSYA = "veri_seti.csv"
LIDAR_ISIN_SAYISI = 25
LIDAR_MESAFE = 300

def generate_data():
    with open(CSV_DOSYA, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # 1. (SÜRDÜR) - 50 örnek
        for _ in range(50):
            # Tüm sensörler max mesafede
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            row.append("SÜRDÜR")
            writer.writerow(row)
            
        # 2. (FREN/YAVAŞLA) - 10 örnek
        for _ in range(10):
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            # Orta sensörler engelleniyor
            for i in range(9, 17):
                row[i] = np.random.randint(50, 100)
            row.append("FREN")
            writer.writerow(row)

        # 3. (SOLA_KAÇIN) - 20 örnek
        for _ in range(20):
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            # Orta sensörler engelleniyor
            for i in range(9, 17):
                row[i] = np.random.randint(50, 150)
            # Sağ sensörler engelleniyor
            for i in range(17, 25):
                row[i] = np.random.randint(50, 150)
            # Sol sensörler açık
            row.append("SOLA_KAÇIN")
            writer.writerow(row)

        # 4. (SAĞA_KAÇIN) - 20 örnek
        for _ in range(20):
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            # Orta sensörler engelleniyor
            for i in range(9, 17):
                row[i] = np.random.randint(50, 150)
            # Sol sensörler engelleniyor
            for i in range(0, 9):
                row[i] = np.random.randint(50, 150)
            # Sağ sensörler açık
            row.append("SAĞA_KAÇIN")
            writer.writerow(row)

        # 5. (SAĞA_KAÇIN) - 10 örnek
        for _ in range(10):
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            # Sol sensörler engelleniyor
            for i in range(0, 9):
                row[i] = np.random.randint(50, 150)
            row.append("SAĞA_KAÇIN")
            writer.writerow(row)

        # 6. (SOLA_KAÇIN) - 10 örnek
        for _ in range(10):
            row = [LIDAR_MESAFE] * LIDAR_ISIN_SAYISI
            # Sağ sensörler engelleniyor
            for i in range(17, 25):
                row[i] = np.random.randint(50, 150)
            row.append("SOLA_KAÇIN")
            writer.writerow(row)
            
    print(f"Veri seti başarıyla oluşturuldu: {CSV_DOSYA}")

if __name__ == "__main__":
    generate_data()
