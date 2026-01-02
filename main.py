import pygame
import sys
import math
import numpy as np
from collections import deque
import random
import csv
import os
import joblib 

# ---------- AYARLAR ----------
SIM_GENISLIK = 1000
PANEL_GENISLIK = 450
EKRAN_G, EKRAN_Y = SIM_GENISLIK + PANEL_GENISLIK, 650
FPS = 60

# Dosya Adları
CSV_DOSYA_ADI = "otonom_veri_seti.csv"
MODEL_DOSYA_ADI = "knn_model.pkl"

# Renkler
RENK_ZEMIN = (10, 15, 20); RENK_ARAC = (0, 255, 255); RENK_GRID = (40, 50, 60)
RENK_PANEL_BG = (25, 25, 35); RENK_ENGEL = (255, 50, 50); RENK_LIDAR = (0, 255, 100)
RENK_YAZI = (220, 220, 220)

# Fiziksel Parametreler
MAKS_HIZ = 100.0
LIDAR_MESAFE = 250
LIDAR_ACILAR = np.linspace(-60, 60, 40) * math.pi/180 

# Aksiyon kısımları
AKSIYON_LISTESI = ["SOLA_KAÇIN", "SAĞA_KAÇIN", "FREN", "SÜRDÜR"]
AKSIYON_DICT = {aks: i for i, aks in enumerate(AKSIYON_LISTESI)}
# ----------------------------



#pencere olusturma
pygame.init()
ekran = pygame.display.set_mode((EKRAN_G, EKRAN_Y))
pygame.display.set_caption("Otonom Araç: Adaptif Hız Kontrollü Karar Destek Sistemi")
saat = pygame.time.Clock()

font_baslik = pygame.font.SysFont("Consolas", 22, bold=True)
font_veri = pygame.font.SysFont("Consolas", 16)
font_mini = pygame.font.SysFont("Consolas", 12)
font_buton = pygame.font.SysFont("Arial", 20, bold=True)

BUTON_RECT = pygame.Rect(SIM_GENISLIK + 50, EKRAN_Y - 80, 350, 50)

# --- CSV Fonksiyonu ---
if not os.path.exists(CSV_DOSYA_ADI):
    with open(CSV_DOSYA_ADI, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        basliklar = [f"Lidar_{i}" for i in range(len(LIDAR_ACILAR))] + ["HIZ", "IVME", "AKSIYON"]
        writer.writerow(basliklar)

def veri_kaydet(lidar_verisi, arac_hizi, arac_ivmesi, aksiyon):
    with open(CSV_DOSYA_ADI, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        row = [round(x, 1) for x in lidar_verisi] + [round(arac_hizi, 1), round(arac_ivmesi, 1), aksiyon]
        writer.writerow(row)

# --- YAPAY ZEKA SÜRÜCÜSÜ ---
class YapayZekaSurucusu:
    def __init__(self):
        self.model = self._modeli_yukle()
        self.egitildi = self.model is not None
        
    def _modeli_yukle(self):
        try:
            if os.path.exists(MODEL_DOSYA_ADI):
                model = joblib.load(MODEL_DOSYA_ADI)
                print(f"BİLGİ: '{MODEL_DOSYA_ADI}' yüklendi. HİBRİT MOD devrede.")
                return model
            else:
                print(f"UYARI: '{MODEL_DOSYA_ADI}' yok. Sadece Matematik Modu çalışıyor.")
                return None
        except Exception as e:
            print(f"HATA: Model yüklenemedi: {e}")
            return None

    def olasiliklari_getir(self, lidar_verisi, hiz, ivme):
        if not self.egitildi:
            return np.array([0.25, 0.25, 0.25, 0.25])
        
        giris_verisi = lidar_verisi + [hiz, ivme]
        
        try:
            ham_olasilik = self.model.predict_proba([giris_verisi])[0]
            sirali_olasiliklar = np.zeros(4)
            modelin_siniflari = self.model.classes_
            
            for i, sinif_adi in enumerate(modelin_siniflari):
                if sinif_adi in AKSIYON_DICT:
                    target_idx = AKSIYON_DICT[sinif_adi]
                    sirali_olasiliklar[target_idx] = ham_olasilik[i]
            return sirali_olasiliklar
        except Exception as e:
            return np.array([0.25, 0.25, 0.25, 0.25])

#  KARAR VERİCİ
class MCDMKararVerici:
    def __init__(self):
        self.weights = np.array([0.60, 0.20, 0.20]) 
        self.impacts = np.array([1, 1, -1]) 

    def topsis_hesapla(self, karar_matrisi):
        matris = np.array(karar_matrisi, dtype=float)
        rows, cols = matris.shape
        payda = np.sqrt(np.sum(matris**2, axis=0))
        payda[payda == 0] = 1 
        norm_matris = matris / payda
        agirlikli = norm_matris * self.weights
        ideal = np.zeros(cols)
        negatif_ideal = np.zeros(cols)
        for i in range(cols):
            if self.impacts[i] == 1: 
                ideal[i] = np.max(agirlikli[:, i])
                negatif_ideal[i] = np.min(agirlikli[:, i])
            else: 
                ideal[i] = np.min(agirlikli[:, i])
                negatif_ideal[i] = np.max(agirlikli[:, i])
        dist_pos = np.sqrt(np.sum((agirlikli - ideal)**2, axis=1))
        dist_neg = np.sqrt(np.sum((agirlikli - negatif_ideal)**2, axis=1))
        toplam_dist = dist_pos + dist_neg
        toplam_dist[toplam_dist == 0] = 1
        skorlar = dist_neg / toplam_dist
        return skorlar

# ARAC, ENGEL, SENSOR SINIFLARI
class SensorPaketi:
    def __init__(self, arac):
        self.arac = arac
        self.gps_noise = 2.0 
    def veri_oku(self):
        gps_x = self.arac.x + random.uniform(-self.gps_noise, self.gps_noise)
        gps_y = self.arac.y + random.uniform(-self.gps_noise, self.gps_noise)
        ivme_x = self.arac.ivme * math.cos(self.arac.yon) / 10.0 
        return {"gps": (gps_x, gps_y), "imu_ivme": ivme_x}

class Arac:
    def __init__(self, x, y):
        self.x = x; self.y = y; self.yon = 0.0; self.hiz = 30.0; self.ivme = 0.0
        self.gecmis = deque(maxlen=100)
    def adim(self, dt):
        self.hiz += self.ivme * dt
        self.hiz = max(0.0, min(MAKS_HIZ, self.hiz))
        self.x += math.cos(self.yon) * self.hiz * dt
        self.y += math.sin(self.yon) * self.hiz * dt
        self.gecmis.append((self.x, self.y))
    def ciz(self, ekran):
        uzunluk = 40
        p1 = (self.x + lengthdir_x(uzunluk, self.yon), self.y + lengthdir_y(uzunluk, self.yon))
        p2 = (self.x + lengthdir_x(uzunluk/2, self.yon + 2.5), self.y + lengthdir_y(uzunluk/2, self.yon + 2.5))
        p3 = (self.x + lengthdir_x(uzunluk/2, self.yon - 2.5), self.y + lengthdir_y(uzunluk/2, self.yon - 2.5))
        pygame.draw.polygon(ekran, RENK_ARAC, [p1, p2, p3], 2)
        pygame.draw.circle(ekran, RENK_ARAC, (int(self.x), int(self.y)), 4)
        if len(self.gecmis) > 2:
            pygame.draw.lines(ekran, (0, 100, 100), False, list(self.gecmis), 1)

class Engel:
    def __init__(self, x, y, w, h, hiz_x=0):
        self.rect = pygame.Rect(x, y, w, h)
        self.hiz_x = hiz_x; self.tip = "DINAMIK" if hiz_x != 0 else "STATIK"
    def adim(self, dt): self.rect.x += self.hiz_x * dt
    def ciz(self, ekran):
        renk = RENK_ENGEL if self.tip == "DINAMIK" else (150, 50, 50)
        pygame.draw.rect(ekran, renk, self.rect, 2)
        pygame.draw.line(ekran, renk, self.rect.topleft, self.rect.bottomright, 1)

def lengthdir_x(len, dir): return len * math.cos(dir)
def lengthdir_y(len, dir): return len * math.sin(dir)
def line_rect_collision(x1, y1, x2, y2, rect):
    min_t = 1.0; hit = False
    lines = [((rect.left, rect.top), (rect.right, rect.top)),
             ((rect.right, rect.top), (rect.right, rect.bottom)),
             ((rect.right, rect.bottom), (rect.left, rect.bottom)),
             ((rect.left, rect.bottom), (rect.left, rect.top))]
    for p3, p4 in lines:
        x3,y3 = p3; x4,y4 = p4
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0: continue
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            if ua < min_t: min_t = ua; hit = True
    return min_t if hit else None

def engelleri_rastgele_olustur(sayi):
    liste = []
    for _ in range(sayi):
        x = random.randint(300, SIM_GENISLIK + 200) 
        y = random.randint(50, EKRAN_Y - 100)
        w = random.randint(30, 60); h = random.randint(30, 60)
        hiz = random.uniform(-80, -20) if random.random() > 0.4 else 0
        liste.append(Engel(x, y, w, h, hiz_x=hiz))
    return liste

def simulasyonu_sifirla():
    global arac, engeller, sensorler
    arac = Arac(100, EKRAN_Y/2)
    engeller = engelleri_rastgele_olustur(12)
    sensorler = SensorPaketi(arac)

# PANEL ÇİZİM
def paneli_ciz(ekran, topsis_scores, yz_probs, final_scores, aksiyon, kayit_durumu, yz_surucu):
    panel_rect = pygame.Rect(SIM_GENISLIK, 0, PANEL_GENISLIK, EKRAN_Y)
    pygame.draw.rect(ekran, RENK_PANEL_BG, panel_rect)
    pygame.draw.line(ekran, (100, 100, 100), (SIM_GENISLIK, 0), (SIM_GENISLIK, EKRAN_Y), 3)

    x_off = SIM_GENISLIK + 20; y_off = 20
    ekran.blit(font_baslik.render("OTONOM KONTROL", True, RENK_ARAC), (x_off, y_off)); y_off += 40
    
    if yz_surucu.egitildi:
        mod_txt = "MOD: HİBRİT (LİDAR + HIZ + İVME)"
        mod_renk = (0, 255, 100)
    else:
        mod_txt = "MOD: Matematiksel (VERİ TOPLA)"
        mod_renk = (255, 200, 0)
    
    kayit_mesaj = "Veri Kaydı AÇIK ('K')" if kayit_durumu else "Veri Kaydı KAPALI ('K')"
    ekran.blit(font_veri.render(mod_txt, True, mod_renk), (x_off, y_off)); y_off += 25
    
    bilgiler = [f"[{kayit_mesaj}]", "'R': Reset | 'K': Kaydı Başlat/Durdur"]
    for b in bilgiler:
        ekran.blit(font_mini.render(b, True, RENK_YAZI), (x_off, y_off)); y_off += 15
    y_off += 15
    
    yz_durum = "YZ HAZIR" if yz_surucu.egitildi else "YZ EĞİTİLMEDİ"
    yz_renk = (0,255,0) if yz_surucu.egitildi else (200,50,50)
    ekran.blit(font_mini.render(yz_durum, True, yz_renk), (x_off, y_off)); y_off += 25

    ekran.blit(font_veri.render("[KARAR FÜZYONU DETAYI]", True, (255, 200, 0)), (x_off, y_off)); y_off += 25
    
    etiketler = AKSIYON_LISTESI
    best_idx = np.argmax(final_scores)
    col1_x = x_off; col2_x = x_off + 110; col3_x = x_off + 180; col4_x = x_off + 260
    
    ekran.blit(font_mini.render("EYLEM", True, (150,150,150)), (col1_x, y_off))
    ekran.blit(font_mini.render("TOPSIS", True, (255,200,0)), (col2_x, y_off))
    ekran.blit(font_mini.render("YZ(KNN)", True, (0,255,255)), (col3_x, y_off))
    ekran.blit(font_mini.render("SONUÇ", True, (0,255,100)), (col4_x, y_off)); y_off += 20
    
    for i in range(4):
        renk = (0, 255, 0) if i == best_idx else (180, 180, 180)
        ekran.blit(font_mini.render(etiketler[i], True, renk), (col1_x, y_off))
        
        t_val = f"%{topsis_scores[i]*100:.0f}"
        y_val = f"%{yz_probs[i]*100:.0f}"
        f_val = f"%{final_scores[i]*100:.0f}"
        
        ekran.blit(font_mini.render(t_val, True, (255,200,0)), (col2_x + 5, y_off))
        ekran.blit(font_mini.render(y_val, True, (0,255,255)), (col3_x + 5, y_off))
        
        bar_x = col4_x
        bar_bg = pygame.Rect(bar_x, y_off+2, 120, 10)
        pygame.draw.rect(ekran, (50,50,50), bar_bg)
        w = int(final_scores[i] * 120)
        bar_fg = pygame.Rect(bar_x, y_off+2, w, 10)
        pygame.draw.rect(ekran, renk, bar_fg)
        ekran.blit(font_mini.render(f_val, True, (255,255,255)), (bar_x + 125, y_off))
        y_off += 20

    y_off += 20
    ekran.blit(font_veri.render("SON KARAR:", True, (255, 255, 255)), (x_off, y_off)); y_off += 25
    karar_kutusu = pygame.Rect(x_off, y_off, 200, 40)
    renk_karar = (0, 100, 0)
    if aksiyon == "FREN": renk_karar = (200, 0, 0)
    elif "KAÇIN" in aksiyon: renk_karar = (200, 150, 0)
    pygame.draw.rect(ekran, renk_karar, karar_kutusu, 0, 5)
    ekran.blit(font_baslik.render(aksiyon, True, (255,255,255)), (x_off + 20, y_off + 10))

    mouse_pos = pygame.mouse.get_pos()
    buton_renk = (255, 140, 0) if BUTON_RECT.collidepoint(mouse_pos) else (200, 100, 0)
    pygame.draw.rect(ekran, buton_renk, BUTON_RECT, border_radius=10)
    text_yuzey = font_buton.render("SİMÜLASYONU YENİLE", True, (255, 255, 255))
    text_rect = text_yuzey.get_rect(center=BUTON_RECT.center)
    ekran.blit(text_yuzey, text_rect)

# GLOBAL
arac = None; engeller = None; sensorler = None
mcdm = MCDMKararVerici()
yz_surucu = YapayZekaSurucusu() 

simulasyonu_sifirla()
kayit_aktif = False 

#   ANA DÖNGÜ
calisiyor = True
aksiyon = "BEKLENİYOR"
duraklatildi = False
frame_sayac = 0
topsis_gosterim = np.zeros(4); yz_gosterim = np.zeros(4); final_gosterim = np.zeros(4)
lidar_cizim = []

while calisiyor:
    dt = saat.tick(FPS) / 1000.0
    frame_sayac +=1
    
    for olay in pygame.event.get(): 
        if olay.type == pygame.QUIT: calisiyor = False
        if olay.type == pygame.MOUSEBUTTONDOWN:
            if BUTON_RECT.collidepoint(olay.pos): simulasyonu_sifirla()

        if olay.type == pygame.KEYDOWN:
            if olay.key == pygame.K_r: simulasyonu_sifirla()
            elif olay.key == pygame.K_k: 
                kayit_aktif = not kayit_aktif
                print(f"Veri Kaydı: {kayit_aktif}")
            elif olay.key == pygame.K_p:
                duraklatildi = not duraklatildi
            
    if not duraklatildi:
        for e in engeller: e.adim(dt);
        if any(e.rect.right < 0 for e in engeller): simulasyonu_sifirla() 

        lidar_data = []; lidar_cizim = []
        for aci in LIDAR_ACILAR:
            gercek_aci = arac.yon + aci
            dx, dy = math.cos(gercek_aci), math.sin(gercek_aci)
            cx, cy = arac.x, arac.y
            min_k = 1.0
            for e in engeller:
                k = line_rect_collision(cx, cy, cx + dx*LIDAR_MESAFE, cy + dy*LIDAR_MESAFE, e.rect)
                if k and k < min_k: min_k = k
            mesafe = min_k * LIDAR_MESAFE
            lidar_data.append(mesafe)
            lidar_cizim.append((cx + dx*mesafe, cy + dy*mesafe))
        
        sens_veri = sensorler.veri_oku()
        
        # 1. TOPSIS
        dilim = len(lidar_data) // 3
        sol = np.min(lidar_data[:dilim]); orta = np.min(lidar_data[dilim:2*dilim]); sag = np.min(lidar_data[2*dilim:])
        m1 = [sol, arac.hiz*0.8, 4] ; m2 = [sag, arac.hiz*0.8, 4]
        # Fren yapmayı en güvenli liman (Güvenlik=250) olarak ayarladık
        m3 = [LIDAR_MESAFE, 0, 1] 
        m4 = [orta, MAKS_HIZ, 0]
        if orta < 60: m4[0] = 0; m4[2] = 100 # Engel çok yakınsa frene basacak
        
        topsis_scores = mcdm.topsis_hesapla([m1, m2, m3, m4])
        topsis_gosterim = topsis_scores

        # 2. YAPAY ZEKA (KNN)
        yz_probs = yz_surucu.olasiliklari_getir(lidar_data, arac.hiz, arac.ivme)
        yz_gosterim = yz_probs

        # 3. FÜZYON
        if yz_surucu.egitildi:
            final_scores = (topsis_scores * 0.6) + (yz_probs * 0.4)
        else:
            final_scores = topsis_scores
            
        # KAYIT
        if kayit_aktif and frame_sayac % 10 == 0:
            idx = np.argmax(final_scores)
            veri_kaydet(lidar_data, arac.hiz, arac.ivme, AKSIYON_LISTESI[idx])

        final_gosterim = final_scores
        best_idx = np.argmax(final_scores)
        aksiyon = AKSIYON_LISTESI[best_idx]

        # AKSİYON UYGULAMA
        if aksiyon == "SOLA_KAÇIN": 
            arac.yon -= 1.5 * dt; arac.ivme = -10
        elif aksiyon == "SAĞA_KAÇIN": 
            arac.yon += 1.5 * dt; arac.ivme = -10
        elif aksiyon == "FREN": 
            arac.ivme = -150
        elif aksiyon == "SÜRDÜR":
            # Adaptif Hız Kontrolü (Önümde engel varsa gazı kes)
            if orta > 200:
                arac.ivme = 50   # Yol temiz, tam gaz
            elif orta > 120:
                arac.ivme = 0    # bu kısımda engel var. gazı kes
            else:
                arac.ivme = -30  # Yaklaştım, hafif fren yap
                
            # Aracı şeritte tut
            arac.yon += (0 - arac.yon) * 0.05
        
        arac.adim(dt)

    # Çizim
    ekran.fill(RENK_ZEMIN)
    for x in range(0, SIM_GENISLIK, 50): pygame.draw.line(ekran, RENK_GRID, (x, 0), (x, EKRAN_Y))
    for y in range(0, EKRAN_Y, 50): pygame.draw.line(ekran, RENK_GRID, (0, y), (SIM_GENISLIK, y))
    
    if lidar_cizim:
        for p in lidar_cizim:
            pygame.draw.line(ekran, (0, 40, 0), (arac.x, arac.y), p, 1)
            if math.sqrt((p[0]-arac.x)**2 + (p[1]-arac.y)**2) < LIDAR_MESAFE - 5:
                pygame.draw.circle(ekran, RENK_LIDAR, (int(p[0]), int(p[1])), 2)
            
    arac.ciz(ekran)
    for e in engeller: e.ciz(ekran)
    paneli_ciz(ekran, topsis_gosterim, yz_gosterim, final_gosterim, aksiyon, kayit_aktif, yz_surucu)
    
    if duraklatildi:
        overlay = pygame.Surface((SIM_GENISLIK, EKRAN_Y), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128)) 
        ekran.blit(overlay, (0, 0))
        txt = font_baslik.render("SİMÜLASYON DURDURULDU (Devam için 'P')", True, (255, 255, 255))
        txt_rect = txt.get_rect(center=(SIM_GENISLIK/2, EKRAN_Y/2))
        ekran.blit(txt, txt_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()