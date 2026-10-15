# Premier League AI

Geçmiş Premier League maçlarından öğrenen modellerle 2026/27 sezonunu maç maç simüle eden bir Python projesi. Takım güçlerini Elo puanlarıyla, maç skorlarını Poisson regresyonuyla tahmin eder; her çalıştırmada farklı bir alternatif sezon senaryosu, puan tablosu ve oyuncu istatistikleri üretir.

## Özellikler

- **Veri toplama:** Geçmiş sezonlar [premier-league-data](https://github.com/AnishKhetani/premier-league-data) paketinden, güncel sezon, Championship maçları ve oyuncu kadroları [football-data.org](https://www.football-data.org/) API'sinden çekilir
- **Elo puanları:** Takım güçleri maç sonuçlarına göre güncellenir; ev sahibi avantajı ve lige yükselen takımlar için ayrı başlangıç puanları kullanılır
- **Özellik çıkarımı:** Son maçlardaki form, atılan/yenilen gol ortalamaları ve Elo farkı gibi maç öncesi özellikler
- **Modeller (scikit-learn):**
  - Maç sonucu (galibiyet/beraberlik/mağlubiyet) için Logistic Regression
  - Ev sahibi ve deplasman gol sayıları için Poisson Regression
- **Sezon simülatörü:** Kalan maçları tahmin edilen gol beklentileriyle simüle eder; formu ve Elo puanlarını sezon boyunca dinamik olarak günceller, şampiyonu ve küme düşenleri belirler
- **Oyuncu simülatörü:** Maç kadrolarını, oynanan dakikaları, golleri ve asistleri oyunculara dağıtarak gol krallığı gibi istatistikler üretir
- **Tekrarlanabilir senaryolar:** Simülasyonlara bir `seed` verilerek aynı sezon senaryosu tekrar üretilebilir

## Kullanılan Teknolojiler

- Python 3.12
- pandas, NumPy
- scikit-learn
- requests, BeautifulSoup
- python-dotenv

## Proje Yapısı

```text
.
├── data/
│   ├── raw/                 # API önbelleği (git'e eklenmez)
│   └── processed/           # İşlenmiş maç özellikleri (match_features.csv)
├── src/
│   ├── data_loader.py       # Geçmiş ve güncel sezon maç verileri
│   ├── standings.py         # Puan tablosu hesaplama
│   ├── elo.py               # Elo puanı hesaplama
│   ├── features.py          # Maç özellik veri setini oluşturma
│   ├── model.py             # Maç sonucu modeli (Logistic Regression)
│   ├── goal_model.py        # Gol modeli (Poisson Regression)
│   ├── simulator.py         # Sezon simülatörü
│   ├── player_data.py       # Oyuncu kadroları ve gol krallığı verileri
│   ├── player_simulator.py  # Oyuncu bazlı sezon simülatörü
│   └── api_test.py          # API bağlantı testi
├── requirements.txt
└── .env.example
```

## Kurulum

```bash
git clone https://github.com/yavuzhan642/premier-league-ai.git
cd premier-league-ai

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[football-data.org](https://www.football-data.org/client/register) üzerinden ücretsiz bir API anahtarı alın ve `.env` dosyasına yazın:

```bash
cp .env.example .env
# .env dosyasını açıp FOOTBALL_DATA_API_KEY değerini girin
```

API bağlantısını test etmek için:

```bash
python src/api_test.py
```

## Kullanım

Komutları proje ana klasöründen çalıştırın:

```bash
# Maç özellik veri setini oluştur
python src/features.py

# Modelleri eğit ve değerlendir
python src/model.py
python src/goal_model.py

# Güncel Elo puanlarını göster
python src/elo.py

# 2026/27 sezonunu simüle et (seed vermek isteğe bağlıdır)
python src/simulator.py
python src/simulator.py 42

# Oyuncu istatistikleriyle birlikte simüle et
python src/player_simulator.py
```

## Not

Simülasyon sonuçları istatistiksel tahminlerdir; gerçek maç sonuçlarını öngörme iddiası taşımaz.
