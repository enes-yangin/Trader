# TraderAI Kapsamlı Mantık Hataları (Bug) Raporu

Bu rapor, TraderAI kod tabanında yapılan derin inceleme sonucu tespit edilen hataları ve mevcut durumlarını listeler. 

---

## 1. Düzeltilmiş Mantık Hataları (Resolved Bugs Log)
*Önceki oturumlarda tespit edilerek başarıyla düzeltilmiş ve unit testleri ile kilitlenmiş hatalar:*

### Veri ve Öznitelik Hattı (data/)
*   **BUG-01 (`news_fetcher.py` L29):** UTC zaman damgası normalize edilirken `tz_localize(None)` kullanılıyordu. Bu durum halihazırda timezone-aware olan zaman damgalarında `TypeError` fırlatıyordu. `tz_convert(None)` kullanılarak düzeltildi.
*   **BUG-02 (`news_features.py` L46):** Haber öznitelikleriyle fiyat verisi merge edildiğinde fiyat DataFrame'inin `DatetimeIndex`'i sıfırlanıp yerine integer `RangeIndex` geliyordu. Merge öncesinde index yedeklenerek sonrasında geri yüklenecek şekilde düzeltildi.
*   **BUG-03 (`labeling.py` L29-39):** `atr_pct` içindeki `NaN` değerler boolean maskelemeyi bozarak dizin hatalarına veya yanlış sınıflandırmalara neden oluyordu. Maske serilerine `.fillna(False)` eklenerek düzeltildi. `threshold=0` iken HOLD sınıfının hiç oluşmaması sorunu giderildi.
*   **BUG-15a & BUG-15b (`news_fetcher.py` L129, `fetcher.py` L54):** `np.random.seed()` kullanımı global rastgelelik durumunu sıfırlayarak model eğitimindeki shuffle/validation adımlarını bozuyordu. Yerel `np.random.default_rng()` ile izole edilerek düzeltildi.
*   **BUG-17 (`smoothing.py` L58-63):** `add_smoothing_features` fonksiyonu gelen DataFrame'i kopyalamadan (in-place) doğrudan mutasyona uğratıyordu. `df = df.copy()` eklenerek state mutation engellendi.
*   **BUG-25 (`orderbook.py` L137-138):** Sentetik emir defteri üretiminde `skew > 1.0` olduğunda negatif ask/bid hacimleri oluşuyordu. Hacimler `max(volume, 0)` ile sınırlandırıldı.
*   **BUG-26 (`dataset.py` L48-51):** `ensure` fonksiyonu önbellekteki verinin sadece varlığına bakıyor, güncelliğini kontrol etmiyordu. Freshness check doğrulaması eklendi.

### Hesaplama ve Algoritma Hattı (engine/)
*   **BUG-04 (`ev_optimization.py` L259-261):** Beklenen Değer (EV) hesaplamasındaki işlem bazlı getiriler (per-trade returns) gerçek işlemler yerine ortalamalardan (win/loss ratio) yapay olarak üretiliyor, bu da varyansın aşırı düşük ve Sharpe oranının sahte düzeyde yüksek görünmesine neden oluyordu. Gerçek işlem bazlı seriye geçildi veya kısıtlamalar belgelendi.
*   **BUG-07 (`backtester.py` L178):** Sharpe oranı `np.sqrt(252)` (geleneksel borsa çalışma gün sayısı) ile yıllıklaştırılıyordu. Kripto piyasaları 7/24 ve 365 gün açık olduğu için bu oran `np.sqrt(365)` olarak düzeltildi.
*   **BUG-08 (`cpcv.py` L184):** CPCV modülündeki Sharpe oranı yıllıklaştırılmıyordu, bu da backtest raporlarındaki Sharpe ile uyuşmazlığa neden oluyordu. Yıllıklaştırma çarpanı eklenerek senkronize edildi.
*   **BUG-09 (`backtester.py` L123):** "CLOSE" (zorunlu kapanış) işlemleri PnL listesinden `pop()` edilmesine rağmen `trades` listesinde kalıyor ve win-rate metriklerine dahil ediliyordu. CLOSE işlemlerinin metriklerden de temizlenmesi veya tutarlı sayılması sağlandı.
*   **BUG-12a & BUG-12b (`cpcv.py` L283, `walkforward.py` L107):** LSTM parametre kontrolü case-sensitive `model_name == "lstm"` şeklinde yapılıyordu. Kullanıcı "LSTM" veya "Lstm" yazdığında epoch konfigürasyonu yüklenmiyordu. `.lower() == "lstm"` olarak düzeltildi.
*   **BUG-13 & BUG-13b (`cpcv.py` L179, `walkforward.py` L75):** Yönsel doğruluk (directional accuracy) hesaplanırken `sign(0) == sign(0)` karşılaştırması `True` dönerek sıfır getirili/tahminli barları başarıymış gibi sayıyordu. Zeros maskelemesi yapılarak sadece gerçek yönler hesaba katılacak şekilde düzeltildi.
*   **BUG-20 (`risk.py` L13):** Kelly hesaplamasında sıfır PnL'li nötr işlemler kayıp (loss) listesine ekleniyor, bu da Kelly oranını yapay olarak aşağı çekiyordu. Kayıplarstrictly `p < 0` olarak süzülerek düzeltildi.
*   **BUG-21 (`statarb_signals.py` L101-115):** Spread stratejisinde açık kalan pozisyonların son barındaki fiyat hareketi (PnL) hesaba katılmadan pozisyon sıfır pnl ile kapatılıyordu. Son bar farkı entegre edildi.

### Modeller ve Sınıflandırıcılar (models/)
*   **BUG-05 (`base_model.py` L72,75,78):** Dynamic threshold (dinamik eşik) kullanıldığında ATR'nin sıfır veya sıfıra çok yakın olması durumunda `signal()` fonksiyonunda **sıfıra bölme (ZeroDivisionError)** hatası oluşuyordu. Güvenlik epsilou (`1e-9`) eklenerek düzeltildi.
*   **BUG-10 (`base_model.py` L78):** HOLD güven (confidence) hesabı negatif tahminlerde `sell_th` yerine hep asimetrik olarak `buy_th` kullanıyordu. Tahminin işaretine göre eşik dinamik seçilecek şekilde düzeltildi.
*   **BUG-11 (`base_classifier.py` L43-44):** Tahmin ve etiket serilerindeki boyut uyumsuzlukları sessizce kırpılarak gizli look-ahead sızıntılarına veya hizalama kaymalarına yol açıyordu. Boyut uyuşmazlığında log uyarısı fırlatacak şekilde düzeltildi.
*   **BUG-19 (`base_classifier.py` L63, `classifier_models.py` L94):** Sınıf tahmin olasılık hizalamalarında model sınıfları doğrudan indeks olarak kullanılıyordu. `ALL_LABELS.index(int(cls))` eşlemesiyle daha robust hale getirildi.

### Arayüz ve Portföy (ui/, portfolio/, utils/)
*   **BUG-06 (`ui/app.py` L270-272):** Model eğitimi bittikten sonra `self.bundle` set ediliyor ve sembol değiştirilse dahi arayüzdeki canlı grafik güncellemeleri kilitlenip kalıyordu. Sembol değişiminde bundle sıfırlanarak canlandırıldı.
*   **BUG-18 (`ui/app.py` L253):** Arayüzün can damarı olan periyodik polling tik zamanlayıcısı (`_live_job`) uygulama kapatıldığında iptal edilmiyordu. `after_cancel` eklenerek kaynak sızıntısı giderildi.
*   **BUG-14 (`ui/chart.py` L92-93):** RSI grafik ölçeklendirmesinde kullanılan `1.5` eşiği aşırı satım piyasalarında RSI'ı yanlışlıkla 100x çarpıyordu. Eşik `1.0` olarak düzeltildi.
*   **BUG-16 (`utils/types.py` L80-81):** `__contains__` metodu field değeri `None` olduğunda hatalı `False` dönerek dict yapısını bozuyordu. Sadece field varlığını kontrol edecek şekilde düzeltildi.
*   **BUG-23 (`portfolio/position.py` L60):** `cost_basis <= 0` kontrolü negatif maliyetli komisyon iadelerini (rebates) yutuyordu. `cost_basis == 0` olarak güncellendi.
*   **BUG-27 (`portfolio/pnl.py` L38):** Floating-point duyarlılığı nedeniyle çok küçük pozisyon miktarlarında sıfıra bölme hatası oluşabiliyordu. Minimum tolerans (`1e-12`) sınırı eklenerek düzeltildi.

---

## 2. Aktif Mimari Zafiyetler ve Bekleyen Hatalar (Pending Bugs & Critical Issues)
*Halihazırda plan.md içerisinde düzeltilmesi planlanan ve kod tabanında aktif olan zafiyetler:*

### ⚠️ Kritik Entegrasyon ve Sızıntı Zafiyetleri (High Priority)
1.  **Doğrulama Kalkanının Bypass Edilmesi (A1):**
    *   *Açıklama:* Projenin en güçlü yanı olan CPCV, Purging ve Deflated-p kalkanları, UI üzerindeki gerçek "Train & Predict" döngüsünde tamamen bypass edilmektedir. UI, naif kronolojik bölme yapan `engine/trainer.py` içindeki `split()` metodunu kullanır.
    *   *Etki:* Model eğitim metrikleri aşırı iyimser ve sızıntılı gösterilirken, doğrulama kalkanı bambaşka bir dünyada çalışmaktadır.
2.  **`split()` Metodunda Purging Eksikliği (B6 & A1):**
    *   *Açıklama:* `engine/trainer.py` içindeki naif `split()` metodu, train-val ve val-test sınırlarında herhangi bir purging veya embargo uygulamamaktadır.
    *   *Etki:* İleriye dönük etiketlerin (`pred_horizon` adımlı) sınır barlarında kesişmesi nedeniyle model gelecekteki verileri eğitimde sızdırır (Look-ahead bias).
3.  **UI/Trainer FeatureSpec Tutarsızlığı (B5 & A2):**
    *   *Açıklama:* UI feature spec jenerasyonu 5 aileyi etkinleştirirken, `engine/trainer.py` içindeki `_resolve_spec` yalnızca 3 aileyi (`news`, `micro`, `cross_asset`) filtrelemektedir. 
    *   *Etki:* Gelişmiş özellik grupları (`orderbook`, `macro`, `social`) model eğitimine sessizce katılamamakta veya CPCV ile UI'ın eğittiği özellikler tamamen uyumsuz kalmaktadır.

### ⚠️ Mantıksal ve İşlevsel Hatalar (Medium Priority)
4.  **Sessiz Jeneratörlerin Sentetik Veri Uydurması (B2 & A3):**
    *   *Açıklama:* `orderbook_features.py`, `social_fetcher.py` ve `event_features.py` modülleri gerçek veri havuzu boş olduğunda kullanıcıyı uyarmadan veya durmadan otomatik olarak sentetik/mock veri üretip modele beslemektedir.
    *   *Etki:* Model gerçek verileri analiz ettiğini sanırken aslında saf matematiksel gürültüyü ezberlemekte (overfitting) ve gerçek hayatta çalışmayacak sinyaller üretmektedir.
5.  **Güvenlik Buffer Fiyatı Hesaplama Hatası (B3):**
    *   *Açıklama:* `engine/risk.py:237` altındaki `liquidation_buffer_price` formülü, likidasyon fiyatının hemen üzerinde durdurma emri vermesi gerekirken, entry fiyatı ile likidasyon fiyatının orta noktasını hesaplamaktadır.
    *   *Etki:* Kaldıraçlı backtestlerde pozisyonlar aşırı erken kapatılmakta ve strateji performansı manipüle edilmektedir.
6.  **Ölü Risk Yönetim Fonksiyonları ve Equity Hataları (B4 & A4):**
    *   *Açıklama:* `apply_liquidation_to_equity` fonksiyonunda hesap equity değeri (örn. 10,000) doğrudan fiyat seviyesiyle (örn. 90) karşılaştırılmaktadır. Ayrıca bu fonksiyon ve `maintenance_margin` modülden hiçbir yerde çağrılmamaktadır (ölü kod).

### ⚠️ Sistem ve Yapılandırma Sorunları (Low Priority)
7.  **Determinizm Zafiyeti (A6):**
    *   *Açıklama:* Jeneratörlerde seed hesaplanırken kullanılan `hash(symbol)` fonksiyonu, Python'ın `PYTHONHASHSEED` rastgeleliği nedeniyle her yeni process başlatıldığında farklı bir seed üretir.
    *   *Etki:* Çalıştırmalar arasında determinizm kaybolur, özellikler tekrarlanamaz.
8.  **Reference Cache Temizlenmeme Sorunu (B7):**
    *   *Açıklama:* `reference_series.py` önbelleği uygulama oturumu boyunca temizlenmez. İlk istekteki geçici bir ağ hatasında önbelleğe alınan boş değer, ağ geri gelse bile oturum kapanana kadar boş dönmeye devam eder.
