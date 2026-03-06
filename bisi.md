STE100 RAG Sistemi Kurulum ve Çalıştırma Kılavuzu
Bu kılavuz, asenkron API mimarisine (Frontend - Backend ayrımı) geçirilmiş kurum içi (in-house) RAG sisteminin nasıl çalıştırılacağını açıklar. Sistem arayüzü kilitlenmeleri önlemek amacıyla model sunucusu ve kullanıcı arayüzü olarak iki ayrı süreç (process) halinde çalıştırılmalıdır.

Adım 1: Yapay Zeka Sunucusunu (Backend) Başlatma
Sistemin ana görüntü ve dil modelini (Qwen-VL) arayüzden bağımsız bir API sunucusu olarak ayağa kaldırmanız gerekmektedir. Bu işlem için vLLM motorunu kullanabilirsiniz.

Terminalinizi açın ve model sunucusunu 8000 portunda başlatın:

Bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000 --max-model-len 4096
Önemli Not: Sunucu aktifleştiğinde - adresinden dinleme yapmaya başlayacaktır. Sistemin settings.yaml dosyasındaki api_base_url parametresinin bu adresle eşleştiğinden emin olun.

Adım 2: Gerekli Bağımlılıkların Yüklenmesi
Eğer henüz yüklemediyseniz, yeni eklenen asenkron iletişim kütüphanesi dahil olmak üzere tüm gereksinimleri projenin sanal ortamına (virtual environment) yükleyin:

Bash
pip install -r requirements.txt
pip install aiohttp
Adım 3: Kullanıcı Arayüzünü (Frontend) Başlatma
Model sunucusu arka planda başarıyla çalışmaya başladıktan ve "Uvicorn running on -" benzeri bir log gördükten sonra, yeni bir terminal penceresi açın. Ana dizinde aşağıdaki komutu çalıştırarak Streamlit uygulamasını başlatın:

Bash
streamlit run app.py
Adım 4: Uygulamaya Erişim (Şirket İçi Kullanım)
Streamlit uygulaması ayağa kalktığında terminalde erişim adresleri belirecektir.

Yerel Erişim: Uygulamayı çalıştırdığınız bilgisayarda (örneğin dizüstü bilgisayarınızda) tarayıcınızı açıp - adresine giderek sistemi kullanabilirsiniz.

Ağ Üzerinden Erişim: Aynı ağa (Wi-Fi veya yerel şirket ağı) bağlı olan diğer ekip arkadaşlarınızın sisteme girebilmesi için tarayıcılarına uygulamanın çalıştığı cihazın yerel IP adresini ve port numarasını yazmaları gerekir.