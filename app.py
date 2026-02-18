import streamlit as st
import streamlit_authenticator as stauth
import yaml
import time
from yaml.loader import SafeLoader

# --- KENDİ MODÜLLERİMİZ ---
from src.rag_engine import RAGEngine
from src.ste100_guard import STE100Guard
from src.utils import load_config

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="STE100 Technical Assistant",
    page_icon=" ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. BAŞLATMA VE ÖNBELLEK ---
@st.cache_resource
def get_engine():
    """RAGEngine ağır bir sınıftır, sadece bir kere yüklenir."""
    return RAGEngine()

@st.cache_resource
def get_guard():
    """STE100Guard hafif ama sürekli regex derlemesin diye cache'liyoruz."""
    return STE100Guard()

# Global nesneleri çağır
engine = get_engine()
guard = get_guard()
config = load_config()

# Koleksiyon adı (Admin ingest sırasında belirlenen ad)
COLLECTION_NAME = "doc_kaggle_v1"

# --- 3. KİMLİK DOĞRULAMA (Mevcut yapıyı koruyoruz) ---
def load_auth():
    with open('config/secrets.yaml') as file:
        config_auth = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config_auth['credentials'],
        config_auth['cookie']['name'],
        config_auth['cookie']['key'],
        config_auth['cookie']['expiry_days'],
        # preauthorized=config_auth['preauthorized'] # Versiyon farkına göre gerekebilir/gerekmeyebilir
    )
    return authenticator

authenticator = load_auth()

# Giriş Widget'ı (Streamlit sürümüne göre değişiklik gösterebilir, en güvenli yöntem)
try:
    name, authentication_status, username = authenticator.login("main")
except:
    name, authentication_status, username = authenticator.login()

# --- 4. ANA UYGULAMA MANTIĞI ---

if authentication_status:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/6024/6024190.png", width=50)
        st.title("Admin Panel")
        st.write(f"Hoşgeldin, **{name}**")
        st.divider()
        
        # Ayarlar
        st.subheader(" Sistem Durumu")
        st.success(" RAG Motoru: Aktif")
        st.success(" STE100 Denetimi: Aktif")
        st.info(f"Veri Seti: {COLLECTION_NAME}")
        
        if st.button("Sohbeti Temizle", type="primary"):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        authenticator.logout('Çıkış Yap', 'sidebar')

    # --- ANA EKRAN ---
    st.title("🛠️ In-House Technical Support AI")
    st.caption("Simplified Technical English (ASD-STE100) Compliance Enforced")

    # Sohbet Geçmişi Başlatma
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş Mesajları Göster
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # Eğer mesajda görsel varsa göster
            if "image" in msg:
                st.image(msg["image"], caption="Uploaded Image", width=300)
            
            st.markdown(msg["content"])
            
            # Eğer bu bir asistan cevabıysa ve içinde rapor varsa (context/warnings) expander ile göster
            if msg.get("is_report", False):
                status_color = "red" if msg.get("warnings") else "green"
                status_icon = "Uyarı" if msg.get("warnings") else "Sorunsuz"
                status_title = "STE100 Uyarıları Mevcut" if msg.get("warnings") else "STE100 Uyumlu"
                
                with st.expander(f"{status_icon} Teknik Rapor & Kaynaklar ({status_title})"):
                    # 1. STE100 Uyarıları
                    if msg.get("warnings"):
                        st.error("Tespit Edilen İhlaller:")
                        for w in msg["warnings"]:
                            st.write(f"- {w}")
                    else:
                        st.success("Bu cevap STE100 kelime standartlarına uygundur.")
                    
                    # 2. Kullanılan Context
                    st.divider()
                    st.markdown("**Referans Alınan Teknik Döküman:**")
                    st.info(msg.get("context_text", "Context bulunamadı."))

    # --- KULLANICI GİRİŞİ ---
    if prompt := st.chat_input("Teknik sorunuzu buraya yazın..."):
        # 1. Kullanıcı Mesajını Ekrana Bas
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Asistan Cevabını Üret
        with st.chat_message("assistant"):
            # UI Animasyonu
            status_container = st.status("Analiz ediliyor...", expanded=True)
            
            try:
                # A) Arama ve Üretim (YAML Promptlar devrede)
                status_container.write("Teknik dökümanlar taranıyor (Hybrid Search)...")
                response_text, context_text = engine.search_and_answer(
                    query=prompt, 
                    collection_name=COLLECTION_NAME,
                    history=st.session_state.messages
                )
                
                # B) STE100 Denetimi (Guard devrede)
                status_container.write("STE100 uyumluluk kontrolü yapılıyor...")
                warnings = guard.check_compliance(response_text)
                
                status_container.update(label="Tamamlandı", state="complete", expanded=False)

                # C) Cevabı Yazdır
                st.markdown(response_text)
                
                # D) Rapor Alanı (Expander)
                # İkon ve renk belirleme
                has_warning = len(warnings) > 0
                expander_title = "STE100 Denetim Raporu (İhlal Var)" if has_warning else "Teknik Rapor (Uyumlu)"
                
                with st.expander(expander_title, expanded=has_warning):
                    if has_warning:
                        st.error(f"Bu cevapta {len(warnings)} adet STE100 ihlali tespit edildi:")
                        for w in warnings:
                            st.markdown(f" {w}")
                    else:
                        st.success("Kelime kullanımı ASD-STE100 sözlüğüne uygundur.")
                        
                    st.markdown("---")
                    st.caption("**Kullanılan Bağlam (Context):**")
                    st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)

                # 3. Geçmişe Kaydet
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "is_report": True,
                    "warnings": warnings,
                    "context_text": context_text
                })

            except Exception as e:
                status_container.update(label="Hata", state="error")
                st.error(f"Bir hata oluştu: {str(e)}")

elif authentication_status is False:
    st.error('Kullanıcı adı veya şifre hatalı.')
elif authentication_status is None:
    st.warning('Lütfen giriş yapınız.')