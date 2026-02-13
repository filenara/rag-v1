import streamlit as st
import streamlit_authenticator as stauth
import yaml
import json
import time
from datetime import datetime
from yaml.loader import SafeLoader

# --- Kendi Modüllerimiz ---
from src.rag_engine import RAGEngine
from src.ste100_guard import STE100Guard
from src.database import DatabaseManager

# --- AYARLAR ---
PAGE_TITLE = "Kurumsal AI Asistan"
PAGE_ICON = "🤖"
HISTORY_LIMIT = 6  # Modelin göreceği son mesaj sayısı (3 Soru + 3 Cevap)

st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")

# --- YARDIMCI FONKSİYONLAR ---

def load_config():
    with open('config/settings.yaml') as file:
        return yaml.load(file, Loader=SafeLoader)

def load_secrets():
    with open('config/secrets.yaml') as file:
        return yaml.load(file, Loader=SafeLoader)

@st.cache_resource
def get_rag_engine():
    """RAGEngine'i bir kere başlatır, cache'ler."""
    return RAGEngine()

def download_chat_history():
    """Sohbet geçmişini JSON olarak indirilebilir hale getirir."""
    chat_data = json.dumps(st.session_state.messages, indent=4, ensure_ascii=False)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="📥 Sohbeti İndir (JSON)",
        data=chat_data,
        file_name=f"chat_history_{timestamp}.json",
        mime="application/json"
    )

def reset_chat():
    """Sohbeti güvenli bir şekilde sıfırlar."""
    st.session_state.messages = []
    st.session_state.context_memory = [] # RAG Engine için teknik bağlam
    st.rerun()

# --- BAŞLANGIÇ AYARLARI (SESSION STATE) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

# --- GÜVENLİK VE GİRİŞ ---
secrets = load_secrets()
authenticator = stauth.Authenticate(
    secrets['credentials'],
    secrets['cookie']['name'],
    secrets['cookie']['key'],
    secrets['cookie']['expiry_days']
)

try:
    authenticator.login()
except Exception as e:
    st.error(f"Giriş Modülü Hatası: {e}")

# --- ANA UYGULAMA ---
if st.session_state["authentication_status"]:
    
    # --- SIDEBAR (SOL MENÜ) ---
    with st.sidebar:
        st.title(f"{PAGE_ICON} Kontrol Paneli")
        st.write(f"Kullanıcı: **{st.session_state['name']}**")
        authenticator.logout('Çıkış Yap', 'sidebar')
        st.divider()
        
        # 1. Döküman Seçimi
        st.subheader("📚 Bilgi Bankası")
        db = DatabaseManager()
        cols = db.list_collections()
        
        if cols:
            selected = st.selectbox(
                "Aktif Döküman Seti:", 
                cols, 
                index=None, 
                placeholder="Bir kaynak seçiniz..."
            )
            if selected:
                st.session_state.selected_collection = selected
                st.success(f"Bağlı: {selected}")
        else:
            st.warning("Sistemde yüklü döküman bulunamadı.")
            
        st.divider()

        # 2. Sohbet Yönetimi (Production Level Eklenti)
        st.subheader("🛠️ Sohbet Araçları")
        if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
            reset_chat()
            
        if st.session_state.messages:
            download_chat_history()

        st.caption("v1.2.0 - In-House Production")

    # --- ANA EKRAN (CHAT ARAYÜZÜ) ---
    st.title(PAGE_TITLE)

    if not st.session_state.selected_collection:
        st.info("👋 Başlamak için lütfen sol menüden çalışmak istediğiniz döküman setini seçiniz.")
    else:
        # 1. Geçmiş Mesajları Ekrana Bas
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Eğer mesajda STE100 uyarısı veya kaynak varsa expander ile gösterilebilir (Opsiyonel)

        # 2. Yeni Kullanıcı Girişi
        if prompt := st.chat_input("Teknik sorunuzu buraya yazın..."):
            
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # AI Cevabı Hazırlanıyor
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_placeholder = st.status("Analiz ediliyor...", expanded=True)
                
                try:
                    status_placeholder.write("🔍 Dökümanlar taranıyor...")
                    engine = get_rag_engine()
                    guard = STE100Guard()
                    
                    # --- PRODUCTION CRITICAL: SLIDING WINDOW ---
                    # Tüm geçmişi değil, sadece son N mesajı gönderiyoruz.
                    # Bu, modelin (Qwen) "context length exceeded" hatası vermesini engeller.
                    recent_history = st.session_state.messages[-HISTORY_LIMIT:]
                    
                    status_placeholder.write("🤖 Cevap üretiliyor...")
                    
                    # RAGEngine'e sınırlı geçmişi gönder
                    raw_response, used_context = engine.search_and_answer(
                        prompt, 
                        st.session_state.selected_collection,
                        history=recent_history
                    )
                    
                    # STE100 Denetimi
                    warnings = guard.check_compliance(raw_response)
                    
                    status_placeholder.update(label="Tamamlandı!", state="complete", expanded=False)
                    
                    # Cevabı Göster
                    message_placeholder.markdown(raw_response)
                    
                    # Kaynak ve Uyarıları Göster
                    if used_context or warnings:
                        with st.expander("📝 Kaynaklar ve Teknik Denetim"):
                            if warnings:
                                st.warning("STE100 İhlalleri:")
                                for w in warnings:
                                    st.write(f"- {w}")
                            
                            st.markdown("**Kullanılan Bağlam:**")
                            st.caption(used_context[:500] + "..." if len(used_context) > 500 else used_context)
                    
                    # Geçmişe Kaydet
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": raw_response,
                        "context": used_context # İleride "buna tekrar bak" denirse kullanılacak
                    })

                except Exception as e:
                    status_placeholder.update(label="Hata Oluştu", state="error")
                    st.error(f"Sistem Hatası: {e}")
                    # Hata loglaması için buraya logging eklenebilir

elif st.session_state["authentication_status"] is False:
    st.error('Kullanıcı adı veya şifre hatalı.')
elif st.session_state["authentication_status"] is None:
    st.warning('Lütfen giriş yapınız.')