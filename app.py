import streamlit as st
import streamlit_authenticator as stauth
from src.utils import load_config, load_secrets
from src.rag_engine import RAGEngine
from src.ste100_guard import STE100Guard
from src.database import DatabaseManager
import time

# --- 1. AYARLAR VE GÜVENLİK ---
st.set_page_config(page_title="AI Asistan", layout="wide")
cfg = load_config()
secrets = load_secrets()

# --- KRİTİK OPTİMİZASYON: Modeli Önbelleğe Al ---
# Bu sayede her soruda modelleri tekrar yüklemez, çökme engellenir.
@st.cache_resource
def get_rag_engine():
    return RAGEngine()

# Oturum Durumu (Session State) Başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None

# Giriş Sistemi
authenticator = stauth.Authenticate(
    secrets['credentials'],
    secrets['cookie']['name'],
    secrets['cookie']['key'],
    secrets['cookie']['expiry_days']
)

try:
    authenticator.login()
except Exception as e:
    st.error(f"Sistem Hatası: {e}")

# --- 2. UYGULAMA AKIŞI ---
if st.session_state["authentication_status"]:
    
    # --- SIDEBAR (SOL MENÜ) ---
    with st.sidebar:
        st.write(f"**{st.session_state['name']}**")
        authenticator.logout('Çıkış Yap', 'sidebar')
        st.divider()
        
        st.header("Döküman Seçimi")
        db = DatabaseManager()
        cols = db.list_collections()
        
        if cols:
            selected = st.selectbox("Çalışılacak Döküman:", cols, index=None, placeholder="Seçiniz...")
            if selected:
                st.session_state.selected_collection = selected
                st.success(f"Aktif: {selected}")
        else:
            st.warning("Sistemde yüklü döküman yok. Admin ile görüşün.")
            
        st.divider()
        st.caption(f"Sistem Modu: {'MOCK' if cfg['system']['use_mock_llm'] else 'PRODUCTION'}")

    # --- ANA EKRAN ---
    st.title(f"{cfg['app']['name']}")

    if not st.session_state.selected_collection:
        st.info("Başlamak için lütfen sol menüden bir döküman seçiniz.")
    else:
        # Geçmiş Mesajları Göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Yeni Soru Girişi
        if prompt := st.chat_input("Sorunuzu buraya yazın..."):
            # 1. Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. AI Cevabı Hazırlanıyor
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("*Döküman taranıyor ve cevap üretiliyor...*")
                
                try:
                    # Motoru önbellekten al (Hız kazandırır ve çöküşü önler)
                    engine = get_rag_engine()
                    guard = STE100Guard()
                    
                    # Arama ve Cevaplama (History Gönderiliyor)
                    # used_context artık geçmişe saklanmak üzere geri alınıyor.
                    raw_response, used_context = engine.search_and_answer(
                        prompt, 
                        st.session_state.selected_collection,
                        history=st.session_state.messages
                    )
                    
                    # STE100 Denetimi
                    warnings = guard.check_compliance(raw_response)
                    final_response = raw_response
                    
                    # Cevabı Göster
                    message_placeholder.markdown(final_response)
                    
                    if warnings:
                        with st.expander("⚠️ STE100 Uyumluluk Raporu"):
                            for w in warnings:
                                st.write(w)
                    
                    # 3. Geçmişe Ekle (Context Metadata ile Birlikte)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": final_response,
                        "context": used_context # Bir sonraki tur 'tekrar bak' denirse kullanılacak
                    })

                except Exception as e:
                    st.error(f"İşlem sırasında bir hata oluştu: {e}")
                    st.info("Lütfen terminali kontrol edin veya sistemi yeniden başlatın.")

elif st.session_state["authentication_status"] is False:
    st.error('Kullanıcı adı veya şifre hatalı.')
elif st.session_state["authentication_status"] is None:
    st.warning('Lütfen giriş yapınız.')