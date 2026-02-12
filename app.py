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
        st.write(f"👤 **{st.session_state['name']}**")
        authenticator.logout('Çıkış Yap', 'sidebar')
        st.divider()
        
        st.header("📂 Döküman Seçimi")
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
        st.caption(f"Sistem Modu: {'🛠️ MOCK' if cfg['system']['use_mock_llm'] else '🟢 PRODUCTION'}")

    # --- ANA EKRAN ---
    st.title(f"🚀 {cfg['app']['name']}")

    # Eğer döküman seçilmediyse uyarı ver
    if not st.session_state.selected_collection:
        st.info("👋 Başlamak için lütfen sol menüden bir döküman seçiniz.")
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
                message_placeholder.markdown("⏳ *Döküman taranıyor ve cevap üretiliyor...*")
                
                # Motorları Çalıştır
                engine = RAGEngine()
                guard = STE100Guard()
                
                # Arama ve Cevaplama (Engine)
                raw_response, sources = engine.search_and_answer(
                    prompt, 
                    st.session_state.selected_collection
                )
                
                # STE100 Denetimi (Guard)
                warnings = guard.check_compliance(raw_response)
                final_response = raw_response # İstersen guard.apply_corrections(raw_response) yapabilirsin
                
                # Cevabı Göster
                full_response = final_response + "\n\n"
                if sources:
                    full_response += "**📚 Kaynaklar:**\n" + "\n".join([f"- {s}" for s in sources])
                
                message_placeholder.markdown(full_response)
                
                # Uyarıları Göster (Expandable olarak)
                if warnings:
                    with st.expander("⚠️ STE100 Uyumluluk Raporu"):
                        for w in warnings:
                            st.write(w)
                
                # Geçmişe Ekle
                st.session_state.messages.append({"role": "assistant", "content": full_response})

elif st.session_state["authentication_status"] is False:
    st.error('Kullanıcı adı veya şifre hatalı.')
elif st.session_state["authentication_status"] is None:
    st.warning('Lütfen giriş yapınız.')