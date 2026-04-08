import logging
import requests
import streamlit as st
import streamlit_authenticator as stauth
import os

from src.utils import load_config, load_secrets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="STE100 Technical Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

config = load_config()
COLLECTION_NAME = config.get("vector_db", {}).get("collection_name", "doc_v2_asset_store")
API_URL = config.get("app", {}).get("api_url", "http://127.0.0.1:8050/ask")

def load_auth():
    config_auth = load_secrets()
    if not config_auth:
        logger.error("Kimlik dogrulama ayarlari yuklenemedi veya bos.")
        return None
        
    try:
        authenticator = stauth.Authenticate(
            config_auth.get("credentials", {}),
            config_auth.get("cookie", {}).get("name", "ste100_cookie"),
            config_auth.get("cookie", {}).get("key", "signature_key"),
            config_auth.get("cookie", {}).get("expiry_days", 30)
        )
        return authenticator
    except Exception as e:
        logger.error("Kimlik dogrulama yapilandirma hatasi: %s", e)
        return None

authenticator = load_auth()
is_kaggle_test = os.environ.get("KAGGLE_TEST_MODE") == "1"

if is_kaggle_test:
    name = "Kaggle Test Kullanicisi"
    authentication_status = True
    username = "test_user"
elif authenticator:
    try:
        name, authentication_status, username = authenticator.login("main")
    except Exception as e:
        logger.error("Kimlik dogrulama modulu baslatilamadi: %s", e)
        st.error("Giris sistemi su anda kullanilamiyor. Lutfen yoneticiye basvurun.")
        name, authentication_status, username = None, None, None
else:
    st.error("Sistem ayarlari yuklenemedigi icin giris yapilamiyor.")
    authentication_status = None

if authenticator:
    try:
        name, authentication_status, username = authenticator.login("main")
    except Exception as e:
        logger.error("Kimlik dogrulama modulu baslatilamadi: %s", e)
        st.error("Giris sistemi su anda kullanilamiyor. Lutfen yoneticiye basvurun.")
        name, authentication_status, username = None, None, None
else:
    st.error("Sistem ayarlari yuklenemedigi icin giris yapilamiyor.")
    authentication_status = None

if authentication_status:
    with st.sidebar:
        st.title("Admin Panel")
        st.write(f"Hosgeldin, **{name}**")
        st.divider()
        
        st.subheader("Sistem Ayarlari")
        
        ste100_mode = st.toggle(
            "STE100 Formatinda Yanitla",
            value=False,
            help="Kapatildiginda normal ve akici dille bilgi verir. Acildiginda STE100 kurallarini zorunlu kilar."
        )
        
        strict_mode = False
        template_type = "General"
        
        if ste100_mode:
            template_type = st.radio(
                "Uretim Formati Seciniz:",
                options=["General", "Procedure", "Descriptive", "Safety"],
                help="Modelin uretecegi metnin yapisal kurallarini belirler."
            )
            strict_mode = st.toggle("Siki Denetim (Otomatik Duzeltme)", value=False)
            
        st.info(f"Veri Seti: {COLLECTION_NAME}")
        
        st.divider()
        
        if st.button("Sohbeti Temizle", type="primary"):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        authenticator.logout("Cikis Yap", "sidebar")

    st.title("In-House Technical Support AI")
    st.caption("Simplified Technical English (ASD-STE100) Compliance Enforced")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg.get("is_report", False):
                is_compliant = msg.get("is_compliant", True)
                was_corrected = msg.get("was_corrected", False)
                feedback = msg.get("feedback_report", [])
                
                if is_compliant:
                    status_title = "Teknik Rapor (Kusursuz)"
                elif was_corrected:
                    status_title = "STE100 Duzeltme Raporu (Onarildi)"
                else:
                    status_title = "STE100 Denetim Raporu (Ihlal Var)"
                
                with st.expander(status_title):
                    if not is_compliant:
                        if was_corrected:
                            st.info("Arka planda uygulanan duzeltmeler:")
                        else:
                            st.error("Duzeltilmeyen Ihlaller (Siki Denetim Kapali):")
                        for f in feedback:
                            st.markdown(f)
                    else:
                        st.success("Kelime kullanimi ASD-STE100 sozlugune %100 uygundur.")
                    
                    st.divider()
                    st.markdown("**Referans Alinan Teknik Dokuman:**")
                    st.info(msg.get("context_text", "Context bulunamadi."))

    if prompt := st.chat_input("Teknik sorunuzu buraya yazin..."):
        msg_data = {"role": "user", "content": prompt}
        st.session_state.messages.append(msg_data)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status("Backend ile iletisim kuruluyor...", expanded=True)
            
            try:
                chat_history = st.session_state.messages[:-1]
                if len(chat_history) > 6:
                    chat_history = chat_history[-6:]
                
                payload = {
                    "query": prompt,
                    "collection_name": COLLECTION_NAME,
                    "history": chat_history,
                    "use_ste100": ste100_mode,
                    "strict_mode": strict_mode,
                    "template_type": template_type
                }
                
                status_container.write("Analiz ediliyor ve yanit uretiliyor...")
                
                response = requests.post(API_URL, json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                
                final_text = data.get("final_text", "")
                context_text = data.get("context_text", "")
                is_compliant = data.get("is_compliant", True)
                was_corrected = data.get("was_corrected", False)
                feedback_report = data.get("feedback_report", [])
                
                status_container.update(label="Islem Tamamlandi", state="complete", expanded=False)
                st.markdown(final_text)
                
                if ste100_mode:
                    if is_compliant:
                        expander_title = "Teknik Rapor (Kusursuz)"
                    elif was_corrected:
                        expander_title = "STE100 Duzeltme Raporu (Onarildi)"
                    else:
                        expander_title = "STE100 Denetim Raporu (Ihlal Var)"
                        
                    with st.expander(expander_title, expanded=(not is_compliant and not strict_mode)):
                        if is_compliant:
                            st.success("Kelime kullanimi ASD-STE100 sozlugune %100 uygundur.")
                        else:
                            if was_corrected:
                                st.info("Asagidaki STE100 kurallari taslak metne basariyla uygulandi:")
                            else:
                                st.error(f"Bu cevapta {len(feedback_report)} adet STE100 ihlali tespit edildi:")
                            for f in feedback_report:
                                st.markdown(f)
                        st.markdown("---")
                        st.caption("**Kullanilan Baglam (Context):**")
                        st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)
                else:
                    with st.expander("Kaynak ve Baglam Bilgisi"):
                        st.info("STE100 denetimi kapali. Dogal dilde analiz uretildi.")
                        st.markdown("---")
                        st.caption("**Kullanilan Baglam (Context):**")
                        st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "is_report": ste100_mode,
                    "is_compliant": is_compliant,
                    "was_corrected": was_corrected,
                    "feedback_report": feedback_report,
                    "context_text": context_text
                })

            except requests.exceptions.RequestException as e:
                logger.error("Backend API iletisim hatasi: %s", e, exc_info=True)
                status_container.update(label="Sunucuya Ulasilamadi", state="error")
                st.error("Arka plan servisi (Backend) yanit vermiyor. Lutfen yoneticiye basvurun.")
            except Exception as e:
                logger.error("Uretim sirasinda kritik hata: %s", e, exc_info=True)
                status_container.update(label="Sistem Hatasi", state="error")
                st.error("Isteginizi islerken sistemsel bir sorun olustu. Lutfen daha sonra tekrar deneyin.")