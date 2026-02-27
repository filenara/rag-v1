import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from src.rag_engine import RAGEngine
from src.ste100_guard import STE100Guard
from src.utils import load_config


st.set_page_config(
    page_title="STE100 Technical Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_engine():
    return RAGEngine()


@st.cache_resource
def get_guard():
    return STE100Guard()


engine = get_engine()
guard = get_guard()
config = load_config()

COLLECTION_NAME = config.get("vector_db", {}).get("collection_name", "doc_v2_asset_store")


def load_auth():
    with open("config/secrets.yaml") as file:
        config_auth = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config_auth["credentials"],
        config_auth["cookie"]["name"],
        config_auth["cookie"]["key"],
        config_auth["cookie"]["expiry_days"]
    )
    return authenticator


authenticator = load_auth()

try:
    name, authentication_status, username = authenticator.login("main")
except Exception as e:
    st.error(f"Kimlik dogrulama modulu baslatilamadi: {e}")
    name, authentication_status, username = None, None, None


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
        if ste100_mode:
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
            status_container = st.status("Analiz ediliyor...", expanded=True)
            
            try:
                status_container.write("Teknik dokumanlar taraniyor...")
                
                draft_text, context_text = engine.search_and_answer(
                    query=prompt, 
                    collection_name=COLLECTION_NAME,
                    history=st.session_state.messages,
                    use_ste100=ste100_mode
                )
                
                final_text = draft_text
                was_corrected = False
                feedback_report = []
                is_compliant = True

                if ste100_mode:
                    status_container.write("STE100 kurallari denetleniyor...")
                    is_compliant, feedback_report = guard.analyze_and_report(draft_text)
                    
                    if not is_compliant and strict_mode:
                        max_retries = 2
                        retries = 0
                        current_text = draft_text
                        model, processor = engine.llm_manager.load_vision_model()
                        
                        while not is_compliant and retries < max_retries:
                            retries += 1
                            status_container.write(f"Ihlaller duzeltiliyor... (Deneme {retries}/{max_retries})")
                            
                            previous_text = current_text
                            current_text = engine.refine_answer(
                                current_text,
                                feedback_report,
                                model,
                                processor
                            )
                            
                            if current_text.strip() == previous_text.strip():
                                break
                                
                            is_compliant, feedback_report = guard.analyze_and_report(current_text)
                            
                        final_text = current_text
                        was_corrected = True

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

            except Exception as e:
                status_container.update(label="Hata", state="error")
                st.error(f"Bir hata olustu: {str(e)}")