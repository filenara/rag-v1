import streamlit as st
import streamlit_authenticator as stauth
import yaml
import time
from yaml.loader import SafeLoader
from PIL import Image
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
    return RAGEngine()

@st.cache_resource
def get_guard():
    # Artık JSON sözlüğünü okuyan yeni Guard sınıfımız çalışıyor
    return STE100Guard()

engine = get_engine()
guard = get_guard()
config = load_config()

COLLECTION_NAME = "doc_kaggle_v1"

# --- 3. KİMLİK DOĞRULAMA ---
def load_auth():
    with open('config/secrets.yaml') as file:
        config_auth = yaml.load(file, Loader=SafeLoader)
    
    authenticator = stauth.Authenticate(
        config_auth['credentials'],
        config_auth['cookie']['name'],
        config_auth['cookie']['key'],
        config_auth['cookie']['expiry_days']
    )
    return authenticator

authenticator = load_auth()

try:
    # Güncel versiyonlar için
    name, authentication_status, username = authenticator.login("main")
except TypeError:
    # Eski versiyon uyumluluğu (Eğer "main" argümanı kabul edilmiyorsa)
    name, authentication_status, username = authenticator.login()
except Exception as e:
    # Beklenmeyen kritik hataları gizleme, ekrana bas
    st.error(f"Kimlik doğrulama modülü başlatılamadı: {e}")
    name, authentication_status, username = None, None, None

# --- 4. ANA UYGULAMA MANTIĞI ---

if authentication_status:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/6024/6024190.png", width=50)
        st.title("Admin Panel")
        st.write(f"Hoşgeldin, **{name}**")
        st.divider()
        
        # Ayarlar ve Yeni Toggle Butonu
        st.subheader("⚙️ Sistem Ayarları")
        
        # Self-Correction (Kendi Kendini Düzeltme) Butonu
        strict_mode = st.toggle(
            "Sıkı Denetim (Otomatik Düzeltme)", 
            value=False, 
            help="Açıksa, LLM tespit edilen STE100 ihlallerini kendi kendine yeniden yazarak düzeltir. (Cevap süresini uzatabilir)"
        )
        
        st.info(f"Veri Seti: {COLLECTION_NAME}")

        st.divider()
        st.subheader("Görsel Analiz")
        uploaded_file = st.file_uploader("Sorguya görsel ekleyin (İsteğe bağlı)", type=["png", "jpg", "jpeg"])
        
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
            if "image" in msg:
                st.image(msg["image"], caption="Uploaded Image", width=300)
            
            st.markdown(msg["content"])
            
            # Geçmiş Raporları Göster
            if msg.get("is_report", False):
                is_compliant = msg.get("is_compliant", True)
                was_corrected = msg.get("was_corrected", False)
                feedback = msg.get("feedback_report", [])
                
                # Rapor başlığını duruma göre belirle
                if is_compliant:
                    status_title = "Teknik Rapor (Kusursuz)"
                    status_icon = "✅"
                elif was_corrected:
                    status_title = "STE100 Düzeltme Raporu (Onarıldı)"
                    status_icon = "🔧"
                else:
                    status_title = "STE100 Denetim Raporu (İhlal Var)"
                    status_icon = "⚠️"
                
                with st.expander(f"{status_icon} {status_title}"):
                    if not is_compliant:
                        if was_corrected:
                            st.info("Arka planda uygulanan düzeltmeler:")
                        else:
                            st.error("Düzeltilmeyen İhlaller (Sıkı Denetim Kapalı):")
                        
                        for f in feedback:
                            st.markdown(f)
                    else:
                        st.success("Kelime kullanımı ASD-STE100 sözlüğüne %100 uygundur.")
                    
                    st.divider()
                    st.markdown("**Referans Alınan Teknik Döküman:**")
                    st.info(msg.get("context_text", "Context bulunamadı."))

    # --- KULLANICI GİRİŞİ ---
    if prompt := st.chat_input("Teknik sorunuzu buraya yazın..."):
        
        # Resmi hazırla
        user_image = None
        msg_data = {"role": "user", "content": prompt}
        
        if uploaded_file is not None:
            user_image = Image.open(uploaded_file)
            msg_data["image"] = user_image
            
        # 1. Kullanıcı Mesajını Ekrana Bas
        st.session_state.messages.append(msg_data)
        
        with st.chat_message("user"):
            if user_image:
                st.image(user_image, caption="Kullanıcı Görseli", width=300)
            st.markdown(prompt)

        # 2. Asistan Cevabını Üret
        with st.chat_message("assistant"):
            status_container = st.status("Analiz ediliyor...", expanded=True)
            
            try:
                # A) İlk Taslak (Draft) Üretimi
                status_container.write("Teknik dökümanlar taranıyor...")
                
                # --- DEĞİŞTİRİLEN KISIM: user_image parametresi motora gönderiliyor ---
                draft_text, context_text = engine.search_and_answer(
                    query=prompt, 
                    collection_name=COLLECTION_NAME,
                    history=st.session_state.messages,
                    user_image=user_image 
                )
                
                # B) JSON Sözlüğü ile Teşhis (Guard)
                status_container.write("STE100 kuralları denetleniyor...")
                is_compliant, feedback_report = guard.analyze_and_report(draft_text)
                
                final_text = draft_text
                was_corrected = False

                # C) Self-Correction (Kendi Kendini Düzeltme) Döngüsü
                # C) Self-Correction (Kendi Kendini Düzeltme) Döngüsü
                if not is_compliant and strict_mode:
                    max_retries = 2
                    retries = 0
                    current_text = draft_text
                    
                    # Modeli döngüden önce bir kez yüklüyoruz ki her denemede zaman kaybetmeyelim
                    model, processor = engine.llm_manager.load_vision_model()
                    
                    while not is_compliant and retries < max_retries:
                        retries += 1
                        status_container.write(f"İhlaller analiz ediliyor ve düzeltiliyor... (Deneme {retries}/{max_retries})")
                        
                        previous_text = current_text
                        
                        # Motor düzeltmeyi yapıyor
                        current_text = engine.refine_answer(current_text, feedback_report, model, processor)
                        
                        # Eğer LLM kuralları okuyup "Değişikliğe gerek yok, doğru kullanılmış" derse döngüyü kır
                        if current_text.strip() == previous_text.strip():
                            status_container.write("LLM sözcük türü kullanımını (Part of Speech) onayladı.")
                            is_compliant = True # Yalancı pozitifi (False Positive) yoksay
                            break
                            
                        # Yeni metni tekrar denetle
                        is_compliant, feedback_report = guard.analyze_and_report(current_text)
                        
                    final_text = current_text
                    was_corrected = True
                    
                    if is_compliant:
                        status_container.write("Düzeltme başarıyla tamamlandı.")
                    else:
                        status_container.write(f"Sistem {max_retries} kez denedi ancak tüm kelimeler sözlüğe uydurulamadı.")
                
                # Sıkı denetim kapalıysa ve ihlal varsa expander açık gelsin ki kullanıcı görsün
                with st.expander(expander_title, expanded=(not is_compliant and not strict_mode)):
                    if is_compliant:
                        st.success("Kelime kullanımı ASD-STE100 sözlüğüne %100 uygundur.")
                    else:
                        if was_corrected:
                            st.info("Aşağıdaki STE100 kuralları taslak metne başarıyla uygulandı:")
                        else:
                            st.error(f"Bu cevapta {len(feedback_report)} adet STE100 ihlali tespit edildi. Sıkı Denetim kapalı olduğu için düzeltilmedi:")
                            
                        for f in feedback_report:
                            st.markdown(f)
                        
                    st.markdown("---")
                    st.caption("**Kullanılan Bağlam (Context):**")
                    st.text(context_text[:1000] + "..." if len(context_text) > 1000 else context_text)

                # 3. Geçmişe Kaydet
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_text,
                    "is_report": True,
                    "is_compliant": is_compliant,
                    "was_corrected": was_corrected,
                    "feedback_report": feedback_report,
                    "context_text": context_text
                })

            except Exception as e:
                status_container.update(label="Hata", state="error")
                st.error(f"Bir hata oluştu: {str(e)}")