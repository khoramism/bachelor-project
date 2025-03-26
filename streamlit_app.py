import streamlit as st
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np
import pandas as pd

# Must be first command, just for it to be centered
st.set_page_config(
    page_title="جستجوی ابیات حافظ",
    page_icon=":book:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS with colors and RTL support
st.markdown("""
<style>
    body, html, [class*="css"] {
        direction: rtl;
        text-align: right;
        font-family: 'B Nazanin', Tahoma, sans-serif;
    }
    .stTextArea textarea {
        background-color:  #584B40;
        border: 2px solid #2563eb;
        border-radius: 8px;
        font-size: 18px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: scale(1.05);
    }
    .verse-box {
        background: linear-gradient(145deg, #f0f4ff, #ffffff);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #2563eb;
    }
    .success-msg {
        color: #059669;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    .warning-msg {
        color: #d97706;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models and DB
@st.cache_resource
def load_model():
    return SentenceTransformer('heydariAI/persian-embeddings')

model = load_model()

@st.cache_resource
def init_db():
    return lancedb.connect("lancedb_dir")

db = init_db()

def search_verses(embedding: np.ndarray, top_k=3):
    try:
        table = db.open_table("ghazals")
        results = table.search(embedding.tolist()) \
                     .select(["verse_text", "ghazal_id"]) \
                     .limit(top_k) \
                     .to_pandas()
        return results
    except Exception as e:
        st.error(f"خطای پایگاه داده: {str(e)}")
        return pd.DataFrame()

# Main app interface
st.title("📖 جستجوی هوشمند ابیات حافظ")
user_input = st.text_area(
    "متن یا مفهوم مورد نظر خود را وارد کنید:",
    placeholder="مثال: شعر در مورد عشق و آزادی...",
    height=150
)

if st.button("🔍 انجام جستجو"):
    if user_input.strip():
        with st.spinner("در حال پردازش و جستجو..."):
            query_embedding = model.encode([user_input])[0]
            results = search_verses(query_embedding)
            
            if not results.empty:
                st.markdown(f'<div class="success-msg">🎉 تعداد نتایج یافت شده: {len(results)}</div>', 
                           unsafe_allow_html=True)
                
                for _, row in results.iterrows():
                    st.markdown(f"""
                    <div class="verse-box">
                        <h4 style="color: #2563eb; margin-bottom: 0.5rem;">{row['ghazal_id']}</h4>
                        <p style="font-size: 1.2rem; line-height: 2; color: #1e3a8a;">
                            {row['verse_text']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-msg">⚠️ هیچ نتیجه‌ای یافت نشد</div>', 
                           unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-msg">⚠️ لطفاً متن جستجو را وارد کنید</div>', 
                    unsafe_allow_html=True)

# Sidebar information
st.sidebar.title("ℹ️ درباره برنامه")
st.sidebar.markdown("""
<div style="background-color: #f0f9ff; padding: 1rem; border-radius: 8px;">
    <h4 style="color: #2563eb;">امکانات سیستم:</h4>
    <p style="color: #1e3a8a;">
        • جستجوی معنایی در دیوان حافظ<br>
        • استفاده از هوش مصنوعی برای درک متن<br>
        • نمایش نتایج مرتبط با سبک مدرن<br>
        • پشتیبانی کامل از زبان فارسی
    </p>
</div>
""", unsafe_allow_html=True)