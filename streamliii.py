import streamlit as st
from sentence_transformers import SentenceTransformer
import lancedb
import numpy as np
import re
import pyarrow as pa

# Streamlit UI
st.set_page_config(page_title="جستجوی غزل‌های حافظ", page_icon=":book:")
st.markdown("""
<style>
body, html {
    direction: RTL;
    unicode-bidi: bidi-override;
    text-align: right;
}
p, div, input, label, h1, h2, h3, h4, h5, h6 {
    direction: RTL;
    unicode-bidi: bidi-override;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)

# Initialize the embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer('heydariAI/persian-embeddings')

model = load_model()

# Initialize LanceDB connection
@st.cache_resource
def init_db():
    db = lancedb.connect("lancedb_dir")
    return db

db = init_db()

# Function to search the database
def search(embedding: np.ndarray):
    try:
        table = db.open_table("ghazals")
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        results = table.search(embedding) \
                     .select(["id", "document"]) \
                     .limit(3) \
                     .to_pandas()
        return results
    except Exception as e:
        st.error(f"Error searching database: {str(e)}")
        return None

# Custom CSS for RTL support and styling
st.markdown("""
    <style>
        .reportview-container {
            direction: rtl;
            text-align: right;
        }
        textarea {
            text-align: right !important;
        }
        .stTextArea textarea {
            font-size: 18px !important;
            line-height: 1.6 !important;
        }
        .stMarkdown {
            font-family: "B Nazanin", "Iranian Sans", Tahoma, sans-serif;
        }
        h1 {
            color: #d23669;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("جستجوی غزل‌های حافظ")
st.markdown("""
متن خود را در مورد موضوع مورد نظر بنویسید و مرتبط‌ترین غزل‌های حافظ را پیدا کنید.
""")

# Text input
user_input = st.text_area(
    "متن خود را وارد کنید:", 
    placeholder="مثال: تضاد میان تقوا و صداقت واقعی را نشان می‌دهد...",
    height=150
)

# Search button
if st.button("جستجو"):
    if user_input.strip() == "":
        st.warning("لطفاً متنی برای جستجو وارد کنید")
    else:
        with st.spinner("در حال جستجو..."):
            # Generate embedding for user input
            embedding = model.encode([user_input])[0]
            
            # Search the database
            results = search(embedding)
            
            # Display results
            if results is not None and not results.empty:
                st.success(f"تعداد نتایج یافت شده: {len(results)}")
                
                for idx, row in results.iterrows():
                    with st.expander(f"غزل {idx+1}: {row['id']}"):
                        # Split the document into lines and format nicely
                        lines = row['document'].split('\n')
                        st.markdown(f"**{lines[0]}**")  # Header
                        for line in lines[1:]:
                            if line.strip():  # Skip empty lines
                                st.markdown(f"<div style='line-height: 2.5; font-size: 18px;'>{line}</div>", 
                                           unsafe_allow_html=True)
            else:
                st.info("هیچ نتیجه‌ای یافت نشد. لطفاً متن جستجوی خود را تغییر دهید.")

# Sidebar with info
st.sidebar.title("درباره این برنامه")
st.sidebar.info("""
این برنامه از مدل هوش مصنوعی برای یافتن مرتبط‌ترین غزل‌های حافظ بر اساس متن ورودی شما استفاده می‌کند.

**چگونه کار می‌کند؟**
1. متن شما به یک بردار عددی تبدیل می‌شود
2. این بردار با بردارهای غزل‌های ذخیره شده مقایسه می‌شود
3. نزدیک‌ترین مطالب از نظر معنایی نمایش داده می‌شوند

**داده‌های استفاده شده:**
غزل‌های حافظ از فایل متنی بارگذاری شده‌اند.
""")