import re
from sentence_transformers import SentenceTransformer
import lancedb
import pyarrow as pa
import numpy as np

# -------------------------------
# 1. Define your ghazals text.
# Four ghazals are provided below; we will only index the first three.
text = """
غزل   ۱

الا يا ايها الساقی ادر کاسا و ناولها
که عشق آسان نمود اول ولی افتاد مشکل‌ها

به بوی نافه‌ای کاخر صبا زان طره بگشايد
ز تاب جعد مشکينش چه خون افتاد در دل‌ها

مرا در منزل جانان چه امن عيش چون هر دم
جرس فرياد می‌دارد که بربنديد محمل‌ها

به می سجاده رنگين کن گرت پير مغان گويد
که سالک بی‌خبر نبود ز راه و رسم منزل‌ها

شب تاريک و بيم موج و گردابی چنين هايل
کجا دانند حال ما سبکباران ساحل‌ها

همه کارم ز خود کامی به بدنامی کشيد آخر
نهان کی ماند آن رازی کز او سازند محفل‌ها

حضوری گر همی‌خواهی از او غايب مشو حافظ
متی ما تلق من تهوی دع الدنيا و اهملها


غزل    ۲

صلاح کار کجا و من خراب کجا
ببين تفاوت ره کز کجاست تا به کجا

دلم ز صومعه بگرفت و خرقه سالوس
کجاست دير مغان و شراب ناب کجا

چه نسبت است به رندی صلاح و تقوا را
سماع وعظ کجا نغمه رباب کجا

ز روی دوست دل دشمنان چه دريابد
چراغ مرده کجا شمع آفتاب کجا

چو کحل بينش ما خاک آستان شماست
کجا رويم بفرما از اين جناب کجا

مبين به سيب زنخدان که چاه در راه است
کجا همی‌روی ای دل بدين شتاب کجا

بشد که ياد خوشش باد روزگار وصال
خود آن کرشمه کجا رفت و آن عتاب کجا

قرار و خواب ز حافظ طمع مدار ای دوست
قرار چيست صبوری کدام و خواب کجا


غزل    ۳

اگر آن ترک شيرازی به دست آرد دل ما را
به خال هندويش بخشم سمرقند و بخارا را

بده ساقی می باقی که در جنت نخواهی يافت
کنار آب رکن آباد و گلگشت مصلا را

فغان کاين لوليان شوخ شيرين کار شهرآشوب
چنان بردند صبر از دل که ترکان خوان يغما را

ز عشق ناتمام ما جمال یار مستغنی است
به آب و رنگ و خال و خط چه حاجت روی زیبا را

من از آن حسن روزافزون که یوسف داشت دانستم
که عشق از پرده عصمت برون آرد زلیخا را

اگر دشنام فرمايی و گر نفرین دعا گویم
جواب تلخ می‌زیبد لب لعل شکرخا را

نصیحت گوش کن جانا که از جان دوست‌تر دارند
جوانان سعادتمند پند پیر دانا را

حدیث از مطرب و می گو و راز دهر کمتر جو
که کس نگشود و نگشايد به حکمت این معما را

غزل گفتی و در سفتی بیا و خوش بخوان حافظ
که بر نظم تو افشاند فلک عقد ثریا را


غزل   ۴

صبا به لطف بگو آن غزال رعنا را
که سر به کوه و بیابان تو داده‌ای ما را

شکرفروش که عمرش دراز باد چرا
تفقدی نکند طوطی شکرخا را

غرور حسنت اجازت مگر نداد ای گل
که پرسشی نکنی عندلیب شیده را

به خلق و لطف توان کرد صید اهل نظر
به بند و دام نگیرند مرغ دانا را

ندانم از چه سبب رنگ آشنايي نیست
سهی قدان سیه چشم ماه سیما را

چو با حبیب نشینی و باده پیمايی
به یاد دار محبان بادپیما را

جز اینقدر نتوان گفت در جمال تو عیب
که وضع مهر و وفا نیست روی زیبا را

در آسمان نه عجب گر به گفته حافظ
سرود زهره به رقص آورد مسیحا را
"""

# -------------------------------
# 2. Break down the text into ghazals.
# This regex captures a header (like "غزل   ۱") and then all following text until the next header or end-of-text.
pattern = re.compile(r'(غزل\s*\d+)(.*?)(?=غزل\s*\d+|$)', re.DOTALL)
matches = pattern.findall(text)

# Select only the first three ghazals.
matches = matches[:3]

documents = []  # To store each full ghazal (header and content)
doc_ids = []    # To use the header as a unique ID for each document

for header, body in matches:
    header = header.strip()
    content = body.strip()
    full_doc = f"{header}\n{content}"
    documents.append(full_doc)
    doc_ids.append(header)

# -------------------------------
# 3. Create embeddings for each ghazal using the Persian embedding model.
model = SentenceTransformer('heydariAI/persian-embeddings')
embeddings_list = model.encode(documents)

# Print the embeddings (vectors) for the 3 ghazals.
print("Embeddings for the 3 ghazals:")
for doc_id, embedding in zip(doc_ids, embeddings_list):
    print(doc_id)
    print(embedding)  # This is the numpy vector for the ghazal.
    print("\n----------------\n")

# -------------------------------
# 4. Persist the documents and embeddings in LanceDB.
# Connect (or create) a LanceDB database in the directory "lancedb_dir".
db = lancedb.connect("lancedb_dir")  # The directory "lancedb_dir" will hold the persisted database.

table_name = "ghazals"

# If the table exists from a previous run, delete it to start fresh.
db.drop_all_tables()

# Define a schema for the table.
# We need three columns:
#   - id: the ghazal header
#   - document: the entire ghazal text
#   - embedding: the vector embedding as a list of float32 values
schema = pa.schema([
    ("id", pa.string()),
    ("document", pa.string()),
    ("embedding", pa.list_(pa.float32()))
])

# Prepare records as a list of dictionaries.
records = []
for doc_id, doc, embed in zip(doc_ids, documents, embeddings_list):
    records.append({
        "id": doc_id,
        "document": doc,
        "embedding": embed.tolist()  # Convert numpy array to a Python list for storage.
    })

# Create (or overwrite) the table in LanceDB by providing the data and schema.
# The data parameter accepts a list-of-dicts.
table = db.create_table(table_name, data=records, schema=schema, mode="overwrite")

# -------------------------------
# 5. Read back the stored records and print the vectors.
print("\nStored vectors in LanceDB:")
stored_records = table.to_pandas()  # Converts the table to a list of dictionaries.
for record in stored_records:
    print(record["id"])
    print(record["embedding"])
    print("\n----------------\n")
