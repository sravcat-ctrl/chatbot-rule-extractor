import streamlit as st
import json
import time
import re
from openai import OpenAI

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Programming Rule Extractor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Programming Rules Extractor")
st.write("Upload messy programming guidelines → Get clean rules as JSON")

# ==============================
# LOAD OPENAI KEY FROM SECRETS
# ==============================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key is not configured in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "📄 Upload guidelines file (.txt)",
    type=["txt"]
)

if not uploaded_file:
    st.info("Please upload a .txt file to begin.")
    st.stop()

raw_text = uploaded_file.read().decode("utf-8")

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

cleaned_text = clean_text(raw_text)

# ==============================
# SIMPLE CHUNKING (NO LANGCHAIN)
# ==============================
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

# ==============================
# PROGRESS UI
# ==============================
progress = st.progress(0)
status = st.empty()

status.write("🧹 Cleaning document...")
time.sleep(1)
progress.progress(20)

# ==============================
# CHUNK DOCUMENT
# ==============================
status.write("✂️ Chunking document...")
chunks = chunk_text(cleaned_text)
time.sleep(1)
progress.progress(40)

# ==============================
# RULE EXTRACTION FUNCTION
# ==============================
def extract_rules(chunk, index):
    prompt = f"""
You are an expert programming standards analyst.

Extract ALL programming rules from the text below.
Do not miss any rule.
Return ONLY valid JSON.

JSON format:
{{
  "rules": [
    {{
      "rule_id": "R{index}01",
      "rule": "Clear rule statement",
      "suggested_fix": "Suggested fix or best practice"
    }}
  ]
}}

TEXT:
{chunk}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return json.loads(response.choices[0].message.content)

# ==============================
# EXTRACT RULES FROM ALL CHUNKS
# ==============================
status.write("🧠 Extracting rules...")
all_rules = []

for i, chunk in enumerate(chunks):
    result = extract_rules(chunk, i)
    all_rules.extend(result.get("rules", []))
    progress.progress(40 + int((i + 1) / len(chunks) * 40))

# ==============================
# DEDUPLICATE RULES
# ==============================
status.write("🔍 Removing duplicate rules...")
unique_rules = []
seen = set()

for rule in all_rules:
    rule_text = rule.get("rule", "").lower()
    if rule_text and rule_text not in seen:
        seen.add(rule_text)
        unique_rules.append(rule)

time.sleep(1)
progress.progress(90)

# ==============================
# FINAL OUTPUT
# ==============================
final_output = {
    "rules": unique_rules
}

json_data = json.dumps(final_output, indent=2)

progress.progress(100)
status.write("✅ Extraction complete!")

# ==============================
# DISPLAY RESULTS
# ==============================
st.subheader("📋 Extracted Rules")
st.json(final_output)

# ==============================
# DOWNLOAD BUTTON
# ==============================
st.download_button(
    label="⬇️ Download JSON",
    data=json_data,
    file_name="extracted_rules.json",
    mime="application/json"
)
