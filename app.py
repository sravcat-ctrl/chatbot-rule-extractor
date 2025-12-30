import streamlit as st
import json
import time
import re
from openai import OpenAI
from pypdf import PdfReader

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Rule Extractor Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Programming Rules Extractor Chatbot")

# ==============================
# LOAD OPENAI KEY FROM SECRETS
# ==============================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OpenAI API key not configured in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# CHAT HISTORY
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "📄 Upload programming guidelines (.txt or .pdf)",
    type=["txt", "pdf"]
)

if uploaded_file:
    st.session_state.messages.append({
        "role": "user",
        "content": f"Uploaded file: **{uploaded_file.name}**"
    })

# ==============================
# READ FILE
# ==============================
def read_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        return text
    else:
        return file.read().decode("utf-8")

# ==============================
# CLEAN TEXT
# ==============================
def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

# ==============================
# CHUNK TEXT
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
# SAFE JSON EXTRACTOR
# ==============================
def safe_json_parse(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract JSON block using regex
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return {"rules": []}
        return {"rules": []}

# ==============================
# RULE EXTRACTION
# ==============================
def extract_rules(chunk, index):
    prompt = f"""
You are an expert programming standards analyst.

Extract ALL programming rules from the text below.
Do not miss any rule.

Return ONLY valid JSON.
No explanations.
No markdown.

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
        temperature=0.0
    )

    raw_output = response.choices[0].message.content
    return safe_json_parse(raw_output)

# ==============================
# PROCESS BUTTON
# ==============================
if uploaded_file and st.button("🚀 Extract Rules"):
    with st.chat_message("assistant"):
        st.markdown("📄 Reading document...")
    time.sleep(1)

    text = read_file(uploaded_file)
    text = clean_text(text)

    with st.chat_message("assistant"):
        st.markdown("✂️ Chunking document...")
    chunks = chunk_text(text)

    progress = st.progress(0)
    all_rules = []

    for i, chunk in enumerate(chunks):
        result = extract_rules(chunk, i)
        all_rules.extend(result.get("rules", []))
        progress.progress(int((i + 1) / len(chunks) * 100))

    # ==============================
    # DEDUPLICATE RULES
    # ==============================
    unique_rules = []
    seen = set()

    for rule in all_rules:
        rule_text = rule.get("rule", "").lower()
        if rule_text and rule_text not in seen:
            seen.add(rule_text)
            unique_rules.append(rule)

    final_output = {"rules": unique_rules}
    json_data = json.dumps(final_output, indent=2)

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"✅ Extraction complete! **{len(unique_rules)} rules found.**"
    })

    with st.chat_message("assistant"):
        st.json(final_output)

    st.download_button(
        label="⬇️ Download JSON",
        data=json_data,
        file_name="extracted_rules.json",
        mime="application/json"
    )

