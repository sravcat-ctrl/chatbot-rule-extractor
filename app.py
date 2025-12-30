import streamlit as st
import json
import time
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

st.set_page_config(page_title="Rule Extractor", page_icon="🤖")
st.title("🤖 Programming Rules Extractor")

# Load API key securely
if "OPENAI_API_KEY" not in st.secrets:
    st.error("API key not configured.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("Upload guidelines (.txt)", type=["txt"])

if not uploaded_file:
    st.info("Upload a file to start")
    st.stop()

text = uploaded_file.read().decode("utf-8")

def clean_text(t):
    return re.sub(r'\s+', ' ', t).strip()

text = clean_text(text)

progress = st.progress(0)
status = st.empty()

status.write("Cleaning text...")
progress.progress(20)
time.sleep(1)

status.write("Chunking document...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)
chunks = splitter.split_text(text)
progress.progress(40)

def extract_rules(chunk, i):
    prompt = f"""
Extract ALL programming rules from the text.
Return ONLY JSON.

Format:
{{
  "rules": [
    {{
      "rule_id": "R{i}01",
      "rule": "Rule text",
      "suggested_fix": "Suggested fix"
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

status.write("Extracting rules...")
rules = []

for i, c in enumerate(chunks):
    r = extract_rules(c, i)
    rules.extend(r.get("rules", []))
    progress.progress(40 + int((i+1)/len(chunks)*40))

status.write("Deduplicating rules...")
unique = []
seen = set()

for r in rules:
    key = r["rule"].lower()
    if key not in seen:
        seen.add(key)
        unique.append(r)

progress.progress(100)
status.write("Done!")

output = {"rules": unique}
st.json(output)

st.download_button(
    "Download JSON",
    data=json.dumps(output, indent=2),
    file_name="rules.json",
    mime="application/json"
)
