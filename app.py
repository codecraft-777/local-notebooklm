import os
import re
import json
import time
import hashlib
import numpy as np
import requests
import streamlit as st
import faiss
from datetime import datetime
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="ONGC NotebookLM", layout="wide")

# Hide Streamlit's auto-injected page title globally
st.markdown("""
<style>
    section.main .block-container > div > div > div > div:first-child > div[data-testid="stMarkdownContainer"] ~ div h1:first-of-type,
    div[data-testid="stAppViewBlockContainer"] h1:first-of-type {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# ⚙️ CONFIG
# ==========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DIMENSION       = 384
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 50
BATCH_SIZE      = 64
TOP_K           = 4

OLLAMA_URL     = "http://host.docker.internal:11434/api/chat"
OLLAMA_TIMEOUT = 3600
DEFAULT_MODEL  = "phi3:mini"

# Storage paths
os.makedirs("pdf_data", exist_ok=True)
os.makedirs("user_data/chat_history", exist_ok=True)
os.makedirs("user_data/accounts", exist_ok=True)

ACCOUNTS_FILE = "user_data/accounts/users.json"

# ONGC Logo (base64 embedded)
ONGC_LOGO_B64 = "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"
ONGC_LOGO_SRC = f"data:image/png;base64,{ONGC_LOGO_B64}"

def ongc_logo_html(size=36, margin_right=8, border_radius=4):
    return (
        f'<img src="{ONGC_LOGO_SRC}" '
        f'style="height:{size}px;width:{size}px;object-fit:contain;'
        f'vertical-align:middle;margin-right:{margin_right}px;'
        f'border-radius:{border_radius}px;image-rendering:crisp-edges;">'
    )

# ==========================================================
# MODEL CONFIG
# ==========================================================

MODEL_LIST = ["phi3:mini", "llama3.1:8b", "mistral", "gemma3:1b", "tinyllama"]

MODEL_TIPS = {
    "phi3:mini":   "~2.3GB ✅ Best for CPU server",
    "llama3.1:8b": "~5GB 🔥 Best quality (needs GPU)",
    "mistral":     "~4GB ✅ Good quality on CPU",
    "gemma3:1b":   "~1GB ⚡ Fastest on CPU",
    "tinyllama":   "~600MB ❌ Poor answers",
}

MODEL_TOKENS = {
    "phi3:mini":   1000,
    "llama3.1:8b": 1500,
    "mistral":     1000,
    "gemma3:1b":   600,
    "tinyllama":   250,
}

MODEL_CTX = {
    "phi3:mini":   4000,
    "llama3.1:8b": 8000,
    "mistral":     6000,
    "gemma3:1b":   2000,
    "tinyllama":   1200,
}

# ==========================================================
# ── USER AUTH HELPERS ──────────────────────────────────────
# ==========================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_accounts() -> dict:
    if os.path.exists(ACCOUNTS_FILE):
        with open(ACCOUNTS_FILE, "r") as f:
            return json.load(f)
    default = {
        "admin": {
            "password": hash_password("admin123"),
            "display_name": "Administrator",
            "role": "admin",
            "created_at": datetime.now().isoformat(),
        }
    }
    save_accounts(default)
    return default


def save_accounts(accounts: dict):
    with open(ACCOUNTS_FILE, "w") as f:
        json.dump(accounts, f, indent=2)


def register_user(username: str, password: str, display_name: str) -> tuple:
    accounts = load_accounts()
    username = username.strip().lower()
    if not username or not password:
        return False, "Username and password cannot be empty."
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if username in accounts:
        return False, "Username already exists."
    accounts[username] = {
        "password": hash_password(password),
        "display_name": display_name or username,
        "role": "user",
        "created_at": datetime.now().isoformat(),
    }
    save_accounts(accounts)
    return True, "Account created successfully!"


def verify_login(username: str, password: str) -> tuple:
    accounts = load_accounts()
    username = username.strip().lower()
    user = accounts.get(username)
    if not user:
        return False, {}
    if user["password"] == hash_password(password):
        return True, {"username": username, **user}
    return False, {}

# ==========================================================
# ── CHAT HISTORY HELPERS ───────────────────────────────────
# ==========================================================

def history_path(username: str, pdf_name: str) -> str:
    safe_pdf = re.sub(r"[^\w\-.]", "_", pdf_name)
    return f"user_data/chat_history/{username}__{safe_pdf}.json"


def load_chat_history(username: str, pdf_name: str) -> list:
    path = history_path(username, pdf_name)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_chat_history(username: str, pdf_name: str, messages: list):
    path = history_path(username, pdf_name)
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)


def list_user_histories(username: str) -> list:
    histories = []
    prefix = f"{username}__"
    for fname in os.listdir("user_data/chat_history"):
        if fname.startswith(prefix) and fname.endswith(".json"):
            histories.append(fname[len(prefix):-5])
    return histories


def delete_chat_history(username: str, pdf_name: str):
    path = history_path(username, pdf_name)
    if os.path.exists(path):
        os.remove(path)

# ==========================================================
# SESSION STATE INIT
# ==========================================================

for k, v in [
    ("logged_in", False),
    ("current_user", {}),
    ("auth_tab", "login"),
    ("chunks", []),
    ("index", None),
    ("model", DEFAULT_MODEL),
    ("pdf_name", ""),
    ("pdf_data", {}),
    ("messages", []),
    ("load_history_pdf", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================================
# EMBEDDING MODEL — cached, loaded once
# ==========================================================

@st.cache_resource(show_spinner="⏳ Loading embedding model…")
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL, device="cpu")

embedder = load_embedder()

# ==========================================================
# PDF EXTRACTION
# ==========================================================

def extract_pdf(uploaded_file) -> dict:
    reader = PdfReader(uploaded_file)
    pages  = []
    for i, page in enumerate(reader.pages, start=1):
        raw  = page.extract_text() or ""
        text = re.sub(r"\s+", " ", raw).strip()
        pages.append({"page": i, "text": text})
    meta = {}
    if reader.metadata:
        for key in ["/Title", "/Author", "/CreationDate"]:
            val = reader.metadata.get(key, "")
            if val:
                meta[key.lstrip("/")] = str(val).strip()
    return {"pages": pages, "meta": meta, "page_count": len(pages)}


def build_chunks(pdf_data: dict, filename: str) -> list:
    chunks = []
    title  = pdf_data["meta"].get("Title", "") or os.path.splitext(filename)[0]
    meta_chunk = (
        f"DOCUMENT INFO\nTitle: {title}\nFilename: {filename}\n"
        f"Total Pages: {pdf_data['page_count']}\n"
    )
    if pdf_data["meta"].get("Author"):
        meta_chunk += f"Author: {pdf_data['meta']['Author']}\n"
    chunks.append(meta_chunk)
    for pg in pdf_data["pages"]:
        if not pg["text"]:
            continue
        header    = f"[Page {pg['page']} / {pdf_data['page_count']} — {title}]\n"
        full_text = header + pg["text"]
        words     = full_text.split()
        i = 0
        while i < len(words):
            chunk = " ".join(words[i: i + CHUNK_SIZE])
            chunks.append(chunk)
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def index_chunks(chunks: list, progress_bar=None):
    if not chunks:
        return None
    all_embs = []
    total    = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start: start + BATCH_SIZE]
        embs  = embedder.encode(batch, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
        all_embs.extend(embs)
        if progress_bar:
            done = min(start + BATCH_SIZE, total)
            progress_bar.progress(done / total, text=f"Indexing… {done}/{total} chunks")
    arr = np.array(all_embs, dtype="float32")
    idx = faiss.IndexFlatIP(DIMENSION)
    idx.add(arr)
    return idx

# ==========================================================
# HYBRID SEARCH
# ==========================================================

def _keyword_search(query: str, chunks: list, k: int) -> list:
    words = {w for w in re.sub(r"[^\w\s]", "", query.lower()).split() if len(w) > 2}
    if not words:
        return []
    scored = []
    for i, chunk in enumerate(chunks):
        chunk_words = set(re.sub(r"[^\w\s]", "", chunk.lower()).split())
        overlap = len(words & chunk_words)
        if overlap > 0:
            scored.append((overlap / (len(words) + 0.5), i))
    scored.sort(reverse=True)
    return scored[:k]


def search(query: str, index, chunks: list, k: int = TOP_K) -> list:
    if not chunks or index is None:
        return []
    q_emb    = embedder.encode([query], convert_to_numpy=True,
                               normalize_embeddings=True).astype("float32")
    actual_k = min(k * 3, len(chunks))
    D, I     = index.search(q_emb, k=actual_k)
    seen     = {}
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(chunks):
            key = chunks[idx][:60]
            if key not in seen:
                seen[key] = (float(score), chunks[idx])
    for score, idx in _keyword_search(query, chunks, k):
        key = chunks[idx][:60]
        if key not in seen:
            seen[key] = (float(score) * 0.8, chunks[idx])
    results = sorted(seen.values(), key=lambda x: x[0], reverse=True)
    return results[:k]

# ==========================================================
# INTENT DETECTION
# ==========================================================

def detect_intent(query: str) -> dict:
    q      = query.lower().strip()
    intent = {"format": "auto", "is_summary": False, "requested_n": None}
    num_match = re.search(
        r"\b(\d+)\s*(imp(ortant)?|key|main|critical|major)?\s*"
        r"(points?|steps?|items?|findings?|features?|highlights?|"
        r"takeaways?|reasons?|factors?|recommendations?)\b", q
    )
    if num_match:
        intent["requested_n"] = int(num_match.group(1))
        intent["format"]      = "list"
        return intent
    if re.search(r"\b(summar|overview|brief|gist|outline)\w*\b", q):
        intent["is_summary"] = True
        return intent
    if re.search(
        r"\b(list|key points?|main points?|important points?|imp points?|"
        r"steps?|phases?|advantages?|disadvantages?|benefits?|features?|"
        r"findings?|recommendations?|observations?|clauses?|provisions?)\b", q
    ):
        intent["format"] = "list"
        return intent
    if re.search(
        r"\b(explain|describe|elaborate|discuss|detail|how does|why|"
        r"what is|what are|tell me about|how to)\b", q
    ):
        intent["format"] = "paragraph"
        return intent
    if re.search(
        r"\b(how many pages?|page count|author|title|when was|"
        r"who wrote|what date|which year)\b", q
    ):
        intent["format"] = "short"
    return intent

# ==========================================================
# DIRECT METADATA ANSWERS
# ==========================================================

def try_direct(query: str, pdf_data: dict, filename: str):
    if not pdf_data:
        return None
    q     = query.lower().strip()
    title = pdf_data["meta"].get("Title", "") or os.path.splitext(filename)[0]
    if re.search(r"(how many pages|page count|total pages|number of pages)", q):
        return f"**{title}** has **{pdf_data['page_count']} pages**."
    if re.search(
        r"(what is the title|title of (the|this)|"
        r"name of (the|this) (pdf|document|file)|what is this (pdf|document))", q
    ):
        return f"**Title:** {title}"
    if re.search(r"(author|who wrote|written by|created by)", q):
        author = pdf_data["meta"].get("Author", "")
        return f"**Author:** {author}" if author else "Author not found in metadata."
    page_match = re.search(r"\bpage\s+(\d+)\b", q)
    if page_match:
        pno = int(page_match.group(1))
        for pg in pdf_data["pages"]:
            if pg["page"] == pno:
                txt     = pg["text"]
                preview = txt[:800] + ("…" if len(txt) > 800 else "")
                return f"**Page {pno} of {pdf_data['page_count']}:**\n\n{preview}"
        return f"Page {pno} not found (document has {pdf_data['page_count']} pages)."
    return None

# ==========================================================
# OLLAMA — streaming
# ==========================================================

def stream_ollama(prompt: str, model: str, max_tokens: int):
    try:
        with requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_predict":    max_tokens,
                    "num_ctx":        MODEL_CTX.get(model, 4000),
                    "temperature":    0.1,
                    "top_p":          0.9,
                    "repeat_penalty": 1.1,
                },
            },
            stream=True,
            timeout=OLLAMA_TIMEOUT,
        ) as resp:
            if resp.status_code != 200:
                yield f"⚠ Ollama error {resp.status_code}. Run: `ollama pull {model}`"
                return
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data  = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except requests.exceptions.Timeout:
        yield "\n\n⚠ **Timed out.** Try a shorter question or switch to a smaller model."
    except Exception as e:
        yield f"\n\n⚠ Cannot connect to Ollama. Is it running?\n`ollama serve`\n\nError: {e}"

# ==========================================================
# PROMPT BUILDER
# ==========================================================

def build_prompt(query: str, context: str, intent: dict) -> str:
    n = intent.get("requested_n")
    if intent["is_summary"]:
        fmt = (
            "Write a concise summary using **bold headings** for each main topic. "
            "Under each heading write 1-2 sentences. End with **Key Takeaways** (3 points). "
            "Keep under 400 words."
        )
    elif intent["format"] == "list" and n:
        fmt = (
            f"Give exactly {n} numbered points. Start directly with '1.' — no intro sentence. "
            f"Each point must be one complete sentence. Do NOT stop before point {n}."
        )
    elif intent["format"] == "list":
        fmt = (
            "Give a numbered list. Start with '1.' — no intro sentence. "
            "Extract all relevant points. Aim for 5-8 key points."
        )
    elif intent["format"] == "paragraph":
        fmt = (
            "Write a clear answer in 1-3 paragraphs. "
            "Use **bold** for key terms. Be concise."
        )
    elif intent["format"] == "short":
        fmt = "Give the shortest possible answer — one value, number, name, or date only."
    else:
        fmt = (
            "Answer concisely. Use a numbered list for multiple items, "
            "a short paragraph for explanations, or one sentence for facts."
        )
    return f"""You are an expert document assistant for ONGC (Oil and Natural Gas Corporation).
Answer using ONLY the information in the document context below.
If the answer is not in the context, say: Not found in document.

RULES:
- Do NOT repeat the question
- Do NOT write "Answer:" or "Question:"
- Be specific — use exact values, names, procedures from the context
- Be CONCISE — avoid padding or repetition

FORMAT: {fmt}

=== DOCUMENT CONTEXT ===
{context}
========================

Question: {query}

Answer:"""

# ==========================================================
# ══════════════════════════════════════════════════════════
#  LOGIN / REGISTER PAGE
# ══════════════════════════════════════════════════════════
# ==========================================================

def render_login_page():
    st.markdown("""
    <style>
        .block-container { padding-top: 2rem !important; max-width: 480px !important; margin: auto; }
        .login-header { text-align: center; padding: 1.5rem 0 0.5rem; }
        .login-header p  { color: #888; margin-top: 0.2rem; }
        div[data-testid="stForm"] { background: #1a1a2e; border-radius: 12px;
            padding: 2rem; border: 1px solid #333; }
        .ongc-logo-title { display: flex; align-items: center; justify-content: center; gap: 12px; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="login-header">
        <div class="ongc-logo-title">
            {ongc_logo_html(size=52, margin_right=0, border_radius=6)}
            <span style="font-size:2rem;font-weight:700;">ONGC NotebookLM</span>
        </div>
        <p>AI-powered PDF Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["🔐 Login", "📝 Register"])

    with tab_login:
        with st.form("login_form", clear_on_submit=False):
            st.markdown("#### Welcome back")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Login", use_container_width=True)

            if submitted:
                if not username or not password:
                    st.error("Please enter username and password.")
                else:
                    ok, user_info = verify_login(username, password)
                    if ok:
                        st.session_state.logged_in    = True
                        st.session_state.current_user = user_info
                        st.session_state.messages     = []
                        st.success(f"Welcome back, **{user_info['display_name']}**!")
                        time.sleep(0.5)
                        st.rerun()   # ✅ FIXED: was st.experimental_rerun()
                    else:
                        st.error("❌ Invalid username or password.")

        st.caption("Default admin: `admin` / `admin123`")

    with tab_register:
        with st.form("register_form", clear_on_submit=True):
            st.markdown("#### Create an account")
            new_display = st.text_input("Full Name", placeholder="Your full name")
            new_user    = st.text_input("Username", placeholder="Choose a username (min 3 chars)")
            new_pass    = st.text_input("Password", type="password",
                                        placeholder="Choose a password (min 6 chars)")
            new_pass2   = st.text_input("Confirm Password", type="password",
                                        placeholder="Repeat password")
            submitted2  = st.form_submit_button("Create Account", use_container_width=True)

            if submitted2:
                if new_pass != new_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(new_user, new_pass, new_display)
                    if ok:
                        st.success(f"✅ {msg} Please log in.")
                    else:
                        st.error(f"❌ {msg}")

# ==========================================================
# ══════════════════════════════════════════════════════════
#  MAIN APP (after login)
# ══════════════════════════════════════════════════════════
# ==========================================================

def render_main_app():
    user     = st.session_state.current_user
    username = user["username"]

    # ✅ Handle history-restore trigger
    if st.session_state.get("load_history_pdf"):
        hist_pdf = st.session_state.load_history_pdf
        st.session_state.load_history_pdf = None
        saved_messages = load_chat_history(username, hist_pdf)
        st.session_state.messages = saved_messages
        st.session_state.pdf_name = hist_pdf
        st.session_state.chunks   = []
        st.session_state.index    = None
        st.session_state.pdf_data = {}

    st.markdown("""
    <style>
        .block-container { padding-top: 2.5rem !important; }
        .welcome-heading {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        .hist-btn button { text-align: left !important; font-size: 0.8rem !important; }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0 8px;">'
            f'{ongc_logo_html(size=32, margin_right=2, border_radius=4)}'
            f'<span style="font-size:1.1rem;font-weight:700;">ONGC NotebookLM</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("*Single PDF Assistant*")

        st.markdown("---")
        st.markdown(f"👤 **{user['display_name']}**")
        st.caption(f"@{username} · {user.get('role','user')}")
        if st.button("🚪 Logout", use_container_width=True):
            if st.session_state.pdf_name and st.session_state.messages:
                save_chat_history(username, st.session_state.pdf_name,
                                  st.session_state.messages)
            for k in ["logged_in","current_user","chunks","index",
                      "pdf_name","pdf_data","messages","load_history_pdf"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()   # ✅ FIXED: was st.experimental_rerun()

        st.markdown("---")
        uploaded = st.file_uploader("📎 Upload a PDF", type=["pdf"])

        if uploaded and uploaded.name != st.session_state.pdf_name:
            if st.session_state.pdf_name and st.session_state.messages:
                save_chat_history(username, st.session_state.pdf_name,
                                  st.session_state.messages)

            with st.spinner(f"📖 Reading {uploaded.name}…"):
                pdf_data = extract_pdf(uploaded)
                chunks   = build_chunks(pdf_data, uploaded.name)

            bar   = st.progress(0, text="Indexing…")
            index = index_chunks(chunks, progress_bar=bar)
            bar.empty()

            saved_messages = load_chat_history(username, uploaded.name)

            st.session_state.chunks   = chunks
            st.session_state.index    = index
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_data = pdf_data
            st.session_state.messages = saved_messages

            st.success(
                f"✅ Ready! {pdf_data['page_count']} pages loaded."
                + (f" Loaded **{len(saved_messages)}** previous messages." if saved_messages else "")
            )

        # Show only the PDF name
        if st.session_state.pdf_name:
            st.markdown("---")
            st.markdown("**📄 Loaded PDF:**")
            st.info(st.session_state.pdf_name)

        st.markdown("---")
        st.markdown("**🤖 Model**")
        idx_default  = MODEL_LIST.index(st.session_state.model) \
                       if st.session_state.model in MODEL_LIST else 0
        model_choice = st.selectbox(
            "model", MODEL_LIST, index=idx_default, label_visibility="collapsed"
        )
        st.session_state.model = model_choice
        st.caption(MODEL_TIPS.get(model_choice, ""))

        histories = list_user_histories(username)
        if histories:
            st.markdown("---")
            st.markdown("**📂 Previous Sessions**")
            for hist_pdf in histories[:8]:
                msgs      = load_chat_history(username, hist_pdf)
                msg_count = len(msgs)
                display   = hist_pdf[:22] + ("…" if len(hist_pdf) > 22 else "")
                col1, col2, col3 = st.columns([4, 2, 1])
                with col1:
                    st.caption(f"📄 {display}")
                    st.caption(f"   {msg_count} msgs")
                with col2:
                    if st.button("📂 Open", key=f"open_{hist_pdf}",
                                 use_container_width=True,
                                 help=f"Restore chat for {hist_pdf}"):
                        st.session_state.load_history_pdf = hist_pdf
                        st.experimental_rerun()   # ✅ FIXED: was st.experimental_rerun()
                with col3:
                    if st.button("🗑", key=f"del_{hist_pdf}",
                                 help=f"Delete history for {hist_pdf}"):
                        delete_chat_history(username, hist_pdf)
                        if st.session_state.pdf_name == hist_pdf:
                            st.session_state.messages = []
                            st.session_state.pdf_name = ""
                        st.rerun()   # ✅ FIXED: was st.experimental_rerun()

        st.markdown("---")
        if st.button("🗑 Clear Chat", use_container_width=True):
            if st.session_state.pdf_name:
                delete_chat_history(username, st.session_state.pdf_name)
            st.session_state.messages = []
            st.rerun()   # ✅ FIXED: was st.experimental_rerun()

    # ══════════════════════════════════════════════════════
    # MAIN CHAT AREA
    # ══════════════════════════════════════════════════════

    if not st.session_state.pdf_name:
        st.markdown(
            f'<div class="welcome-heading">👋 Welcome, {user["display_name"]}!</div>',
            unsafe_allow_html=True
        )
        st.info("👈 Upload a PDF from the sidebar to get started, or open a previous session!")
        st.markdown("""
**Example questions you can ask:**
- `summary of pdf`
- `important points from pdf`
- `5 key points about contracts`
- `explain the indenting process`
- `steps for contract closing`
- `what is on page 25`
- `how many pages`
- `who is the author`
        """)
        histories = list_user_histories(username)
        if histories:
            st.markdown("---")
            st.markdown("### 📂 Your Previous Chat Sessions")
            for hist_pdf in histories:
                msgs = load_chat_history(username, hist_pdf)
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"📄 **{hist_pdf}** — {len(msgs)} messages")
                with col2:
                    if st.button("Open", key=f"main_open_{hist_pdf}"):
                        st.session_state.load_history_pdf = hist_pdf
                        st.rerun()   # ✅ FIXED: was st.experimental_rerun()
        st.stop()

    if st.session_state.pdf_name and not st.session_state.chunks:
        st.info(
            f"📖 Viewing saved chat for **{st.session_state.pdf_name}**. "
            "Re-upload the PDF to ask new questions."
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    chat_placeholder = (
        "Ask anything about your PDF…"
        if st.session_state.chunks
        else "Re-upload the PDF to ask new questions"
    )
    query = st.chat_input(chat_placeholder, disabled=not bool(st.session_state.chunks))

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({
            "role": "user", "content": query,
            "timestamp": datetime.now().isoformat()
        })

        start    = time.time()
        intent   = detect_intent(query)
        model    = st.session_state.model
        pdf_data = st.session_state.get("pdf_data", {})

        with st.chat_message("assistant"):

            direct = try_direct(query, pdf_data, st.session_state.pdf_name)

            if direct:
                st.markdown(direct)
                elapsed = round(time.time() - start, 3)
                st.caption(f"⚡ {elapsed}s | metadata lookup")
                st.session_state.messages.append({
                    "role": "assistant", "content": direct,
                    "timestamp": datetime.now().isoformat()
                })

            else:
                results   = search(
                    query, st.session_state.get("index"),
                    st.session_state.chunks, k=TOP_K,
                )
                context   = "\n\n".join(text for _, text in results)
                ctx_limit = MODEL_CTX.get(model, 4000)
                context   = context[:ctx_limit]

                prompt     = build_prompt(query, context, intent)
                max_tokens = MODEL_TOKENS.get(model, 1000)
                if intent["requested_n"]:
                    max_tokens = max(max_tokens, intent["requested_n"] * 80)

                placeholder = st.empty()
                full_answer = ""
                for token in stream_ollama(prompt, model, max_tokens):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")

                placeholder.markdown(full_answer)

                elapsed = round(time.time() - start, 1)
                st.caption(f"⏱ {elapsed}s | {model} | format: `{intent['format']}`")

                st.session_state.messages.append({
                    "role": "assistant", "content": full_answer,
                    "timestamp": datetime.now().isoformat()
                })

                if results:
                    with st.expander("🔍 Retrieved chunks", expanded=False):
                        for score, text in results:
                            st.markdown(f"`score: {score:.3f}`")
                            st.code(text[:400], language=None)
                            st.markdown("---")

        save_chat_history(username, st.session_state.pdf_name, st.session_state.messages)

# ==========================================================
# ── ROUTER ────────────────────────────────────────────────
# ==========================================================

if not st.session_state.logged_in:
    render_login_page()
else:
    render_main_app()