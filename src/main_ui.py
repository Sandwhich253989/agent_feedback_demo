import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000/agent"

st.set_page_config(page_title="LangGraph Human-in-the-Loop Demo", layout="wide")

# ---------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None

if "waiting_for_user" not in st.session_state:
    st.session_state.waiting_for_user = False

if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None

if "logs" not in st.session_state:
    st.session_state.logs = []

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def log(msg):
    st.session_state.logs.append(msg)

def stream_agent():
    url = f"{API_BASE}/stream/{st.session_state.thread_id}"

    try:
        with requests.get(url, stream=True) as r:
            for line in r.iter_lines():
                if not line:
                    continue

                event = json.loads(line.decode("utf-8"))
                event_type = event["type"]

                if event_type == "update":
                    log(event["data"])

                elif event_type == "interrupt":
                    st.session_state.waiting_for_user = True
                    st.session_state.interrupt_data = event["data"]
                    return  # ⛔ stop streaming

                elif event_type == "done":
                    log("✅ Agent execution completed")
                    return

                elif event_type == "error":
                    log(f"❌ {event['message']}")
                    return
    except Exception as e:
        st.error(f"An Error has occured : {e}")


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

st.title("LangGraph Human-in-the-Loop API demo")

# ---------------------------------------------------------------------
# Start Agent
# ---------------------------------------------------------------------

with st.sidebar:
    st.header("Start Agent")

    prompt = st.text_area(
        "Prompt",
        "Write a comprehensive technical guide about Docker containerization for beginners"
    )

    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.8)
    max_regen = st.number_input("Max Regeneration Attempts", 1, 10, 3)

    if st.button("Start Agent"):
        res = requests.post(
            f"{API_BASE}/start",
            json={
                "prompt": prompt,
                "confidence_threshold": confidence,
                "max_regen_attempts": max_regen,
            },
        ).json()

        st.session_state.thread_id = res["thread_id"]
        st.session_state.logs = []
        st.session_state.waiting_for_user = False

        log(f"🧵 Thread created: {st.session_state.thread_id}")

# ---------------------------------------------------------------------
# Execution Controls
# ---------------------------------------------------------------------

if st.session_state.thread_id:

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("▶️ Run / Resume Agent"):
            stream_agent()

    with col2:
        st.code(st.session_state.thread_id, language="text")

# ---------------------------------------------------------------------
# Interrupt UI
# ---------------------------------------------------------------------

if st.session_state.waiting_for_user and st.session_state.interrupt_data:
    st.divider()
    st.subheader("Human Feedback Required")

    interrupt = st.session_state.interrupt_data[0]

    if interrupt.get("details"):
        st.markdown("**Details**")
        st.info(interrupt["details"])

    choice1 = st.radio(
        interrupt.get("question", "Choose an option"),
        ["y", "n", ""],
        horizontal=True,
    )

    choice2 = st.text_input(label="Enter option")

    choice = choice2 if choice2 else choice1


    if st.button("Submit Feedback and Continue"):
        requests.post(
            f"{API_BASE}/respond/{st.session_state.thread_id}",
            json={"response": choice},
        )

        st.session_state.waiting_for_user = False
        st.session_state.interrupt_data = None

        stream_agent()

        log(f"User responded: {choice}")

        st.rerun()


# ---------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------

st.divider()
st.subheader("📜 Execution Log")

for entry in st.session_state.logs:
    st.write(entry)
