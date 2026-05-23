# recoverai_streamlit.py
# Streamlit UI for RecoverAI (Admin • Clinician • Patient)
# Colors: Primary = #6b9ebd, Accent = #f18a81

import os
import random
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import cv2
import mediapipe as mp
from PIL import Image
import numpy as np
from db import init_db, create_user, get_user_for_login, get_conn
# import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from inference.realtime_processor import RecoverAIVideoProcessor
from inference.predictor import RecoverAIPredictor

@st.cache_resource
def get_recoverai_predictor():
    return RecoverAIPredictor(
        checkpoint_path="models/best_conditioned_tcn_clean.pt",
        device="cpu",
    )
EXERCISE_NAME_TO_ID = {
    "Arm Abduction": 1,
    "Arm VW": 2,
    "Push-ups": 3,
    "Leg Abduction": 4,
    "Leg Lunge": 5,
    "Squats": 6,
}
def admin_exists():
    from db import get_conn
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
    exists = cur.fetchone() is not None
    conn.close()
    return exists

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


PRIMARY = "#6b9ebd"
ACCENT = "#fa9b93"
BG_SOFT = "#d7e7f2"

st.set_page_config(page_title="RecoverAI", page_icon="🩺", layout="wide")

@st.cache_resource
def boot_db():
    init_db()
    return True

boot_db()

# ---------- Global styles ----------
st.markdown(
    f"""
    <style>
      :root {{
        --primary: {PRIMARY};
        --accent: {ACCENT};
      }}
      .recoverai-hero {{
        text-align:center;
        padding: 60px 10px 20px;
      }}
      .recoverai-hero h1 {{
        font-size: 3rem;
        margin: 0;
        color: var(--primary);
        letter-spacing: .5px;
      }}
      .recoverai-hero p {{
        color: #345;
        font-size: 1.1rem;
        opacity: .9;
      }}
      .soft-card {{
        background: white;
        border: 1px solid #e9eef3;
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,.03);
      }}
      .pill {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(107,158,189,.12);
        color: var(--primary);
        font-weight: 600;
        font-size: 12px;
        letter-spacing: .2px;
      }}
      .btn-primary button {{
        background: var(--primary) !important;
        color: white !important;
        border-radius: 12px !important;
      }}
      .btn-accent button {{
        background: var(--accent) !important;
        color: white !important;
        border-radius: 12px !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar radio → navigation style ----------
st.markdown(
    """
    <style>
    /* Make sidebar radios look like the flat nav menu */

    /* Layout spacing */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.25rem;
    }

    /* Hide default radio circle */
    [data-testid="stSidebar"] [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }

    /* Nav item base style */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.45rem 0.75rem;
        border-radius: 8px;
        width: 100%;
        cursor: pointer;
        transition: 0.2s ease-in-out; /* smooth hover */
    }

    /* 🔹 Hover (darker background) */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #c9d3e6 !important; /* darker tone */
    }

    /* Selected page style */
    [data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"] {
        background-color: #e1e5f2 !important;
        font-weight: 600;
        border-left: 4px solid var(--primary); /* optional highlight */
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Session bootstrap ----------
if "page" not in st.session_state:
    st.session_state.page = "first"  # first -> welcome -> role flows
if "auth" not in st.session_state:
    st.session_state.auth = {"role": None, "email": None, "name": None}
if "db" not in st.session_state:
    st.session_state.db = {
        "clinicians": pd.DataFrame([
            {
                "id": 1001, "first":"Hanan", "last":"Aljindan", "username":"hanan",
                "email":"hanan@example.com", "birthdate":"1990-02-14",
                "assigned_patients": 8, "status":"active", "specialty":"Physiotherapy"
            }
        ]),
        "patients": pd.DataFrame([
            {
                "id": 2001, "first":"Aseel","last":"Alkhaldi","username":"aseel_a",
                "email":"aseel@example.com", "birthdate":"2004-01-29",
                "assigned_clinician":"hanan","adherence":86
            }
        ]),
        "logs": pd.DataFrame(columns=["username","role","action","timestamp","device"]),
        "announcements": []
    }

# Auto-increment demo IDs
if "c_next_id" not in st.session_state:
    st.session_state.c_next_id = int(st.session_state.db["clinicians"].get("id", pd.Series([1000])).max()) + 1
if "p_next_id" not in st.session_state:
    st.session_state.p_next_id = int(st.session_state.db["patients"].get("id", pd.Series([2000])).max()) + 1

if "nav_history" not in st.session_state:
    st.session_state.nav_history = []

# --- Admin Profile store (bootstrap) ---
if "admin_profile" not in st.session_state:
    st.session_state.admin_profile = {
        "first_name": "Raghad",
        "last_name": "Aljiban",
        "username": "raghad",
        "email": "Raghad@recover.ai",
        "phone": "+9665XXXXXXX",
    }

# ---------- Language State ----------
if "lang" not in st.session_state:
    st.session_state.lang = "EN"  # default language

# --- Notifications bootstrap (demo data for all roles) ---
if "notifications" not in st.session_state:
    st.session_state.notifications = pd.DataFrame([
        {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "text": "New patient assigned: Aseel Alkhaldi. Review their rehabilitation plan.",
            "read": False
        },
        {
            "timestamp": (pd.Timestamp.now()-pd.Timedelta(hours=5)).strftime("%Y-%m-%d %H:%M"),
            "text": "Daily summary available: 3 patients completed all exercises.",
            "read": False
        },
        {
            "timestamp": (pd.Timestamp.now()-pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
            "text": "Admin Announcement: System maintenance scheduled tomorrow from 2:00–3:00 PM. Please save your work.",
            "read": True
        },
    ])




# --- Demo password store + reset state ---
if "passwords" not in st.session_state:
    st.session_state.passwords = {
        "admin": {"1000": "@R2003d", "admin@recover.ai": "@R2003d"},
        "clinician": {},  # optional demo storage
        "patient": {},    # optional demo storage
    }

if "reset_codes" not in st.session_state:
    st.session_state.reset_codes = {}   # {email_or_id: {"code": "123456", "role": "admin/clinician/patient"}}

# ---------- Navigation helpers ----------
def go_to(new_page: str):
    """Navigate to a new page while pushing current page onto a stack."""
    cur = st.session_state.page
    if cur != new_page:
        st.session_state.nav_history.append(cur)
    st.session_state.page = new_page
    st.rerun()

def can_go_back():
    return len(st.session_state.nav_history) > 0

def back():
    """Pop the last page from the stack; fallback to 'welcome'."""
    if st.session_state.nav_history:
        st.session_state.page = st.session_state.nav_history.pop()
    else:
        st.session_state.page = "welcome"
    st.rerun()

def back_arrow():
    cols = st.columns([0.1, 0.9])
    with cols[0]:
        if st.button("⬅️ Back", use_container_width=True, key=f"back_{st.session_state.page}"):
            back()

def go_to_forgot_password(role: str):
    """Remember originating role and navigate to forgot password page."""
    st.session_state.fp_role = role
    go_to("forgot_password")

# ---------- Logs helper ----------
def add_log(username, role, action):
    now = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"username": username or "-", "role": role or "-", "action": action, "timestamp": now, "device": "Web"}
    st.session_state.db["logs"] = pd.concat([st.session_state.db["logs"], pd.DataFrame([row])], ignore_index=True)

def _unread_count():
    """Return count of unread notifications."""
    df = st.session_state.notifications
    return int((~df["read"]).sum()) if not df.empty else 0

# ---------- Visual helpers ----------
def show_logo(center=True, big=False):
    """Render the logo image if available; otherwise, styled text. Safe for center/left usage."""
    logo_candidates = [
        "RecoverAI_Logo.png",
        os.path.join(os.path.dirname(__file__), "RecoverAI_Logo.png")
    ]
    logo_path = next((p for p in logo_candidates if os.path.exists(p)), None)

    target = st.columns([1,2,1])[1] if center else st

    if logo_path:
        target.image(logo_path, use_container_width=False, width=420 if big else 260)
    else:
        target.markdown(
            f"<div style='text-align:{'center' if center else 'left'};"
            f"color:{PRIMARY};font-size:{'52px' if big else '36px'};font-weight:700'>"
            "RecoverAI</div>",
            unsafe_allow_html=True
        )

# ---------- Header Logo (Profile icon + Notification + Language + Centered Logo) ----------
def header_logo(show_bell: bool = True, show_lang: bool = True, show_profile_icon: bool = True):
    """Header with tiny profile icon, notifications, language switcher (left) and centered logo."""
    st.markdown(
        """
        <style>
          /* tighten top padding */
          div.block-container { padding-top: 6px !important; }

          /* centered header logo */
          .ra-header img {
              height: 150px;
              width: auto;
              display: block;
              margin: 0 auto;
          }

          /* thin divider directly under header */
          .ra-divider {
              height: 1px;
              background: #e3e6eb;
              margin: 2px 0 12px;
              width: 100%;
          }

          /* left controls: make buttons compact and same height */
          .header-left .stButton>button{
              background: var(--accent) !important;
              color: #fff !important;
              padding: 6px 12px !important;
              border-radius: 10px !important;
              height: 36px;
          }

          /* tiny profile icon */
          .profile-icon-small {
              width: 28px !important;
              height: 28px !important;
              border-radius: 50%;
              object-fit: cover;
              display: block;
              margin-top: 4px;  /* align with buttons */
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # layout: left controls | centered logo | right spacer
    left, center, right = st.columns([0.24, 0.52, 0.24])

    # ---------- LEFT: profile icon + bell + language (all on one row) ----------
    with left:
        st.markdown('<div class="header-left">', unsafe_allow_html=True)
        # three small columns for inline layout
        ico_col, bell_col, lang_col = st.columns([0.22, 0.39, 0.39])

        # profile icon (no action)
        with ico_col:
            if show_profile_icon and os.path.exists("user_icon.png"):
                import base64
                with open("user_icon.png", "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                st.markdown(
                    f'<img class="profile-icon-small" src="data:image/png;base64,{b64}" />',
                    unsafe_allow_html=True,
                )
            else:
                st.write("")

        # notification button
        with bell_col:
            if show_bell:
                unread = _unread_count()
                bell_label = f"🔔 {unread}" if unread else "🔔"
                if st.button(bell_label, key="header_notify", help="Notifications", use_container_width=True):
                    st.session_state.page = "notifications"
                    st.rerun()
            else:
                st.write("")

        # language switch button (label: عربي if current EN, else English)
        with lang_col:
            if show_lang:
                current = st.session_state.get("lang", "EN")
                lang_label = "عربي🌐" if current == "EN" else "English🌐"
                if st.button(lang_label, key="lang_switch", use_container_width=True):
                    st.session_state["lang"] = "AR" if current == "EN" else "EN"
                    st.rerun()
            else:
                st.write("")

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- CENTER: logo ----------
    logo_file = "header_logo.png"
    if os.path.exists(logo_file):
        import base64
        with open(logo_file, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        center.markdown(
            f'<div class="ra-header"><img src="data:image/png;base64,{b64}" alt="RecoverAI"/></div>',
            unsafe_allow_html=True,
        )
    else:
        center.markdown(
            f"<h3 style='text-align:center;color:{PRIMARY};font-weight:700;margin:0;'>RecoverAI</h3>",
            unsafe_allow_html=True,
        )

    # divider tight under header
    st.markdown('<div class="ra-divider"></div>', unsafe_allow_html=True)



def notifications_page():
    header_logo()
    back_arrow()
    st.markdown("## Notifications")

    df = st.session_state.notifications.copy()

    if df.empty:
        st.info("No notifications.")
        return

    # Sort newest first
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    st.caption("Your latest system notifications:")

    st.data_editor(
        df[["timestamp", "text"]],   # Only show time and message
        hide_index=True,
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Time"),
            "text": st.column_config.TextColumn("Notification"),
        },
        disabled=True,  # 🔒 Fully read-only
        height=320
    )



def sign_out():
    add_log(st.session_state.auth.get("email"), st.session_state.auth.get("role"), "sign_out")
    st.session_state.auth = {"role": None, "email": None, "name": None}
    st.session_state.nav_history.clear()
    st.session_state.page = "welcome"
    st.rerun()

# ---------- Forgot Password Page ----------
def forgot_password_page():
    role = st.session_state.get("fp_role", "patient")
    header_logo(show_bell=False, show_lang=False, show_profile_icon=False)
    back_arrow()

    st.markdown("## Forgot Password")
    st.caption("Enter your email (or ID), then the verification code, and create a new password.")

    # Email Field (Text)
    email = st.text_input("Email / User ID *", placeholder="e.g., admin@recover.ai")

    # Send verification code
    if st.button("Send Verification Code"):
        if not email:
            st.error("Please enter your email (or user ID).")
        else:
            code = str(random.randint(100000, 999999))
            st.session_state.reset_codes[email] = {"code": code, "role": role}
            st.info(f"Verification code sent: **{code}**")

    # Verification Code Field (Number input)
    vcode = st.text_input("Verification Code *", placeholder="6-digit code", max_chars=6)

    # New Password Field / Confirm New Password Field
    new_pw = st.text_input("New Password *", type="password")
    confirm_pw = st.text_input("Confirm New Password *", type="password")

    # Reset Password Button (NO 'Back to Sign-In' button)
    submit = st.button("Reset Password", type="primary")

    if submit:
        if not email:
            st.error("Please enter your email / user ID.")
            return
        if not vcode:
            st.error("Please enter your verification code.")
            return
        if new_pw != confirm_pw:
            st.error("New password and confirmation do not match.")
            return

        entry = st.session_state.reset_codes.get(email)
        if not entry:
            st.error("No verification request found for this email. Click 'Send Verification Code' first.")
            return
        if entry["code"] != vcode or entry["role"] != role:
            st.error("Invalid verification code.")
            return

        # Save new password (demo, in-memory)
        st.session_state.passwords.setdefault(role, {})
        st.session_state.passwords[role][email] = new_pw
        del st.session_state.reset_codes[email]

        st.success("Password reset successful. You can sign in now.")
        dest = {"admin": "admin_signin", "clinician": "clinician_signin", "patient": "patient_signin"}.get(role, "welcome")
        if st.button("Go to Sign-In", type="secondary"):
            go_to(dest)

# ---------- Landing (First) Page ----------
def first_page():
    # Keep content at the top
    st.markdown(
        """
        <style>
            section.main > div {
                padding-top: 16px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # One centered column for both logo and button
    left, center, right = st.columns([1, 2, 1])
    with center:
        show_logo(center=True, big=True)
        st.write("")  # spacing before button
        if st.button("Let’s Start ▶", type="primary", use_container_width=True):
            go_to("welcome")

# ---------- Welcome Page ----------
def welcome_page():
    header_logo(show_bell=False, show_lang=False, show_profile_icon=False)
    back_arrow()
    st.markdown("## Welcome to RecoverAI")

    # 🌸 Custom style for the three role buttons only
    st.markdown("""
    <style>
    .role-buttons .stButton>button {
        background-color: #fa9b93 !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        transition: 0.2s ease-in-out;
        padding: 0.6em 0 !important;
    }
    .role-buttons .stButton>button:hover {
        background-color: #f68e86 !important;
        transform: scale(1.03);
    }
    </style>
    """, unsafe_allow_html=True)

    # 🌟 Button layout
    st.markdown('<div class="role-buttons">', unsafe_allow_html=True)
    st.markdown("###### Choose your role to continue:")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Admin", use_container_width=True):
            go_to("admin_signin")
    with c2:
        if st.button("Clinician", use_container_width=True):
            go_to("clinician_signin")
    with c3:
        if st.button("Patient", use_container_width=True):
            go_to("patient_signin")
    st.markdown('</div>', unsafe_allow_html=True)


# ---------- Admin Flow ----------
def admin_sidebar():
    with st.sidebar:
        st.markdown("### Admin Menu")
        if st.button("⬅️ Back to Welcome", use_container_width=True):
            go_to("welcome")
        choice = st.radio(
            "Navigate",
            ["Access Homepage", "View Profile", "Manage Clinicians", "Manage Patients", "Monitor User Logs", "🔒 Sign Out"],
            label_visibility="collapsed",
        )
    return choice

def admin_signin():
    header_logo(show_bell=False, show_lang=False, show_profile_icon=False)
    back_arrow()
    st.markdown("## Admin Sign-In")

    # -------------------------
    # First-time admin bootstrap (ONLY if no admin exists)
    # -------------------------
    def admin_exists():
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1")
        exists = cur.fetchone() is not None
        conn.close()
        return exists

    if not admin_exists():
        st.warning("No admin account found. Create the first admin account (one time).")

        if st.button("Create default admin (one time)", type="primary"):
            ok, msg = create_user(
                role="admin",
                username="admin",
                email="admin@recover.ai",
                password="@R2003d",
                first_name="Admin",
                last_name="User"
            )
            (st.success if ok else st.error)(msg)

            if ok:
                st.info("Now sign in using: admin / @R2003d")

        st.stop()  # stop rendering the login form until admin exists

    # -------------------------
    # Normal Admin Sign-In (DB)
    # -------------------------
    user_field = st.text_input("User ID / Email *")
    pwd = st.text_input("Password *", type="password")

    col1, col2 = st.columns([1, 3])
    with col1:
        ok = st.button("Sign In", type="primary")

    if ok:
        uid = (user_field or "").strip()

        if not uid or not pwd:
            st.error("Please enter your User ID/Email and Password.")
        else:
            row = get_user_for_login("admin", uid)
            # row = (id, role, username, email, password, first_name, last_name)

            if row and pwd == row[4]:
                st.session_state.auth = {
                    "role": "admin",
                    "email": row[3] or uid,
                    "name": row[5] or "Admin"
                }
                add_log(row[2] or uid, "admin", "sign_in")
                go_to("admin_app")
            else:
                st.error("Invalid ID/Email or Password")

    # Forgot password link
    if st.button("Forgot password?"):
        go_to_forgot_password("admin")

def admin_home():
    header_logo()
    back_arrow()
    st.markdown("## Admin Home Page")
    db = st.session_state.db
    total_users = len(db["patients"]) + len(db["clinicians"])
    active_sessions = np.random.randint(5, 35)
    alerts = np.random.randint(0, 3)

    c1, c2, c3, c4 = st.columns(4)
    for c, title, val in [
        (c1,"Total Users", total_users),
        (c2,"Active Sessions Today", active_sessions),
        (c3,"System Alerts", alerts),
        (c4,"New Users (24h)", np.random.randint(0, 8)),
    ]:
        with c:
            st.markdown(
                f"<div class='soft-card'><div class='pill'>{title}</div>"
                f"<h2 style='margin-top:8px;color:{PRIMARY}'>{val}</h2></div>",
                unsafe_allow_html=True
            )

    st.markdown("#### Trends")
    users_daily = pd.DataFrame({
        "day": pd.date_range(pd.Timestamp.today()-pd.Timedelta(days=14), periods=15),
        "new_users": np.random.poisson(3, 15)
    })
    exercises = pd.DataFrame({
        "day": users_daily["day"],
        "engagement": np.random.randint(40, 95, 15)
    })

    line1 = alt.Chart(users_daily).mark_line(point=True).encode(
        x="day:T", y="new_users:Q"
    ).properties(height=220, title="Daily New Users")
    line2 = alt.Chart(exercises).mark_line(point=True, color=ACCENT).encode(
        x="day:T", y="engagement:Q"
    ).properties(height=220, title="Exercise Engagement (%)")

    c5, c6 = st.columns(2)
    with c5: st.altair_chart(line1, use_container_width=True)
    with c6: st.altair_chart(line2, use_container_width=True)

    # -------------------------
    # User Management (Dropdown Forms)
    # -------------------------
    st.markdown("#### User Management")

    # helper function for selecting dataframe by role
    def _df_for_role(role: str):
        return st.session_state.db["patients"] if role == "patient" else st.session_state.db["clinicians"]

    # ---------- ADD USER ----------
    with st.expander("➕ Add New User"):
        add_role = st.selectbox("Role", ["patient", "clinician"], key="um_add_role")

        if add_role == "clinician":
            cid = st.session_state.get("c_next_id", 1)
            c1, c2, c3 = st.columns(3)
            fn = c1.text_input("First Name")
            ln = c2.text_input("Last Name")
            un = c3.text_input("Username")
            c4, c5, c6 = st.columns(3)
            em = c4.text_input("Email")
            sp = c5.text_input("Specialization")
            bd = c6.date_input("Birthdate")
            status = st.selectbox("Status", ["active", "inactive", "pending"], index=0)
            if st.button("Add Clinician", type="primary", key="um_add_clin"):
                row = {"id": cid, "first": fn, "last": ln, "username": un, "email": em,
                       "birthdate": str(bd), "assigned_patients": 0, "status": status, "specialty": sp}
                st.session_state.db["clinicians"] = pd.concat(
                    [st.session_state.db["clinicians"], pd.DataFrame([row])], ignore_index=True)
                st.session_state.c_next_id = cid + 1
                add_log(un, "clinician", "created_by_admin")
                st.success("Clinician added ✅")

        else:
            pid = st.session_state.get("p_next_id", 1)
            c1, c2, c3 = st.columns(3)
            fn = c1.text_input("First Name", key="um_p_fn")
            ln = c2.text_input("Last Name", key="um_p_ln")
            un = c3.text_input("Username", key="um_p_un")
            c4, c5, c6 = st.columns(3)
            em = c4.text_input("Email", key="um_p_em")
            bd = c5.date_input("Birthdate", key="um_p_bd")
            ac = c6.selectbox("Assign Clinician", st.session_state.db["clinicians"]["username"].dropna().tolist())
            adh = st.slider("Initial adherence (%)", 60, 99, 86)
            if st.button("Add Patient", type="primary", key="um_add_pat"):
                row = {"id": pid, "first": fn, "last": ln, "username": un, "email": em,
                       "birthdate": str(bd), "assigned_clinician": ac, "adherence": adh}
                st.session_state.db["patients"] = pd.concat(
                    [st.session_state.db["patients"], pd.DataFrame([row])], ignore_index=True)
                st.session_state.p_next_id = pid + 1
                add_log(un, "patient", "created_by_admin")
                st.success("Patient added ✅")

    # ---------- DELETE USER ----------
    with st.expander("🗑️ Delete User"):
        del_role = st.selectbox("Role", ["patient", "clinician"], key="um_del_role")
        key_val = st.text_input("Enter ID or Username", key="um_del_key")
        if st.button("Delete", type="primary", key="um_del_btn"):
            df = _df_for_role(del_role)
            if not df.empty:
                mask = (df["id"].astype(str) == key_val) | (df["username"] == key_val)
                if mask.any():
                    uname = df.loc[mask, "username"].iloc[0]
                    st.session_state.db["patients" if del_role == "patient" else "clinicians"] = df.loc[~mask]
                    add_log(uname, del_role, "deleted_by_admin")
                    st.success(f"{del_role.title()} '{key_val}' deleted ✅")
                else:
                    st.warning("User not found.")
            else:
                st.warning("No users available.")

    # ---------- UPDATE USER ----------
    with st.expander("✏️ Update User"):
        upd_role = st.selectbox("Role", ["patient", "clinician"], key="um_upd_role")
        lookup = st.text_input("Enter ID or Username", key="um_upd_lookup")
        df = _df_for_role(upd_role)

        record = None
        idx = None
        if lookup and not df.empty:
            mask = (df["id"].astype(str) == lookup) | (df["username"] == lookup)
            if mask.any():
                idx = df.index[mask][0]
                record = df.loc[idx]

        if lookup and record is None:
            st.warning("User not found.")

        if record is not None:
            st.info(f"Editing {upd_role}: **{record['username']}** (ID: {record['id']})")

            if upd_role == "clinician":
                # --- same style & fields as ADD CLINICIAN ---
                c1, c2, c3 = st.columns(3)
                _id = c1.text_input("ID (read-only)", value=str(record["id"]), disabled=True)
                un = c2.text_input("Username", value=record["username"], key="um_uc_un")
                em = c3.text_input("Email", value=record["email"], key="um_uc_em")

                c4, c5, c6 = st.columns(3)
                fn = c4.text_input("First Name", value=record["first"], key="um_uc_fn")
                ln = c5.text_input("Last Name", value=record["last"], key="um_uc_ln")
                sp = c6.text_input("Specialization", value=record.get("specialty", ""), key="um_uc_sp")

                c7, c8, c9 = st.columns(3)
                status = c7.selectbox(
                    "Status",
                    ["active", "inactive", "pending"],
                    index=["active", "inactive", "pending"].index(record.get("status", "active")),
                    key="um_uc_status"
                )
                bd = c8.date_input(
                    "Birthdate",
                    value=pd.to_datetime(record.get("birthdate", pd.Timestamp.today())).date(),
                    key="um_uc_bd"
                )
                assigned_pats = c9.number_input(
                    "Assigned Patients",
                    min_value=0,
                    value=int(record.get("assigned_patients", 0)),
                    step=1,
                    key="um_uc_assigned"
                )

                if st.button("Save Change", type="primary", key="um_uc_save"):
                    updates = {
                        "username": un,
                        "email": em,
                        "first": fn,
                        "last": ln,
                        "specialty": sp,
                        "status": status,
                        "birthdate": str(bd),
                        "assigned_patients": assigned_pats,
                    }
                    for k, v in updates.items():
                        st.session_state.db["clinicians"].at[idx, k] = v
                    add_log(record["username"], "clinician", "updated_by_admin")
                    st.success("Clinician updated ✅")

            else:
                # --- same style & fields as ADD PATIENT ---
                c1, c2, c3 = st.columns(3)
                _id = c1.text_input("ID (read-only)", value=str(record["id"]), disabled=True)
                un = c2.text_input("Username", value=record["username"], key="um_up_un")
                em = c3.text_input("Email", value=record["email"], key="um_up_em")

                c4, c5, c6 = st.columns(3)
                fn = c4.text_input("First Name", value=record["first"], key="um_up_fn")
                ln = c5.text_input("Last Name", value=record["last"], key="um_up_ln")
                bd = c6.date_input(
                    "Birthdate",
                    value=pd.to_datetime(record.get("birthdate", pd.Timestamp.today())).date(),
                    key="um_up_bd"
                )

                c7, c8 = st.columns(2)
                clinician_list = st.session_state.db["clinicians"]["username"].dropna().tolist()
                cur_assigned = record.get("assigned_clinician", "")
                idx_assigned = clinician_list.index(
                    cur_assigned) if cur_assigned in clinician_list and clinician_list else 0
                ac = c7.selectbox(
                    "Assigned Clinician",
                    options=clinician_list,
                    index=idx_assigned if clinician_list else 0,
                    key="um_up_ac"
                )
                adh = c8.slider(
                    "Adherence (%)",
                    0, 100,
                    int(record.get("adherence", 85)),
                    key="um_up_adh"
                )

                if st.button("Save Change", type="primary", key="um_up_save"):
                    updates = {
                        "username": un,
                        "email": em,
                        "first": fn,
                        "last": ln,
                        "birthdate": str(bd),
                        "assigned_clinician": ac,
                        "adherence": adh,
                    }
                    for k, v in updates.items():
                        st.session_state.db["patients"].at[idx, k] = v
                    add_log(record["username"], "patient", "updated_by_admin")
                    st.success("Patient updated ✅")

    # -------------------------
    # 📣 Announcement Section
    # -------------------------
    st.markdown("#### 📢 Announcements")

    if "show_announcement_form" not in st.session_state:
        st.session_state.show_announcement_form = False

    if st.button("📣 Send Announcement to All", use_container_width=True):
        st.session_state.show_announcement_form = not st.session_state.show_announcement_form

    if st.session_state.show_announcement_form:
        st.markdown("### 📝 Send Announcement")

        with st.form("announcement_form", clear_on_submit=True):
            # NEW FIELD: Announcement Subject
            title = st.text_input("Announcement Subject *", placeholder="Enter Subject...")

            # Message Field
            message = st.text_area(
                "Announcement Message *",
                placeholder="Type your announcement here..."
            )

            send_btn = st.form_submit_button("Send Announcement")

            if send_btn:
                if title.strip() and message.strip():
                    st.session_state.db["announcements"].append({
                        "title": title.strip(),
                        "text": message.strip(),
                        "time": pd.Timestamp.now()
                    })
                    st.success("✅ Announcement sent to all users!")
                    st.session_state.show_announcement_form = False
                else:
                    st.warning("⚠️ Title and message are required.")


    # -------------------------
    # Recent Activity Section
    # -------------------------
    st.markdown("#### Recent Activity")
    st.dataframe(st.session_state.db["logs"].tail(10), use_container_width=True, height=220)


def admin_my_profile():
    header_logo()
    back_arrow()
    st.markdown("## 👤 My Profile")

    prof = st.session_state.admin_profile

    # ---- Editable fields (Inputs) ----
    st.markdown("#### Admin Profile Information")

    e1, e2 = st.columns(2)
    with e1:
        first_name = st.text_input("First Name *", value=prof["first_name"])
    with e2:
        last_name  = st.text_input("Last Name *", value=prof["last_name"])

    e3, e4 = st.columns(2)
    with e3:
        username = st.text_input("Username *", value=prof["username"])
    with e4:
        email    = st.text_input("Email Field *", value=prof["email"])

    phone = st.text_input("Phone Number *", value=prof["phone"], help="Numeric phone (include country code).")

    # ---- Save button + validation ----
    save = st.button("Save Changes", type="primary")
    if save:
        errors = []
        if not first_name.strip(): errors.append("First Name is required.")
        if not last_name.strip():  errors.append("Last Name is required.")
        if not username.strip():   errors.append("Username is required.")
        if not email.strip():      errors.append("Email is required.")
        if not phone.strip():      errors.append("Phone Number is required.")

        # simple email / phone checks
        if email and "@" not in email:
            errors.append("Email format looks invalid.")
        if phone and not any(ch.isdigit() for ch in phone):
            errors.append("Phone Number must contain digits.")

        if errors:
            for e in errors:
                st.error(e)
            st.caption("**Error Validation Message**: Please fix the required fields above.")
        else:
            # Save to session (acts like DB here)
            st.session_state.admin_profile = {
                "first_name": first_name.strip(),
                "last_name":  last_name.strip(),
                "username":   username.strip(),
                "email":      email.strip(),
                "phone":      phone.strip(),
            }
            add_log(username, "admin", "profile_updated")
            st.success("**Success Message**: Profile updated successfully.")


def admin_manage_clinicians():
    header_logo()
    back_arrow()
    st.markdown("## Manage Clinicians")

    df = st.session_state.db["clinicians"].copy()

    # Ensure ID column exists
    if "id" not in df.columns:
        df.insert(0, "id", range(1001, 1001 + len(df)))
        st.session_state.db["clinicians"] = df.copy()
        st.session_state.c_next_id = int(df["id"].max()) + 1

    # ---------- Add New Clinician (Form 1) ----------
    with st.expander("➕ Add New Clinician"):
        c1, c2, c3 = st.columns(3)
        fn = c1.text_input("First Name *")
        ln = c2.text_input("Last Name *")
        un = c3.text_input("Username (for login) *")

        c4, c5, c6 = st.columns(3)
        em = c4.text_input("Email *")
        sp = c5.text_input("Specialization *")
        bd = c6.date_input("Birthdate *")

        if st.button("Add Clinician", type="primary", key="mc_add_btn"):
            cid = st.session_state.c_next_id
            st.session_state.c_next_id += 1

            row = {
                "id": cid,
                "username": un,
                "first": fn,
                "last": ln,
                "email": em,
                "birthdate": str(bd),
                "assigned_patients": 0,
                "status": "pending",
                "specialty": sp
            }

            st.session_state.db["clinicians"] = pd.concat(
                [st.session_state.db["clinicians"], pd.DataFrame([row])],
                ignore_index=True
            )

            add_log(un, "clinician", "created_by_admin")
            st.success("Clinician added.")
            df = st.session_state.db["clinicians"].copy()

    # ---------- 🗑️ Delete Clinician (Form 2) ----------
    with st.expander("🗑️ Delete Clinician"):
        del_key = st.text_input("Enter Clinician ID or Username", key="mc_del_key")
        if st.button("Delete Clinician", type="primary", key="mc_del_btn"):
            df = st.session_state.db["clinicians"].copy()
            if not df.empty:
                mask = (df["id"].astype(str) == del_key) | (df["username"] == del_key)
                if mask.any():
                    uname = df.loc[mask, "username"].iloc[0]
                    st.session_state.db["clinicians"] = df.loc[~mask].reset_index(drop=True)
                    add_log(uname, "clinician", "deleted_by_admin")
                    st.success(f"Clinician '{del_key}' deleted ✅")
                else:
                    st.warning("Clinician not found.")
            else:
                st.warning("No clinicians available.")

    # ---------- ✏️ Update Clinician (Form 3) ----------
    with st.expander("✏️ Update Clinician"):
        lookup = st.text_input("Enter Clinician ID or Username", key="mc_up_lookup")
        df = st.session_state.db["clinicians"].copy()

        record = None
        idx = None
        if lookup and not df.empty:
            mask = (df["id"].astype(str) == lookup) | (df["username"] == lookup)
            if mask.any():
                idx = df.index[mask][0]
                record = df.loc[idx]

        if lookup and record is None:
            st.warning("Clinician not found.")

        if record is not None:
            st.info(f"Editing clinician: **{record['username']}** (ID: {record['id']})")

            c1, c2, c3 = st.columns(3)
            _id = c1.text_input("ID (read-only)", value=str(record["id"]), disabled=True)
            un = c2.text_input("Username", value=record["username"], key="mc_up_un")
            em = c3.text_input("Email", value=record["email"], key="mc_up_em")

            c4, c5, c6 = st.columns(3)
            fn = c4.text_input("First Name", value=record["first"], key="mc_up_fn")
            ln = c5.text_input("Last Name", value=record["last"], key="mc_up_ln")
            sp = c6.text_input("Specialization", value=record.get("specialty", ""), key="mc_up_sp")

            c7, c8, c9 = st.columns(3)
            status = c7.selectbox(
                "Status",
                ["active", "inactive", "pending"],
                index=["active", "inactive", "pending"].index(record.get("status", "pending")),
                key="mc_up_status"
            )
            bd = c8.date_input(
                "Birthdate",
                value=pd.to_datetime(record.get("birthdate", pd.Timestamp.today())).date(),
                key="mc_up_bd"
            )
            assigned_pats = c9.number_input(
                "Assigned Patients",
                min_value=0,
                value=int(record.get("assigned_patients", 0)),
                step=1,
                key="mc_up_assigned"
            )

            if st.button("Save Change", type="primary", key="mc_up_save"):
                updates = {
                    "username": un,
                    "email": em,
                    "first": fn,
                    "last": ln,
                    "specialty": sp,
                    "status": status,
                    "birthdate": str(bd),
                    "assigned_patients": assigned_pats,
                }
                for k, v in updates.items():
                    st.session_state.db["clinicians"].at[idx, k] = v
                add_log(record["username"], "clinician", "updated_by_admin")
                st.success("Clinician updated ✅")

    # ---------- Send Message to Clinician (existing Form) ----------
    with st.expander("📩 Send Message to Clinician"):
        df = st.session_state.db["clinicians"].copy()

        if df.empty:
            st.info("No clinicians available yet. Add a clinician first.")
        else:
            # Dropdown for choosing clinician
            options = df.apply(
                lambda r: f"{r['id']} – {r['first']} {r['last']} ({r['username']})",
                axis=1
            ).tolist()

            selected = st.selectbox("Select Clinician", options)

            subject = st.text_input("Subject *", key="clin_msg_subject")
            body = st.text_area("Message *", height=120, key="clin_msg_body")

            if st.button("Send Message", key="clin_msg_send"):
                if not subject.strip() or not body.strip():
                    st.warning("Please fill in both subject and message.")
                else:
                    selected_id = int(selected.split("–")[0].strip())
                    row_sel = df[df["id"] == selected_id].iloc[0]
                    target_username = row_sel["username"]

                    if "clinician_messages" not in st.session_state:
                        st.session_state.clinician_messages = []

                    st.session_state.clinician_messages.append({
                        "to_id": selected_id,
                        "to_username": target_username,
                        "subject": subject.strip(),
                        "body": body.strip(),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        "sender": "admin"
                    })

                    add_log(target_username, "clinician", "message_sent_by_admin")
                    st.success(f"Message sent to {row_sel['first']} {row_sel['last']}.")

    # ---------- Search + Status Filter (existing table) ----------
    st.markdown("### Browse Clinicians")
    df = st.session_state.db["clinicians"].copy()

    f1, f2 = st.columns([2, 1])
    with f1:
        q = st.text_input("Search by ID", placeholder="e.g., 1023").strip()
    with f2:
        status_opts = ["All"] + sorted(df["status"].dropna().unique().tolist())
        status_sel = st.selectbox("Status", status_opts, index=0)

    filt = df.copy()

    if q:
        mask = filt["id"].astype(str).str.contains(q, case=False, na=False)
        filt = filt[mask]

    if status_sel != "All":
        filt = filt[filt["status"].fillna("").str.lower() == status_sel.lower()]

    ordered = ["id", "username", "first", "last", "email"]
    ordered += [c for c in filt.columns if c not in ordered]

    st.caption(f"Showing {len(filt)} of {len(df)} clinicians")
    st.dataframe(filt[ordered], use_container_width=True, hide_index=True)



def admin_manage_patients():
    header_logo()
    back_arrow()
    st.markdown("## Manage Patients")

    df = st.session_state.db["patients"].copy()

    # Ensure ID exists
    if "id" not in df.columns:
        df.insert(0, "id", range(2001, 2001 + len(df)))
        st.session_state.db["patients"] = df.copy()
        st.session_state.p_next_id = int(df["id"].max()) + 1

    # ---------- Add New Patient (Form 1) ----------
    with st.expander("➕ Add New Patient"):
        c1, c2, c3 = st.columns(3)
        fn = c1.text_input("First Name *")
        ln = c2.text_input("Last Name *")
        un = c3.text_input("Username (for login) *")

        c4, c5, c6 = st.columns(3)
        em = c4.text_input("Email *")
        bd = c5.date_input("Birthdate *")
        ac = c6.selectbox(
            "Assign Clinician *",
            options=st.session_state.db["clinicians"]["username"].tolist()
            if not st.session_state.db["clinicians"].empty else []
        )

        if st.button("Add Patient", type="primary", key="mp_add_btn"):
            pid = st.session_state.p_next_id
            st.session_state.p_next_id += 1

            row = {
                "id": pid,
                "username": un,
                "first": fn,
                "last": ln,
                "email": em,
                "birthdate": str(bd),
                "assigned_clinician": ac,
                "adherence": np.random.randint(60, 95),
            }

            st.session_state.db["patients"] = pd.concat(
                [st.session_state.db["patients"], pd.DataFrame([row])],
                ignore_index=True
            )
            add_log(un, "patient", "created_by_admin")
            st.success("Patient added.")
            df = st.session_state.db["patients"].copy()

    # ---------- 🗑️ Delete Patient (Form 2) ----------
    with st.expander("🗑️ Delete Patient"):
        del_key = st.text_input("Enter Patient ID or Username", key="mp_del_key")
        if st.button("Delete Patient", type="primary", key="mp_del_btn"):
            df = st.session_state.db["patients"].copy()
            if not df.empty:
                mask = (df["id"].astype(str) == del_key) | (df["username"] == del_key)
                if mask.any():
                    uname = df.loc[mask, "username"].iloc[0]
                    st.session_state.db["patients"] = df.loc[~mask].reset_index(drop=True)
                    add_log(uname, "patient", "deleted_by_admin")
                    st.success(f"Patient '{del_key}' deleted ✅")
                else:
                    st.warning("Patient not found.")
            else:
                st.warning("No patients available.")

    # ---------- ✏️ Update Patient (Form 3) ----------
    with st.expander("✏️ Update Patient"):
        lookup = st.text_input("Enter Patient ID or Username", key="mp_up_lookup")
        df = st.session_state.db["patients"].copy()

        record = None
        idx = None
        if lookup and not df.empty:
            mask = (df["id"].astype(str) == lookup) | (df["username"] == lookup)
            if mask.any():
                idx = df.index[mask][0]
                record = df.loc[idx]

        if lookup and record is None:
            st.warning("Patient not found.")

        if record is not None:
            st.info(f"Editing patient: **{record['username']}** (ID: {record['id']})")

            c1, c2, c3 = st.columns(3)
            _id = c1.text_input("ID (read-only)", value=str(record["id"]), disabled=True)
            un = c2.text_input("Username", value=record["username"], key="mp_up_un")
            em = c3.text_input("Email", value=record["email"], key="mp_up_em")

            c4, c5, c6 = st.columns(3)
            fn = c4.text_input("First Name", value=record["first"], key="mp_up_fn")
            ln = c5.text_input("Last Name", value=record["last"], key="mp_up_ln")
            bd = c6.date_input(
                "Birthdate",
                value=pd.to_datetime(record.get("birthdate", pd.Timestamp.today())).date(),
                key="mp_up_bd"
            )

            c7, c8 = st.columns(2)
            clinician_list = st.session_state.db["clinicians"]["username"].dropna().tolist()
            cur_assigned = record.get("assigned_clinician", "")
            idx_assigned = clinician_list.index(cur_assigned) if cur_assigned in clinician_list and clinician_list else 0
            ac = c7.selectbox(
                "Assigned Clinician",
                options=clinician_list,
                index=idx_assigned if clinician_list else 0,
                key="mp_up_ac"
            )
            adh = c8.slider(
                "Adherence (%)",
                0, 100,
                int(record.get("adherence", 85)),
                key="mp_up_adh"
            )

            if st.button("Save Change", type="primary", key="mp_up_save"):
                updates = {
                    "username": un,
                    "email": em,
                    "first": fn,
                    "last": ln,
                    "birthdate": str(bd),
                    "assigned_clinician": ac,
                    "adherence": adh,
                }
                for k, v in updates.items():
                    st.session_state.db["patients"].at[idx, k] = v
                add_log(record["username"], "patient", "updated_by_admin")
                st.success("Patient updated ✅")

    # ---------- Send Message to Patient (existing Form) ----------
    with st.expander("📩 Send Message to Patient"):
        df = st.session_state.db["patients"].copy()  # refresh to latest

        if df.empty:
            st.info("No patients available yet. Add a patient first.")
        else:
            # Dropdown for choosing patient
            options = df.apply(
                lambda r: f"{r['id']} – {r['first']} {r['last']} ({r['username']})",
                axis=1
            ).tolist()
            selected = st.selectbox("Select Patient", options)

            subject = st.text_input("Subject *", key="patient_msg_subject")
            body = st.text_area("Message *", height=120, key="patient_msg_body")

            if st.button("Send Message", key="patient_msg_send"):
                if not subject.strip() or not body.strip():
                    st.warning("Please fill in both subject and message.")
                else:
                    selected_id = int(selected.split("–")[0].strip())
                    row_sel = df[df["id"] == selected_id].iloc[0]
                    target_username = row_sel["username"]

                    # Store messages in session_state (can be used later in patient interface)
                    if "patient_messages" not in st.session_state:
                        st.session_state.patient_messages = []

                    st.session_state.patient_messages.append({
                        "to_id": selected_id,
                        "to_username": target_username,
                        "subject": subject.strip(),
                        "body": body.strip(),
                        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                        "sender": "admin"
                    })

                    add_log(target_username, "patient", "message_sent_by_admin")
                    st.success(f"Message sent to {row_sel['first']} {row_sel['last']}.")

    # ---------- Browse Patients (existing table) ----------
    st.markdown("### Browse Patients")

    df = st.session_state.db["patients"].copy()

    f1, f2 = st.columns([2, 1])
    with f1:
        q = st.text_input("Search by Patient ID", placeholder="e.g., 2005").strip()
    with f2:
        assigned_opts = ["All"] + sorted(
            st.session_state.db["clinicians"]["username"].dropna().unique().tolist()
        )
        assigned_sel = st.selectbox("Assigned Clinician", assigned_opts, index=0)

    filt = df.copy()

    # ✅ Search ONLY by ID
    if q:
        mask = filt["id"].astype(str).str.contains(q, case=False, na=False)
        filt = filt[mask]

    if assigned_sel != "All":
        filt = filt[filt["assigned_clinician"].fillna("") == assigned_sel]

    ordered = ["id", "username", "first", "last", "email"]
    ordered += [c for c in filt.columns if c not in ordered]

    st.caption(f"Showing {len(filt)} of {len(df)} patients")
    st.dataframe(filt[ordered], use_container_width=True, hide_index=True)




def admin_user_logs():
    header_logo()
    back_arrow()
    st.markdown("## Viewing & Filtering User Logs")

    logs = st.session_state.db["logs"].copy()

    # ---------------- Filters ----------------
    c1, c2, c3 = st.columns(3)
    with c1:
        role = st.selectbox("User Role", ["All", "patient", "clinician", "admin"])
    with c2:
        activity = st.text_input("Activity contains...")
    with c3:
        days = st.slider("Days range", 1, 60, 14)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)

    filt = logs.copy()
    if role != "All":
        filt = filt[filt["role"] == role]
    if activity:
        filt = filt[filt["action"].str.contains(activity, case=False, na=False)]

    if not filt.empty:
        filt["timestamp"] = pd.to_datetime(filt["timestamp"], errors="coerce")
        filt = filt[filt["timestamp"] >= cutoff]

    # ---------------- Charts Section ----------------
    if filt.empty:
        st.info("No logs found for the selected filters.")
    else:
        st.markdown("### Log Analytics Overview")

        # ------------------------------------------
        # 1) Daily activity trend (with demo filler)
        # ------------------------------------------
        daily = filt.copy()
        # keep as datetime64 (normalized to midnight)
        daily["date"] = daily["timestamp"].dt.normalize()

        # Aggregate real values
        daily_counts = (
            daily.groupby("date")
            .size()
            .reset_index(name="count")
            .sort_values("date")
        )

        # If not enough data, auto-generate demo activity for more dates
        min_points = 7  # at least ~1 week visible
        if daily_counts.shape[0] < min_points:
            today = pd.Timestamp.today().normalize()
            start_date = today - pd.Timedelta(days=min_points - 1)
            all_days = pd.date_range(start=start_date, end=today, freq="D")

            full = pd.DataFrame({"date": all_days})

            # Both 'full.date' and 'daily_counts.date' are datetime64[ns]
            merged = full.merge(daily_counts, on="date", how="left")

            # 🔧 FIX: fill NaNs using a mask instead of passing ndarray to fillna
            mask = merged["count"].isna()
            if mask.any():
                merged.loc[mask, "count"] = np.random.randint(1, 6, size=mask.sum())

            daily_counts = merged

        # ------------------------------------
        # 2) Logs by role
        # ------------------------------------
        role_counts = (
            filt["role"]
            .value_counts()
            .reset_index(name="count")
        )
        role_counts.columns = ["role", "count"]

        # ------------------------------------
        # 3) Most common actions
        # ------------------------------------
        action_counts = (
            filt["action"]
            .value_counts()
            .reset_index(name="count")
        )
        action_counts.columns = ["action", "count"]

        # --- Layout for charts ---
        col1, col2 = st.columns(2)

        # Chart 1: Daily activity line chart
        with col1:
            trend_chart = (
                alt.Chart(daily_counts)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("count:Q", title="Log Count"),
                    tooltip=["date", "count"],
                )
                .properties(
                    title="Daily Activity Trend",
                    height=260,
                )
            )
            st.altair_chart(trend_chart, use_container_width=True)

        # Chart 2: Logs by role (pie chart)
        with col2:
            if not role_counts.empty:
                role_chart = (
                    alt.Chart(role_counts)
                    .mark_arc()
                    .encode(
                        theta="count:Q",
                        color=alt.Color("role:N", title="Role"),
                        tooltip=["role", "count"],
                    )
                    .properties(
                        title="Log Distribution by Role",
                        height=260,
                    )
                )
                st.altair_chart(role_chart, use_container_width=True)
            else:
                st.caption("No role distribution data for this filter.")

        # Chart 3: Most common actions
        st.markdown("### Most Common Actions")
        if not action_counts.empty:
            action_chart = (
                alt.Chart(action_counts)
                .mark_bar()
                .encode(
                    x=alt.X("action:N", title="Action"),
                    y=alt.Y("count:Q", title="Count"),
                    tooltip=["action", "count"],
                )
                .properties(
                    height=260,
                )
            )
            st.altair_chart(action_chart, use_container_width=True)
        else:
            st.caption("No action statistics for this filter.")

    # ---------------- Actions (Download / Clear / Archive) ----------------
    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    with c4:
        st.download_button(
            "⬇ Download CSV",
            data=filt.to_csv(index=False),
            file_name="recoverai_logs.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c5:
        if st.button("🧹 Clear Old Logs", use_container_width=True):
            keep = pd.Timestamp.now() - pd.Timedelta(days=7)
            st.session_state.db["logs"] = logs[
                pd.to_datetime(logs["timestamp"], errors="coerce") >= keep
            ]
            st.info("Old logs cleared (kept 7 days).")
    with c6:
        st.button("🗃️ Archive", use_container_width=True)




def admin_app():
    choice = admin_sidebar()
    if choice == "Access Homepage":
        admin_home()
    elif choice == "View Profile":
        admin_my_profile()
    elif choice == "Manage Clinicians":
        admin_manage_clinicians()
    elif choice == "Manage Patients":
        admin_manage_patients()
    elif choice == "Monitor User Logs":
        admin_user_logs()
    elif choice == "🔒 Sign Out":
        sign_out()


# ---------- Clinician Flow ----------
def clinician_sidebar():
    with st.sidebar:
        st.markdown("### Clinician Menu")
        if st.button("⬅️ Back to Welcome", use_container_width=True):
            go_to("welcome")

        choice = st.radio(
            "Navigate",
            [
                "Access Homepage",
                "View Profile",
                "View My Patients",
                "Generate Reports",
                "Chat My Patient",
                "🔒 Sign Out",
            ],
            label_visibility="collapsed",
        )
    return choice




def clinician_signin():
    header_logo(show_bell=False, show_lang=False, show_profile_icon=False)
    back_arrow()
    st.markdown("## Clinician Sign-In")
    email = st.text_input("User ID / Email *")
    pwd = st.text_input("Password *", type="password")
    if st.button("Sign In", type="primary"):
        uid = (email or "").strip()
        row = get_user_for_login("clinician", uid)

        if row and pwd == row[4]:
            st.session_state.auth = {"role":"clinician","email": row[3] or uid, "name": row[5] or "Clinician"}
            add_log(uid, "clinician", "sign_in")
            go_to("clinician_app")
        else:
            st.error("Invalid ID/Email or Password")


    if st.button("Forgot password?"):
        go_to_forgot_password("clinician")

def clinician_home():
    header_logo()
    back_arrow()
    st.markdown("## Welcome, clinician!")
    st.info("Notifications: 2 new messages • 1 patient missed a session")

    # --- Quick navigation buttons (same format as patient welcome) ---
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View Profile", type="primary", use_container_width=True):
            st.session_state.clinician_menu_default = "View Profile"
            go_to("clinician_app")
    with c2:
        if st.button("View My Patients", type="primary", use_container_width=True):
            st.session_state.clinician_menu_default = "View My Patients"
            go_to("clinician_app")
    with c3:
        if st.button("Generate Reports", type="primary", use_container_width=True):
            st.session_state.clinician_menu_default = "Generate Reports"
            go_to("clinician_app")

    st.markdown("---")

    # --- Existing dashboard tables ---
    p = st.session_state.db["patients"]
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("#### Active Patients")
        st.dataframe(
            p[["first", "last", "username", "assigned_clinician", "adherence"]],
            use_container_width=True,
            height=250,
        )
    with s2:
        st.markdown("#### Alerts")
        alerts = p[p["adherence"] < 75][["username", "adherence"]]
        st.dataframe(
            alerts if not alerts.empty else pd.DataFrame(columns=["username", "adherence"]),
            use_container_width=True,
            height=250,
        )


def clinician_profile():
    header_logo()
    back_arrow()
    st.markdown("## 👤 My Profile")

    # ---- Load current clinician row from the DB (match by email if possible)
    cdf = st.session_state.db["clinicians"].copy()
    cur_email = (st.session_state.auth or {}).get("email", "")
    idx = None
    row = None
    if not cdf.empty and "email" in cdf.columns:
        mask = cdf["email"].astype(str) == str(cur_email)
        if mask.any():
            idx = cdf[mask].index[0]
            row = cdf.loc[idx].to_dict()

    # Fallback demo row if not found
    if row is None:
        row = {
            "id":  st.session_state.get("c_next_id", 1001) - 1,
            "first": "Clinician",
            "last":  "Name",
            "username": "clin_user",
            "email":  cur_email or "clinician@recover.ai",
            # optional fields may not exist yet in your DF
            "phone": "",
            "gender": "",
            "age": "",
        }

    # Ensure optional columns exist in the DataFrame for saving
    for col in ["phone", "gender", "age"]:
        if col not in cdf.columns:
            cdf[col] = ""

    col1, col2 = st.columns(2)

    # ---- Read-only labels (as disabled inputs for consistent look)
    with col1:
        st.text_input("First Name", value=str(row.get("first", "")), disabled=True)
        st.text_input("Last Name",  value=str(row.get("last", "")),  disabled=True)
        st.text_input("ID",         value=str(row.get("id", "")),    disabled=True)

    # ---- Editable fields
    with col2:
        phone  = st.text_input("Phone Number", value=str(row.get("phone", "")))
        gender = st.text_input("Gender",       value=str(row.get("gender", "")))
        birthdate = st.date_input(
            "Birthdate",
            value=pd.to_datetime(row.get("birthdate", "1990-01-01")).date()
        )
        email  = st.text_input("Email *",      value=str(row.get("email", "")))

    st.caption("Fields shown as disabled (First Name, Last Name, ID) are read-only.")

    if st.button("Save Changes", type="primary"):
        # If the user existed in the table, update in place; else append a new row.
        updated = {
            "phone": phone,
            "gender": gender,
            "birthdate": str(birthdate),
            "email": email,
        }
        if idx is not None:
            for k, v in updated.items():
                cdf.at[idx, k] = v
        else:
            # Create a new record on the fly (demo)
            new_row = row.copy()
            new_row.update(updated)
            cdf = pd.concat([cdf, pd.DataFrame([new_row])], ignore_index=True)

        st.session_state.db["clinicians"] = cdf
        st.success("Profile updated.")


def clinician_patients():
    header_logo()
    back_arrow()
    st.markdown("## My Patients")

    # --- Load & ensure ID column exists ---
    p = st.session_state.db["patients"].copy()
    if "id" not in p.columns:
        p["id"] = range(2001, 2001 + len(p))
        st.session_state.db["patients"] = p.copy()
        st.session_state.p_next_id = int(p["id"].max()) + 1

    # --- Ensure demo fields exist (injury, duration, exercises) ---
    changed = False
    if "injury" not in p.columns:
        demo_injuries = [
            "Shoulder impingement",
            "ACL reconstruction",
            "Lower back pain",
            "Ankle sprain",
            "Post-hip surgery",
        ]
        p["injury"] = [demo_injuries[i % len(demo_injuries)] for i in range(len(p))]
        changed = True

    if "program_duration_weeks" not in p.columns:
        p["program_duration_weeks"] = 8  # default 8-week program
        changed = True

    if "assigned_exercises" not in p.columns:
        default_plan = "Shoulder Mobility; Hamstring Stretch; Hip Activation"
        p["assigned_exercises"] = default_plan
        changed = True

    if changed:
        st.session_state.db["patients"] = p.copy()

    # --- Search ONLY by patient ID ---
    q = st.text_input("Search by Patient ID", placeholder="e.g., 2005").strip()
    p_view = p.copy()
    if q:
        p_view = p_view[p_view["id"].astype(str).str.contains(q, case=False, na=False)]

    # --- Main table ---
    base_order = ["id", "username", "first", "last", "email"]
    ordered_cols = [c for c in base_order if c in p_view.columns] + [
        c for c in p_view.columns if c not in base_order
    ]

    st.caption(f"Showing {len(p_view)} of {len(p)} patients")
    st.dataframe(p_view[ordered_cols], use_container_width=True, height=280)

    # ===============================
    # 🔍 Select a patient to view details
    # ===============================
    st.markdown("#### Select a patient to view details")

    usernames = ["—"] + (
        p_view["username"].dropna().tolist() if "username" in p_view.columns else []
    )
    sel = st.selectbox("Patient", usernames)

    if sel != "—":
        row = p[p["username"] == sel].iloc[0]

        full_name = f"{row.get('first', '')} {row.get('last', '')}".strip()
        injury = row.get("injury", "—")
        duration_weeks = row.get("program_duration_weeks", "—")
        assigned_clinician = row.get("assigned_clinician", "—")
        adherence = row.get("adherence", "—")
        email = row.get("email", "—")
        birthdate = row.get("birthdate", "—")
        pid = row.get("id", "—")

        exercises_str = str(row.get("assigned_exercises", "") or "")
        exercises_list = [e.strip() for e in exercises_str.split(";") if e.strip()]

        st.markdown("### Patient Details")

        # ✔ Contact patient button
        link_col, _ = st.columns([1, 3])
        with link_col:
            if st.button("💬 Contact this patient", key="contact_patient"):
                st.session_state.selected_chat_patient = sel
                st.session_state.goto_chat_patient = True
                st.rerun()

        d1, d2 = st.columns([2, 1])

        with d1:
            st.markdown(
                f"""
                <div style="
                    background: #ffffff;
                    border-radius: 16px;
                    border: 1px solid #e4e7ef;
                    padding: 14px 18px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
                ">
                  <h4 style="margin:0 0 6px 0; color:{PRIMARY};">{full_name or sel}</h4>
                  <p style="margin:0; font-size:13px; color:#555;">
                    <b>ID:</b> {pid} &nbsp;•&nbsp;
                    <b>Username:</b> {sel}<br/>
                    <b>Email:</b> {email}<br/>
                    <b>Birthdate:</b> {birthdate}<br/>
                    <b>Assigned Clinician:</b> {assigned_clinician}
                  </p>
                  <p style="margin:6px 0 0 0; font-size:13px; color:#444;">
                    <b>Injury / Condition:</b> {injury}<br/>
                    <b>Program Duration:</b> {duration_weeks} weeks
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d2:
            st.markdown("**Quick Stats**")
            st.metric("Adherence", f"{adherence}%" if adherence != "—" else "—")
            st.metric("Exercises Assigned", len(exercises_list) if exercises_list else 0)

        if exercises_list:
            st.markdown("#### Assigned Exercises")
            ex_rows = []
            for ex in exercises_list:
                ex_rows.append(
                    {
                        "Exercise": ex,
                        "Sets × Reps": "2 × 10",
                        "Frequency": "3x / week",
                    }
                )
            ex_df = pd.DataFrame(ex_rows)
            st.table(ex_df)




def clinician_chat_my_patient():
    header_logo()
    back_arrow()
    st.markdown("## Chat My Patient")

    # --- Load patients ---
    p = st.session_state.db["patients"].copy()
    if p.empty:
        st.info("No patients available to chat with yet.")
        return

    # ---- Build selector list with default empty selection ----
    options = ["—"] + p.apply(
        lambda r: f"{r['id']} – {r['first']} {r['last']} ({r['username']})",
        axis=1,
    ).tolist()

    selected = st.selectbox("Select Patient", options)

    # If no patient selected -> show hint and stop
    if selected == "—":
        st.info("Please select a patient to start chatting.")
        return

    # Parse patient info
    selected_id = int(selected.split("–")[0].strip())
    row = p[p["id"] == selected_id].iloc[0]
    uname = row["username"]
    full_name = f"{row['first']} {row['last']}"

    # ---- Initialize chat storage per patient ----
    if "clinician_patient_chats" not in st.session_state:
        st.session_state.clinician_patient_chats = {}

    if uname not in st.session_state.clinician_patient_chats:
        st.session_state.clinician_patient_chats[uname] = [
            {
                "role": "patient",
                "text": f"Hi doctor, I’ve just finished today’s {row.get('assigned_exercises', 'exercise')} session.",
            },
            {
                "role": "clinician",
                "text": "Great job! Remember to keep movements slow and controlled. Any pain today?",
            },
        ]

    chat_history = st.session_state.clinician_patient_chats[uname]

    # ---- Layout: left chat + right mini progress ----
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### Chat with {full_name}")

        # Display chat history with clinician avatar
        for msg in chat_history:
            if msg["role"] == "clinician":
                with st.chat_message("assistant", avatar=CLINICIAN_ICON):
                    st.write(msg["text"])
            else:
                with st.chat_message("user"):
                    st.write(msg["text"])

        # Input for new message
        prompt = st.chat_input(f"Write a message to {full_name}…")

        if prompt:
            # Store clinician message
            chat_history.append({"role": "clinician", "text": prompt})

            # Demo auto patient reply
            reply = (
                "Thank you for the update. 🌿\n\n"
                "Please remember to keep your movements slow and controlled. If you feel any pain, stop the exercise and let me know."
            )
            chat_history.append({"role": "patient", "text": reply})

            st.session_state.clinician_patient_chats[uname] = chat_history
            st.rerun()

    with col2:
        st.markdown("### Recent Progress")
        days = pd.date_range(pd.Timestamp.today() - pd.Timedelta(days=10), periods=11)
        acc = np.random.randint(70, 98, len(days))
        chart = (
            alt.Chart(pd.DataFrame({"day": days, "accuracy": acc}))
            .mark_line(point=True)
            .encode(x="day:T", y="accuracy:Q")
            .properties(height=220)
        )
        st.altair_chart(chart, use_container_width=True)



def clinician_report():
    header_logo()
    back_arrow()
    st.markdown("## Generate Reports")

    p = st.session_state.db["patients"].copy()
    if p.empty:
        st.info("No patient data available yet to generate reports.")
        return

    # ---------------------------------------
    # Patient Filter
    # ---------------------------------------
    st.markdown("### Filter Options")

    patient_options = ["All Patients"] + p["username"].tolist()
    selected_patient = st.selectbox("Report For", patient_options)

    # Filter dataset if a specific patient is selected
    if selected_patient != "All Patients":
        p = p[p["username"] == selected_patient]

    # ---------------------------------------
    # Time Period Filter
    # ---------------------------------------
    period = st.selectbox(
        "Filter period",
        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
    )

    days_map = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All time": 60,  # demo window
    }
    n_days = days_map[period]

    # ---------------------------------------
    # Demo Data Generation
    # ---------------------------------------
    end = pd.Timestamp.today().normalize()
    days = pd.date_range(end - pd.Timedelta(days=n_days - 1), end, freq="D")

    records = []
    rng = np.random.default_rng(42)

    for uname in p["username"]:
        vals = rng.integers(65, 100, size=len(days))
        for d, v in zip(days, vals):
            records.append({"day": d, "username": uname, "accuracy": int(v)})

    # If for some reason no rows (edge edge-case), create a tiny demo set
    if not records:
        demo_unames = ["demo_patient_1", "demo_patient_2"]
        for uname in demo_unames:
            vals = rng.integers(65, 100, size=len(days))
            for d, v in zip(days, vals):
                records.append({"day": d, "username": uname, "accuracy": int(v)})

    ts = pd.DataFrame(records)

    # Per-patient average accuracy (real users only)
    per_patient = (
        ts.groupby("username", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "avg_accuracy"})
    )
    per_patient["avg_accuracy"] = per_patient["avg_accuracy"].round(1)

    # Daily average accuracy
    daily_avg = (
        ts.groupby("day", as_index=False)["accuracy"]
        .mean()
        .rename(columns={"accuracy": "avg_accuracy"})
    )

    # Adherence from DB (if available)
    has_adherence = "adherence" in p.columns
    adherence_vals = p["adherence"].astype(float) if has_adherence else None

    st.markdown("---")

    # ============================================================
    # 🔹 Make nicer demo visuals if few patients
    # ============================================================
    per_patient_vis = per_patient.copy()
    unique_patients = per_patient_vis["username"].nunique()

    if unique_patients < 3:
        # Add demo patients only for visualization (not for CSV summary)
        demo_rows = pd.DataFrame({
            "username": ["Demo Patient A", "Demo Patient B"],
            "avg_accuracy": rng.integers(65, 100, size=2)
        })
        per_patient_vis = pd.concat([per_patient_vis, demo_rows], ignore_index=True)

    if has_adherence:
        adherence_df = pd.DataFrame({"adherence": adherence_vals})
        adherence_df["range"] = pd.cut(
            adherence_df["adherence"],
            bins=[0, 50, 70, 85, 100],
            labels=["Low (0–50)", "Medium (50–70)", "Good (70–85)", "High (85–100)"],
        )

        adherence_vis = adherence_df.copy()

        # If only one bucket, add demo values in other ranges for nicer pie
        if adherence_vis["range"].nunique() < 3:
            demo_adherence = pd.DataFrame({"adherence": [55, 78, 92]})
            demo_adherence["range"] = pd.cut(
                demo_adherence["adherence"],
                bins=[0, 50, 70, 85, 100],
                labels=["Low (0–50)", "Medium (50–70)", "Good (70–85)", "High (85–100)"],
            )
            adherence_vis = pd.concat([adherence_vis, demo_adherence], ignore_index=True)
    else:
        adherence_df = None
        adherence_vis = None

    # ============================================================
    # Charts Layout
    #   Row 1: (1) Daily trend  • (2) Per-patient pie
    #   Row 2: (3) Adherence pie • (4) Sessions bar chart
    # ============================================================

    # ---------- Row 1 ----------
    col1, col2 = st.columns(2)

    # 1) Daily Average Accuracy Trend (Line)
    with col1:
        st.markdown("### Daily Average Exercises Accuracy Trend")
        trend_chart = (
            alt.Chart(daily_avg)
            .mark_line(point=True)
            .encode(
                x=alt.X("day:T", title="Date"),
                y=alt.Y(
                    "avg_accuracy:Q",
                    title="Average Exercises Accuracy (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=["day", "avg_accuracy"],
            )
            .properties(height=260)
        )
        st.altair_chart(trend_chart, use_container_width=True)

    # 2) Accuracy Contribution Per Patient (Pie)
    with col2:
        st.markdown("### Accuracy Contribution Per Patient")
        pie_chart_acc = (
            alt.Chart(per_patient_vis)
            .mark_arc()
            .encode(
                theta="avg_accuracy:Q",
                color="username:N",
                tooltip=["username", "avg_accuracy"],
            )
            .properties(height=260)
        )
        st.altair_chart(pie_chart_acc, use_container_width=True)

    # ---------- Row 2 ----------
    col3, col4 = st.columns(2)

    # 3) Adherence Distribution (Pie) – left
    with col3:
        if has_adherence and adherence_vis is not None:
            st.markdown("### Adherence Distribution")
            pie_chart_adherence = (
                alt.Chart(adherence_vis)
                .mark_arc()
                .encode(
                    theta="count():Q",
                    color="range:N",
                    tooltip=["range", "count()"],
                )
                .properties(height=260)
            )
            st.altair_chart(pie_chart_adherence, use_container_width=True)
        else:
            st.markdown("### Adherence Distribution")
            st.caption("No adherence data available yet. Demo will show once adherence is stored.")

    # 4) Sessions Completed vs Missed (Bar Chart) – right
    with col4:
        st.markdown("### Sessions Completed vs Missed")
        sessions_df = pd.DataFrame({
            "status": ["Completed Sessions", "Missed Sessions"],
            "count": [int(rng.integers(25, 60)), int(rng.integers(3, 15))]
        })
        bar_chart_sessions = (
            alt.Chart(sessions_df)
            .mark_bar()
            .encode(
                x=alt.X("status:N", title="Session Status"),
                y=alt.Y("count:Q", title="Number of Sessions"),
                tooltip=["status", "count"],
            )
            .properties(height=260)
        )
        st.altair_chart(bar_chart_sessions, use_container_width=True)

    # ============================================================
    # Download Summary (uses ONLY real per_patient data)
    # ============================================================
    st.markdown("### Download Report")

    summary = per_patient.copy()  # original, no demo rows
    if has_adherence:
        summary = summary.merge(
            p[["username", "adherence"]],
            on="username",
            how="left",
        )

    st.download_button(
        "⬇ Download Patients Report (CSV)",
        data=summary.to_csv(index=False),
        file_name=f"clinician_report_{selected_patient.replace(' ', '_').lower()}_{period.replace(' ', '').lower()}.csv",
        mime="text/csv",
        use_container_width=True,
    )





def clinician_app():
    # ✅ If we came from "Contact this patient" in My Patients, jump directly to Chat My Patient once
    if st.session_state.get("goto_chat_patient"):
        st.session_state.goto_chat_patient = False
        clinician_chat_my_patient()
        return

    choice = clinician_sidebar()

    if choice == "Access Homepage":
        clinician_home()
    elif choice == "View Profile":
        clinician_profile()
    elif choice == "View My Patients":
        clinician_patients()
    elif choice == "Generate Reports":
        clinician_report()
    elif choice == "Chat My Patient":
        clinician_chat_my_patient()
    elif choice == "🔒 Sign Out":
        sign_out()




# ---------- Patient Flow ----------
def patient_sidebar():
    with st.sidebar:
        st.markdown("### Patient Menu")
        
        if st.button("⬅️ Back to Welcome", use_container_width=True):
            go_to("welcome")
        
        choice = st.radio(
            "Navigate",
            [
                "Access Homepage",
                "View Profile",
                "Start My Exercise",
                "My Exercise Progress",   # 🔹 updated name
                "Contact My Clinician",
                "Consult AI",
                "About RecoverAI",
                "🔒 Sign Out"
            ],
            label_visibility="collapsed"
        )
    return choice


def patient_signin():
    header_logo(show_bell=False, show_lang=False, show_profile_icon=False)
    back_arrow()
    st.markdown("## Patient Sign-In / Sign-Up")

    tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

    # -------------------------
    # Sign In (DB)
    # -------------------------
    with tab1:
        identifier = st.text_input("User ID / Email *", key="p_email")
        pwd = st.text_input("Password *", type="password", key="p_pwd")

        if st.button("Sign In", type="primary", key="p_signin"):
            uid = (identifier or "").strip()

            if not uid or not pwd:
                st.error("Please enter your User ID/Email and Password.")
            else:
                row = get_user_for_login("patient", uid)
                # row = (id, role, username, email, password, first_name, last_name)
                if row and pwd == row[4]:
                    full_name = (f"{row[5] or ''} {row[6] or ''}").strip() or "Patient"
                    st.session_state.auth = {
                        "role": "patient",
                        "email": row[3] or uid,
                        "name": full_name
                    }
                    add_log(row[2] or uid, "patient", "sign_in")
                    go_to("patient_app")
                else:
                    st.error("Invalid ID/Email or Password")

        if st.button("Forgot password?", key="p_forgot"):
            go_to_forgot_password("patient")

    # -------------------------
    # Sign Up (DB)
    # -------------------------
    with tab2:
        f = st.text_input("First Name *", key="p_su_first")
        l = st.text_input("Last Name *", key="p_su_last")
        u = st.text_input("Username *", key="p_su_username")
        e = st.text_input("Email *", key="p_su_email")
        b = st.date_input("Birthdate *", key="p_su_birthdate")
        ph = st.text_input("Phone Number *", key="p_su_phone")
        g = st.selectbox("Gender *", ["Male", "Female"], key="p_su_gender")
        pw1 = st.text_input("Create Password *", type="password", key="p_su_pw1")
        pw2 = st.text_input("Confirm Password *", type="password", key="p_su_pw2")

        if st.button("Sign Up", type="primary", key="p_signup_btn"):
            # Basic validation
            errors = []
            if not f.strip(): errors.append("First Name is required.")
            if not l.strip(): errors.append("Last Name is required.")
            if not u.strip(): errors.append("Username is required.")
            if not e.strip(): errors.append("Email is required.")
            if e and "@" not in e: errors.append("Email format looks invalid.")
            if not ph.strip(): errors.append("Phone Number is required.")
            if not pw1: errors.append("Password is required.")
            if pw1 != pw2: errors.append("Passwords do not match.")

            if errors:
                for err in errors:
                    st.error(err)
            else:
                ok, msg = create_user(
                    role="patient",
                    username=u.strip(),
                    email=e.strip(),
                    password=pw1,
                    first_name=f.strip(),
                    last_name=l.strip(),
                    birthdate=str(b),
                    phone=ph.strip(),
                    gender=g
                )
                (st.success if ok else st.error)(msg)

                if ok:
                    add_log(u.strip(), "patient", "sign_up")
                    st.info("Account created. Now go to **Sign In** tab and login.")


def patient_welcome():
    header_logo()
    back_arrow()
    st.markdown("## Patient Homepage")

    # --- Row 3: labels (outputs) ---
    m1, m2 = st.columns(2)
    with m1:
        st.metric("🔥 Streak", value=f"{np.random.randint(1, 12)} days")
    with m2:
        st.metric("🗓️ Days Left", value=f"{np.random.randint(7, 28)}")

    st.write("")
    # --- Row 1: the three required buttons ---
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("View Profile", type="primary", use_container_width=True):
            st.session_state.patient_menu_default = "View Profile"
            go_to("patient_app")
    with c2:
        if st.button("Start My Exercise", type="primary", use_container_width=True):
            st.session_state.patient_menu_default = "Start My Exercise"
            go_to("patient_app")
    with c3:
        if st.button("My Exercise Progress", type="primary", use_container_width=True):
            st.session_state.patient_menu_default = "Generate Reports"
            go_to("patient_app")

    st.write("")
    # --- Row 2: Contact Clinician + AI Assistant buttons ---
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Contact My Clinician", type="primary", use_container_width=True):
            st.session_state.patient_menu_default = "Contact Your Clinician"
            go_to("patient_app")

    with c2:
        if st.button("Consult AI", type="primary", use_container_width=True):
            st.session_state.patient_menu_default = "Chat"
            go_to("patient_app")

def patient_profile():
    header_logo()
    back_arrow()
    st.markdown("## 👤 My Profile")

    # ---- Load patients table ----
    pdf = st.session_state.db["patients"].copy()

    # ---- Try to find current patient from auth email ----
    auth_email = (st.session_state.auth or {}).get("email", "")
    idx = None
    row = None

    if not pdf.empty and "email" in pdf.columns and auth_email:
        mask = pdf["email"].astype(str) == str(auth_email)
        if mask.any():
            idx = pdf[mask].index[0]
            row = pdf.loc[idx].to_dict()

    # ---- Fallback demo row if not found ----
    if row is None:
        # Ensure required columns exist in DF
        for col in ["id", "first", "last", "username", "email", "birthdate", "phone"]:
            if col not in pdf.columns:
                if col == "id":
                    pdf["id"] = range(2001, 2001 + len(pdf))
                else:
                    pdf[col] = ""
        st.session_state.db["patients"] = pdf.copy()

        if not pdf.empty:
            row = pdf.iloc[0].to_dict()
            idx = pdf.index[0]
        else:
            # totally empty table – create a temporary demo structure
            row = {
                "id": 2001,
                "first": "",
                "last": "",
                "username": "",
                "email": auth_email or "",
                "birthdate": pd.Timestamp.today().date(),
                "phone": "",
            }
            idx = None

    # ---- ID (Read-Only) ----
    st.text_input("ID", value=str(row.get("id", "—")), disabled=True)

    c1, c2 = st.columns(2)
    with c1:
        first_name = st.text_input("First Name *", value=str(row.get("first", "")))
        last_name  = st.text_input("Last Name *", value=str(row.get("last", "")))
        username   = st.text_input("Username *", value=str(row.get("username", "")))
    with c2:
        email      = st.text_input("Email *", value=str(row.get("email", "")))
        birthdate  = st.date_input(
            "Birthdate *",
            value=pd.to_datetime(row.get("birthdate", pd.Timestamp.today())).date()
        )
        phone      = st.text_input("Phone Number *", value=str(row.get("phone", "")), placeholder="+966 5XXXXXXXX")

    s1, s2 = st.columns([1, 3])
    with s1:
        save = st.button("Save Changes", type="primary")

    if save:
        errors = []

        # ---- Required fields check ----
        if not first_name.strip(): errors.append("First Name is required.")
        if not last_name.strip():  errors.append("Last Name is required.")
        if not username.strip():   errors.append("Username is required.")
        if not email.strip():      errors.append("Email is required.")
        if not phone.strip():      errors.append("Phone Number is required.")

        # ---- Format checks ----
        if email and "@" not in email:
            errors.append("Email format looks invalid.")
        if phone and not any(ch.isdigit() for ch in phone):
            errors.append("Phone Number must contain digits.")

        # ---- Show errors or save ----
        if errors:
            for e in errors:
                st.error(e)
            st.caption("**Error Validation Message**: Please fix the required fields above.")
        else:
            # Ensure columns exist before saving
            for col in ["first", "last", "username", "email", "birthdate", "phone"]:
                if col not in pdf.columns:
                    pdf[col] = ""

            updated = {
                "first": first_name.strip(),
                "last":  last_name.strip(),
                "username": username.strip(),
                "email": email.strip(),
                "birthdate": str(birthdate),
                "phone": phone.strip(),
            }

            if idx is not None:
                for k, v in updated.items():
                    pdf.at[idx, k] = v
            else:
                # append as new patient record
                pdf = pd.concat([pdf, pd.DataFrame([updated])], ignore_index=True)

            st.session_state.db["patients"] = pdf
            add_log(username, "patient", "profile_updated")
            st.success("**Success Message**: Profile updated successfully.")


def run_pose_estimation(image_file):
    """Run pose estimation on a Streamlit camera_input image and return annotated image + score."""
    # Read image from UploadedFile
    img = Image.open(image_file)
    img_rgb = np.array(img)  # PIL -> numpy (RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:
        results = pose.process(img_rgb)

        annotated = img_rgb.copy()
        score = 0

        if results.pose_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # Simple demo "score" based on average visibility of landmarks
            visibilities = [lm.visibility for lm in results.pose_landmarks.landmark]
            score = int(np.mean(visibilities) * 100)

        # ---------------- Return ONLY annotated image + score ----------------
        return annotated, score


def patient_journey():
    header_logo()
    back_arrow()
    st.markdown("## Start My Exercise")

    predictor = get_recoverai_predictor()

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    # ---------------- Streak ----------------
    streak_days = st.session_state.get("patient_streak", int(np.random.randint(3, 15)))
    st.metric("🔥 Streak", f"{streak_days} days")

    # ---------------- Today’s Workout ----------------
    todays = {
        "exercise": "Arm Abduction",   # must match model-supported names
        "sets": 2,
        "reps": 10,
        "notes": "Keep shoulders relaxed; slow controlled range.",
    }

    st.markdown("### Today’s Workout")
    st.write(
        f"**{todays['exercise']}** — {todays['sets']} sets × {todays['reps']} reps  \n"
        f"_Notes:_ {todays['notes']}"
    )

    if st.button("🎬 Watch Tutorial", use_container_width=False):
        st.markdown("#### Tutorial Video")
        if os.path.exists("tutorial.mp4"):
            st.video("tutorial.mp4")
        else:
            st.info("Please add **tutorial.mp4** to the project folder.")

    st.markdown("---")

    # ---------------- My Schedule ----------------
    st.markdown("### My Schedule")
    today = pd.Timestamp.today().date()
    schedule = pd.DataFrame([
        {"date": today, "exercise": "Arm Abduction", "type": "Arms", "done": False},
        {"date": today, "exercise": "Leg Lunge", "type": "Legs", "done": True},
        {"date": today + pd.Timedelta(days=1), "exercise": "Squats", "type": "Legs", "done": False},
    ])

    st.dataframe(schedule, use_container_width=True, height=240)

    st.markdown("### Perform Exercise")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        div[data-testid="stLinkButton"] a {
            background-color: #6b9ebd !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 0.75rem 1rem !important;
            font-weight: 600 !important;
            border: none !important;
            box-shadow: 0 4px 12px rgba(107,158,189,0.25) !important;
            text-decoration: none !important;
        }
        div[data-testid="stLinkButton"] a:hover {
            background-color: #5b8cac !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.link_button(
        "Open Smart Exercise Interface",
        "http://127.0.0.1:8000/",
        use_container_width=True,
    )

    st.markdown("---")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


def patient_report():
    header_logo()
    back_arrow()
    st.markdown("## My Exercise Progress")

    # ---- Filter (Dropdown • Required • Output) ----
    filter_option = st.selectbox(
        "View Progress For:",
        ["Last 7 Days", "Last 14 Days", "Last 30 Days", "All"],
    )

    # Determine the number of days based on selection
    if filter_option == "Last 7 Days":
        period = 7
    elif filter_option == "Last 14 Days":
        period = 14
    elif filter_option == "Last 30 Days":
        period = 30
    else:
        period = 60   # default full range

    # Create filtered dummy progress data
    days = pd.date_range(pd.Timestamp.today() - pd.Timedelta(days=period), periods=period + 1)
    acc = np.random.randint(70, 99, len(days))
    df = pd.DataFrame({"day": days, "accuracy": acc})

    # ---- Chart Output ----
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x="day:T", y="accuracy:Q")
        .properties(title=f"My Exercise Accuracy Trend ({filter_option})")
    )

    st.altair_chart(chart, use_container_width=True)

    # ---- Save CSV ----
    st.download_button(
        "📥 Download My Progress",
        data=df.to_csv(index=False),
        file_name="my_exercise_progress.csv",
        mime="text/csv",
    )


CLINICIAN_ICON = "clincian_icon.jpg"
def patient_contact_clinician():
    header_logo()
    back_arrow()
    st.markdown("## Contact My Clinician")

    # --- Find assigned clinician name (simple demo) ---
    p_df = st.session_state.db["patients"].copy()
    c_df = st.session_state.db["clinicians"].copy()

    assigned_display = "Your Clinician"
    if not p_df.empty and not c_df.empty:
        # demo: take first patient row
        assigned_username = p_df.iloc[0].get("assigned_clinician", "")
        row = c_df[c_df["username"] == assigned_username]
        if not row.empty:
            assigned_display = f"{row.iloc[0]['first']} {row.iloc[0]['last']}"

    st.info(f"You are chatting with **{assigned_display}**.")

    # --- Simple chat history in session_state ---
    if "clinician_chat" not in st.session_state:
        st.session_state.clinician_chat = [
            {
                "role": "assistant",
                "text": f"Hello, I'm {assigned_display}. How are you feeling after your last exercise session?"
            }
        ]

    # Show previous messages
    for msg in st.session_state.clinician_chat:
        if msg["role"] == "assistant":
            # Clinician message with avatar
            with st.chat_message("assistant", avatar=CLINICIAN_ICON):
                st.write(msg["text"])
        else:
            # Patient message
            with st.chat_message("user"):
                st.write(msg["text"])

    # Input for new message
    prompt = st.chat_input("Write a message to your clinician…")

    if prompt:
        # Add patient message
        st.session_state.clinician_chat.append({"role": "user", "text": prompt})

        # Demo clinician reply
        reply = (
            "Thank you for your update. 🌿\n\n"
            "Please remember to keep your movements slow and controlled. "
            "If you feel any pain, stop the exercise and let me know."
        )
        st.session_state.clinician_chat.append({"role": "assistant", "text": reply})

        # Re-render with new messages
        st.rerun()

def patient_chat():
    header_logo()
    back_arrow()
    st.markdown("## 🤖 AI Assistant")

    st.chat_message("ai").write("Hello! I'm your RecoverAI assistant. How can I help you today?")

    # Chat input (default Streamlit send button)
    prompt = st.chat_input("Ask about today's exercises…")

    # If user typed something
    if prompt:
        st.chat_message("user").write(prompt)
        st.chat_message("ai").write("Thanks! I recommend 2 sets of 10 reps. Keep your posture aligned.")

    # -------------------------------
    # ✅ FLOATING MIC BUTTON (no upload)
    # -------------------------------
    st.markdown(
        """
        <style>
        /* Positioning mic near send arrow inside the input bar */
        .mic-btn {
            position: fixed;
            bottom: 90px;              /* distance from bottom of chat input */
            right: 70px;               /* position next to send arrow */
            background-color: #fa9b93; /* matches your theme */
            border-radius: 50%;
            width: 34px;
            height: 34px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
            z-index: 9999;
        }
        .mic-btn:hover {
            background-color: #ff9c98;
        }
        .mic-btn img {
            width: 18px;
            height: 18px;
        }
        </style>

        <!-- Mic Icon -->
        <div class="mic-btn" onclick="alert('🎤 Voice input coming soon')">
            <img src="https://img.icons8.com/ios-filled/50/ffffff/microphone.png"/>
        </div>
        """,
        unsafe_allow_html=True
    )

def patient_about_recoverai():
    header_logo()
    back_arrow()
    st.markdown("## About RecoverAI")

    # -------- Idea & Goal --------
    st.markdown("### Our Idea & Goal")
    st.markdown(
        """
        RecoverAI is a **telerehabilitation assistant** designed to support patients as they 
        perform their physiotherapy exercises at home.  
        
        Our main goals are:
        - Make rehabilitation **easier to follow** with clear exercise guidance.
        - Keep patients **connected to their clinicians** between sessions.
        - Provide **simple progress tracking** so patients can see their improvement.
        """
    )

    # -------- What Patients Can Do --------
    st.markdown("### What Can I Do as a Patient?")
    st.markdown(
        """
        As a patient, you can use RecoverAI to:
        - 🏠 **Access Homepage** – See quick information such as your streak and days left.
        - 👤 **View Profile** – Review and update your personal details.
        - 🏋️ **Start My Exercise** – Follow today's exercise plan, open the camera, and let the system estimate your pose and form.
        - 📊 **Generate Reports** – View charts that summarize your exercise accuracy over time.
        - 💬 **Contact My Clinician** – Chat with your assigned clinician and send updates about pain, progress, or questions.
        - 🤖 **Consult AI** – Ask the RecoverAI assistant for basic guidance about your exercises and routine.
        """
    )

    st.markdown(
        """
        You can move between these pages **anytime** using the menu on the left sidebar.  
        Just click the item you want (for example: *Start My Exercise* or *Contact My Clinician*).
        """
    )

    # -------- Contact Us --------
    st.markdown("### 📞 Contact Us")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            **Support Email**  
            📧 support@recoverai.sa  
            """
        )
    with c2:
        st.markdown(
            """
            **WhatsApp (Support)**  
            📱 +966 5 0000 0001  
            """
        )

    st.info(
        "If you have any technical issues, cannot sign in, or feel pain during exercises, "
        "please contact your clinician or our support team."
    )

    # -------- Follow Us --------
    st.markdown("### 🌐 Follow Us")

    st.markdown(
    """
    Stay connected with us on social media for tips, updates, and new features:

    - **Instagram:** <a href="https://www.instagram.com/recoverai.sa" target="_blank">@recoverai.sa</a>  
    - **TikTok:** <a href="https://www.tiktok.com/@recoverai" target="_blank">@recoverai</a>  
    - **LinkedIn:** <a href="https://www.linkedin.com/company/recoverai-digital-rehabilitation" target="_blank">RecoverAI – Digital Rehabilitation</a>  
    - **X:** <a href="https://x.com/recoverai_sa" target="_blank">@recoverai_sa</a>
    """,
    unsafe_allow_html=True
)



def patient_app():
    choice = patient_sidebar()
    if choice == "Access Homepage":
        patient_welcome()
    elif choice == "View Profile":
        patient_profile()
    elif choice == "Start My Exercise":
        patient_journey()
    elif choice == "My Exercise Progress":
        patient_report()
    elif choice == "Contact My Clinician":
        patient_contact_clinician()
    elif choice == "Consult AI":
        patient_chat()
    elif choice == "About RecoverAI":
        patient_about_recoverai()
    elif choice == "🔒 Sign Out":
        sign_out()


# ---------- Router ----------
if st.session_state.page == "first":
    first_page()
elif st.session_state.page == "welcome":
    welcome_page()
elif st.session_state.page == "admin_signin":
    admin_signin()
elif st.session_state.page == "admin_app":
    admin_app()
elif st.session_state.page == "clinician_signin":
    clinician_signin()
elif st.session_state.page == "clinician_app":
    clinician_app()
elif st.session_state.page == "patient_signin":
    patient_signin()
elif st.session_state.page == "patient_app":
    patient_app()
elif st.session_state.page == "forgot_password":
    forgot_password_page()
elif st.session_state.page == "notifications":
    notifications_page()

