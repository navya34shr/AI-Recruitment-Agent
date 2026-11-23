
import json
import re
import smtplib
import requests
import streamlit as st
import pypdf

from datetime import datetime, timedelta
from email.message import EmailMessage


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Recruitment Assistant",
    page_icon="🤖",
    layout="wide"
)

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"
TIMEZONE = "Asia/Kolkata"


# =========================================================
# ROLE REQUIREMENTS
# =========================================================

ROLE_REQUIREMENTS = {
    "AI/ML Engineer": """
    Python, Machine Learning, Deep Learning,
    TensorFlow, PyTorch, Data Analysis,
    Neural Networks, NLP, LLM, RAG,
    Prompt Engineering and Model Deployment.
    """,

    "Frontend Engineer": """
    HTML, CSS, JavaScript, React,
    Responsive Design, TypeScript,
    UI Development, API Integration,
    Git and Frontend Testing.
    """,

    "Backend Engineer": """
    Python, Java, Node.js, REST APIs,
    SQL, MongoDB, Database Design,
    Authentication, Docker, Cloud,
    System Design and Git.
    """
}


# =========================================================
# SESSION STATE
# =========================================================

DEFAULTS = {
    "ollama_model": DEFAULT_MODEL,
    "company_name": "AI Recruiting Team",
    "gmail_address": "",
    "gmail_password": "",
    "zoom_account_id": "",
    "zoom_client_id": "",
    "zoom_client_secret": "",
    "resume_text": "",
    "resume_name": "",
    "analysis": None,
    "meeting": None,
    "email_sent": False
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# OLLAMA MODEL CONNECTION
# =========================================================

def get_ollama_models():
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=10
        )
        response.raise_for_status()

        models = response.json().get("models", [])

        return [
            model["name"] for model in models
        ]

    except requests.RequestException:
        return []


def analyze_resume(resume_text, role):

    model = st.session_state.ollama_model

    prompt = f"""
You are an experienced technical recruiter.

Analyze the candidate resume for this position:

ROLE:
{role}

REQUIRED SKILLS:
{ROLE_REQUIREMENTS[role]}

RESUME:
{resume_text[:18000]}

Evaluate the candidate fairly.

Return ONLY a valid JSON object in this exact structure:

{{
    "candidate_name": "Candidate name",
    "candidate_email": "Email or empty string",
    "experience_years": 0,
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["Docker"],
    "score": 7,
    "selected": true,
    "feedback": "Explain the decision in 2 to 4 sentences."
}}

Rules:
1. Score must be between 0 and 10.
2. Select only if the score is at least 7.
3. Count projects as practical experience.
4. Do not invent qualifications.
5. Return JSON only.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional technical recruiter. "
                        "Return valid JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0
            }
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()
    raw = result.get("message", {}).get("content", "")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            raise ValueError(
                "Ollama did not return valid JSON. "
                "Try another installed model."
            )

        data = json.loads(match.group(0))

    required = [
        "candidate_name",
        "candidate_email",
        "experience_years",
        "matched_skills",
        "missing_skills",
        "score",
        "selected",
        "feedback"
    ]

    for key in required:
        if key not in data:
            raise ValueError(
                f"AI response is missing: {key}"
            )

    data["score"] = max(
        0,
        min(10, float(data["score"]))
    )

    data["selected"] = data["score"] >= 7

    return data


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_resume_text(uploaded_file):

    try:
        reader = pypdf.PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:
            pages.append(page.extract_text() or "")

        return "\n".join(pages).strip()

    except Exception as error:
        st.error(f"PDF extraction error: {error}")
        return ""


# =========================================================
# ZOOM AUTHENTICATION
# =========================================================

def get_zoom_access_token():

    account_id = st.session_state.zoom_account_id.strip()
    client_id = st.session_state.zoom_client_id.strip()
    client_secret = st.session_state.zoom_client_secret.strip()

    if not account_id or not client_id or not client_secret:
        raise ValueError(
            "Enter all three Zoom credentials."
        )

    url = "https://zoom.us/oauth/token"

    response = requests.post(
        url,
        params={
            "grant_type": "account_credentials",
            "account_id": account_id
        },
        auth=(client_id, client_secret),
        headers={
            "Content-Type": "application/x-www-form-urlencoded"
        },
        timeout=30
    )

    if not response.ok:
        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            f"Zoom authentication failed "
            f"({response.status_code}): {error_details}"
        )

    token = response.json().get("access_token")

    if not token:
        raise RuntimeError(
            "Zoom did not return an access token."
        )

    return token


# =========================================================
# CREATE ZOOM MEETING
# =========================================================

def create_zoom_meeting(
    candidate_name,
    role,
    interview_datetime,
    duration
):

    token = get_zoom_access_token()

    url = "https://api.zoom.us/v2/users/me/meetings"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    payload = {
        "topic": f"{role} Interview - {candidate_name}",
        "type": 2,
        "start_time": interview_datetime.isoformat(
            timespec="seconds"
        ),
        "duration": duration,
        "timezone": TIMEZONE,
        "settings": {
            "host_video": True,
            "participant_video": True,
            "waiting_room": True,
            "join_before_host": False,
            "mute_upon_entry": True
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=30
    )

    if not response.ok:
        try:
            error_details = response.json()
        except ValueError:
            error_details = response.text

        raise RuntimeError(
            f"Zoom meeting creation failed "
            f"({response.status_code}): {error_details}"
        )

    meeting = response.json()

    return {
        "meeting_id": meeting.get("id"),
        "join_url": meeting.get("join_url"),
        "password": meeting.get("password", ""),
        "start_url": meeting.get("start_url"),
        "duration": meeting.get("duration")
    }


# =========================================================
# GMAIL SMTP
# =========================================================

def send_email(receiver, subject, body):

    sender = st.session_state.gmail_address.strip()
    password = st.session_state.gmail_password.strip()

    if not sender or not password:
        raise ValueError(
            "Enter your Gmail address and App Password."
        )

    message = EmailMessage()

    message["Subject"] = subject
    message["From"] = sender
    message["To"] = receiver

    message.set_content(body)

    with smtplib.SMTP(
        "smtp.gmail.com",
        587,
        timeout=30
    ) as server:

        server.starttls()

        server.login(sender, password)

        server.send_message(message)

    return True


# =========================================================
# EMAIL TEMPLATES
# =========================================================

def send_selection_email(
    candidate_name,
    candidate_email,
    role,
    meeting,
    interview_datetime
):

    company = st.session_state.company_name

    subject = f"Interview Invitation - {role}"

    body = f"""
Dear {candidate_name or 'Candidate'},

Congratulations!

Your resume has been shortlisted for the {role} position at {company}.

Your interview has been scheduled.

INTERVIEW DETAILS

Date: {interview_datetime.strftime('%d %B %Y')}

Time: {interview_datetime.strftime('%I:%M %p')} IST

Duration: {meeting['duration']} minutes

Zoom Meeting Link:
{meeting['join_url']}

Meeting ID: {meeting['meeting_id']}

Passcode: {meeting['password'] or 'Not required'}

Please join 5 minutes early.

Best regards,
Recruitment Team
{company}
"""

    return send_email(
        candidate_email,
        subject,
        body
    )


def send_rejection_email(
    candidate_name,
    candidate_email,
    role,
    feedback
):

    company = st.session_state.company_name

    subject = f"Application Update - {role}"

    body = f"""
Dear {candidate_name or 'Candidate'},

Thank you for applying for the {role} position at {company}.

After reviewing your resume, we have decided not to
move forward with your application at this time.

Feedback:
{feedback}

We encourage you to continue improving your technical
skills and apply for future opportunities.

Best wishes,
Recruitment Team
{company}
"""

    return send_email(
        candidate_email,
        subject,
        body
    )


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

def render_sidebar():

    st.sidebar.title("⚙️ Settings")

    st.session_state.company_name = st.sidebar.text_input(
        "Company name",
        value=st.session_state.company_name
    )

    st.sidebar.divider()

    st.sidebar.subheader("Ollama AI Settings")

    models = get_ollama_models()

    if models:

        current = st.session_state.ollama_model

        default_index = (
            models.index(current)
            if current in models
            else 0
        )

        st.session_state.ollama_model = (
            st.sidebar.selectbox(
                "Installed AI model",
                models,
                index=default_index
            )
        )

        st.sidebar.success(
            f"{len(models)} Ollama model(s) available"
        )

    else:
        st.sidebar.error(
            "Ollama is not running. Start Ollama first."
        )

    st.sidebar.divider()

    st.sidebar.subheader("Gmail Settings")

    st.session_state.gmail_address = st.sidebar.text_input(
        "Gmail address",
        value=st.session_state.gmail_address
    )

    st.session_state.gmail_password = st.sidebar.text_input(
        "Gmail App Password",
        value=st.session_state.gmail_password,
        type="password"
    )

    st.sidebar.caption(
        "Use a Google App Password, not your normal Gmail password."
    )

    st.sidebar.divider()

    st.sidebar.subheader("Zoom Settings")

    st.session_state.zoom_account_id = st.sidebar.text_input(
        "Zoom Account ID",
        value=st.session_state.zoom_account_id,
        type="password"
    )

    st.session_state.zoom_client_id = st.sidebar.text_input(
        "Zoom Client ID",
        value=st.session_state.zoom_client_id,
        type="password"
    )

    st.session_state.zoom_client_secret = st.sidebar.text_input(
        "Zoom Client Secret",
        value=st.session_state.zoom_client_secret,
        type="password"
    )

    st.sidebar.caption(
        "Use credentials from your Zoom Server-to-Server OAuth app."
    )

    st.sidebar.divider()

    if st.sidebar.button("Reset Application", use_container_width=True):

        for key in [
            "resume_text",
            "resume_name",
            "analysis",
            "meeting",
            "email_sent"
        ]:
            st.session_state[key] = DEFAULTS[key]

        st.rerun()


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    render_sidebar()

    st.title("🤖 AI Recruitment Assistant")

    st.caption(
        "AI-powered resume screening, Zoom interview scheduling "
        "and automatic Gmail communication."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        role = st.selectbox(
            "Select recruitment role",
            list(ROLE_REQUIREMENTS.keys())
        )

        with st.expander("View role requirements"):
            st.write(ROLE_REQUIREMENTS[role])

    with col2:

        uploaded_file = st.file_uploader(
            "Upload candidate resume (PDF)",
            type=["pdf"]
        )

    if uploaded_file is None:

        st.info("Upload a PDF resume to begin.")

        return

    if uploaded_file.name != st.session_state.resume_name:

        st.session_state.resume_name = uploaded_file.name

        st.session_state.resume_text = extract_resume_text(
            uploaded_file
        )

        st.session_state.analysis = None
        st.session_state.meeting = None
        st.session_state.email_sent = False

    if not st.session_state.resume_text:

        st.error(
            "Could not extract resume text. "
            "Please upload a text-based PDF."
        )

        return

    with st.expander("📄 Resume preview"):

        uploaded_file.seek(0)

        st.download_button(
            "Download resume",
            data=uploaded_file,
            file_name=uploaded_file.name,
            mime="application/pdf"
        )

        st.text_area(
            "Extracted resume text",
            st.session_state.resume_text,
            height=200
        )

    st.divider()

    if st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            f"Analyzing with Ollama ({st.session_state.ollama_model})..."
        ):

            try:

                result = analyze_resume(
                    st.session_state.resume_text,
                    role
                )

                st.session_state.analysis = result
                st.session_state.meeting = None
                st.session_state.email_sent = False

            except Exception as error:

                st.error(
                    f"Resume analysis failed: {error}"
                )

                return

    analysis = st.session_state.analysis

    if not analysis:
        return

    st.divider()

    st.header("📊 Resume Analysis Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Resume Score",
        f"{analysis['score']:.1f}/10"
    )

    c2.metric(
        "Experience",
        f"{analysis['experience_years']} years"
    )

    c3.metric(
        "Decision",
        "Selected" if analysis["selected"] else "Not Selected"
    )

    st.subheader("Candidate Information")

    st.write(
        "**Candidate:**",
        analysis["candidate_name"]
    )

    st.write(
        "**Detected Email:**",
        analysis["candidate_email"] or "Not found"
    )

    left, right = st.columns(2)

    with left:

        st.subheader("✅ Matched Skills")

        for skill in analysis["matched_skills"]:
            st.write(f"- {skill}")

    with right:

        st.subheader("❌ Missing Skills")

        for skill in analysis["missing_skills"]:
            st.write(f"- {skill}")

    st.subheader("Recruiter Feedback")

    st.write(analysis["feedback"])

    st.divider()

    candidate_email = st.text_input(
        "Candidate email address",
        value=analysis["candidate_email"] or ""
    )

    if not analysis["selected"]:

        st.warning(
            "The candidate did not meet the selection threshold."
        )

        if st.button("📧 Send Rejection Email"):

            if not candidate_email or "@" not in candidate_email:
                st.error("Enter a valid candidate email.")
                return

            try:

                with st.spinner("Sending rejection email..."):

                    send_rejection_email(
                        analysis["candidate_name"],
                        candidate_email,
                        role,
                        analysis["feedback"]
                    )

                st.success("Rejection email sent successfully!")

            except Exception as error:

                st.error(f"Email error: {error}")

        return

    st.success(
        "🎉 Congratulations! The candidate has been shortlisted."
    )

    st.header("📅 Schedule Zoom Interview")

    tomorrow = datetime.now() + timedelta(days=1)

    col_date, col_time, col_duration = st.columns(3)

    with col_date:

        interview_date = st.date_input(
            "Interview date",
            value=tomorrow.date(),
            min_value=tomorrow.date()
        )

    with col_time:

        interview_time = st.time_input(
            "Interview time (IST)",
            value=datetime.strptime("11:00", "%H:%M").time()
        )

    with col_duration:

        duration = st.selectbox(
            "Duration",
            [30, 45, 60, 90],
            index=2
        )

    if st.button(
        "🚀 Create Zoom Meeting and Send Email",
        type="primary",
        use_container_width=True
    ):

        if not candidate_email or "@" not in candidate_email:

            st.error("Enter a valid candidate email.")

            return

        interview_datetime = datetime.combine(
            interview_date,
            interview_time
        )

        if interview_datetime <= datetime.now():

            st.error("Select a future interview date and time.")

            return

        try:

            with st.spinner("Creating Zoom meeting..."):

                meeting = create_zoom_meeting(
                    analysis["candidate_name"],
                    role,
                    interview_datetime,
                    duration
                )

            st.session_state.meeting = meeting

            with st.spinner("Sending interview invitation..."):

                send_selection_email(
                    analysis["candidate_name"],
                    candidate_email,
                    role,
                    meeting,
                    interview_datetime
                )

            st.session_state.email_sent = True

            st.success(
                "Zoom meeting created and invitation email sent!"
            )

        except Exception as error:

            st.error(
                f"Scheduling failed: {error}"
            )

    if st.session_state.meeting:

        meeting = st.session_state.meeting

        st.divider()

        st.subheader("🎥 Zoom Meeting Details")

        st.write("**Meeting ID:**", meeting["meeting_id"])

        st.write(
            "**Passcode:**",
            meeting["password"] or "Not required"
        )

        st.write("**Meeting Link:**", meeting["join_url"])

        st.link_button(
            "Join Zoom Meeting",
            meeting["join_url"]
        )

        if st.session_state.email_sent:

            st.success(
                f"Interview invitation sent to {candidate_email}"
            )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
