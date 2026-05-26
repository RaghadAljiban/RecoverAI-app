# RecoverAI – AI-Powered Tele-Rehabilitation System

RecoverAI is an AI-powered tele-rehabilitation platform designed to support patients during home-based recovery through intelligent rehabilitation monitoring, exercise assessment, and clinician-patient interaction.

---

## 📌 Features

### 👨‍⚕️ Admin
- Manage clinicians and patients
- Monitor system activity
- Send announcements
- View platform analytics

### 🩺 Clinician
- Monitor assigned patients
- Review uploaded exercise videos
- Generate rehabilitation reports
- Track patient adherence and progress
- Communicate with patients

### 🧍 Patient
- Access personalized rehabilitation plans
- Upload exercise performance videos
- View rehabilitation progress
- Receive AI-assisted exercise feedback
- Track completed sessions

---

## 🧠 AI Technologies Used

- MediaPipe Pose Estimation
- PyTorch Deep Learning Models
- Human Pose Analysis
- Exercise Recognition
- Video Processing with OpenCV

### Supported Exercises
- Arm Abduction
- Arm VW
- Push-ups
- Leg Abduction
- Leg Lunge
- Squats

---

## 🛠️ Technologies

- Python
- Streamlit
- FastAPI
- PyTorch
- MediaPipe
- OpenCV
- Pandas
- Altair

---

## 📂 Project Structure

```bash
RecoverAI-app/
│
├── recoverai_streamlit.py
├── db.py
├── requirements.txt
└── README.md
│
└── Recoverai-demo/
    │
    ├── requirements.txt
    │
    ├── backend/
    │   ├── api.py
    │   ├── deps.py
    │   ├── main.py
    │   └── __init__.py
    │
    ├── frontend/
    │   ├── app.js
    │   ├── index.html
    │   └── style.css
    │
    ├── model/
    │   ├── best_conditioned_tcn_clean.pt
    │   ├── best_conditioned_tcn_clean77.pt
    │   └── best_exercise_recognition_tcn.pt
    │
    └── telerehab/
        ├── checkpoint.py
        ├── classifier.py
        ├── config.py
        ├── features.py
        ├── model.py
        ├── overlay.py
        ├── pose.py
        └── __init__.py
```

---

## 🎨 UI Design

RecoverAI uses a modern healthcare-inspired interface focused on usability and accessibility.

### Interface Highlights
- Responsive healthcare-themed design
- Rehabilitation monitoring dashboard
- Exercise upload workflow
- AI-assisted assessment interface
- Progress analytics and visualizations

### Primary Colors
- Primary: `#6b9ebd`
- Accent: `#fa9b93`

---

## 🎥 RecoverAI Demo

The platform includes a rehabilitation demo interface for testing exercise uploads and AI-based exercise evaluation.

### Demo Capabilities
- Exercise upload interface
- Guided rehabilitation workflow
- Video-based exercise assessment
- AI prediction and scoring display
- Patient-friendly dashboard

---

## 🚀 Future Improvements

- Advanced rehabilitation performance analysis
- Mobile application integration
- Personalized rehabilitation recommendations
- Cloud database deployment
- Enhanced exercise quality assessment
- Expanded rehabilitation exercise library

---

## 👩‍💻 Developed By

RecoverAI Team  
Artificial Intelligence Department  
Imam Abdulrahman Bin Faisal University

---

## 📜 License

This project is developed for academic and research purposes.
