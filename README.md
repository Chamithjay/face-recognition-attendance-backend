# 🎭 Face Recognition Attendance Backend

A robust Python-based backend system for automated attendance management using facial recognition technology. This system leverages computer vision and machine learning to provide secure, contactless attendance tracking.

## 🌟 Overview

This backend API powers a face recognition attendance system that can identify and verify individuals for attendance tracking in educational institutions, corporate environments, or events. It provides RESTful endpoints for face enrollment, recognition, and attendance management.

## 🛠️ Tech Stack

- **Language:** Python (96.7%)
- **Containerization:** Docker (3.3%)
- **Face Recognition:** OpenCV, face_recognition, or dlib
- **Framework:**FastAPI
- **Database:** PostgreSQL
- **Image Processing:**  NumPy

## ✨ Key Features

- 👤 **Face Enrollment** - Register new users with facial data
- 🔍 **Face Recognition** - Real-time face detection and identification
- 📊 **Attendance Tracking** - Automatic attendance logging
- 🐳 **Docker Support** - Easy deployment with Docker containers
- 🌐 **RESTful API** - Easy integration with front-end applications
- ⚡ **Fast Processing** - Optimized face recognition algorithms

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)
- Webcam or camera access (for face capture)
- CMake (for dlib installation)

## 🚀 Quick Start



## 📊 Face Recognition Workflow

1. **Enrollment Phase**
   - User provides personal details
   - System captures face video
   - Face encodings are generated and stored

2. **Recognition Phase**
   - Camera captures live video
   - System detects faces in the frames
   - Compares with enrolled face encodings
   - Matches are verified and attendance is marked

3. **Verification**
   - Confidence score is calculated
   - If score exceeds threshold, attendance is logged
   - Timestamp and metadata are recorded


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Chamith Jay**

- GitHub: [@Chamithjay](https://github.com/Chamithjay)

## 🙏 Acknowledgments

⭐️ If you find this project useful, please give it a star!
