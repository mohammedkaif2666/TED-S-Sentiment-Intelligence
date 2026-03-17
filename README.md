# TED-S: Advanced Sentiment Analytics & Intelligence

TED-S is a powerful, real-time sentiment analysis dashboard designed to analyze Twitter Event Data (TED) using an ensemble of high-performance natural language processing (NLP) models. This project provides a sophisticated web interface for executing deep learning models, streaming real-time logs, and visualizing complex emotional trends through dynamic analytics.

## 🚀 Features

- **Multi-Model Pipeline:** Execute and compare multiple sentiment analysis engines:
  - **VADER:** Rule-based sentiment analysis for social media.
  - **TextBlob:** Simple and effective polarity/subjectivity scoring.
  - **CNN (Convolutional Neural Network):** Deep learning optimized for text classification.
  - **LSTM (Long Short-Term Memory):** RNN-based sequential analysis for context retention.
  - **BERT Transformer:** State-of-the-art transformer-based sentiment extraction.
- **Real-Time Log Streaming:** Live server-sent events (SSE) allow users to monitor model execution and training progress directly in the browser terminal.
- **Advanced Visualizations:** Dynamic generation of sentiment distribution pies, chronological timelines, intensity heatmaps, and ensemble trend lines.
- **Premium Responsive UI:** A sleek, glassmorphism-inspired interface that adapts perfectly to desktop, tablet, and mobile devices.
- **Interactive Analytics:** Click-to-zoom functionality for all analytical charts, providing high-resolution insights on any screen.
- **Automated Reporting:** Generates downloadable Excel reports with granular sentiment predictions for every model run.

## 🛠️ Technology Stack

- **Backend:** Python, Flask (Gunicorn for production)
- **NLP/ML:** TensorFlow, Transformers (BERT), NLTK, TextBlob, Scikit-learn
- **Data:** Pandas, Matplotlib, Seaborn, Openpyxl
- **Frontend:** HTML5, CSS3 (Vanilla), JavaScript (ES6+), SSE
- **Deployment:** Optimized for Render with `render.yaml` and `Procfile` support.

## 📦 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/TED-S.git
   cd TED-S
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**

   ```bash
   python app.py
   ```

   Open `http://localhost:5000` in your browser.

## 💻 Developer

Developed by **[Mohammed kaif & Surya]** as an advanced sentiment analytics solution.

---

© 2026 Developed with passion for Sentiment Intelligence.
