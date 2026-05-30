import os
import re
import tempfile
import hmac
from datetime import timedelta
from collections import Counter
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template, redirect, request, session, url_for
from openai import OpenAI
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Password protection for demo sharing.
# Set APP_PASSWORD and SECRET_KEY in Render Environment Variables.
app.secret_key = os.getenv("SECRET_KEY", os.urandom(32))
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=int(os.getenv("SESSION_HOURS", "8")))

# OpenAI's current audio upload limit is 25 MB. This app enforces the same limit.
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

ALLOWED_EXTENSIONS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg"}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "for", "from", "had", "has",
    "have", "he", "her", "hers", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "just", "like", "may", "me", "might", "more", "most",
    "my", "no", "not", "of", "on", "or", "our", "out", "over", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "to", "up", "us", "very", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "will",
    "with", "would", "you", "your", "yours", "about", "also", "because",
    "between", "during", "each", "few", "further", "here", "once", "only",
    "own", "same", "through", "under", "until", "while", "within", "without"
}

def allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def transcribe_audio(audio_path: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it in Render Environment Variables.")

    model = os.getenv("TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
    client = OpenAI(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text"
        )

    if isinstance(result, str):
        return result.strip()

    # Fallback for SDK versions that return an object.
    return getattr(result, "text", "").strip()

def tokenize(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']+", text.lower())
    return [w.strip("'") for w in words if len(w.strip("'")) > 2 and w.strip("'") not in STOPWORDS]

def top_phrases(tokens: List[str], limit: int = 8) -> List[str]:
    if not tokens:
        return []

    bigrams = [
        f"{tokens[i]} {tokens[i + 1]}"
        for i in range(len(tokens) - 1)
        if tokens[i] not in STOPWORDS and tokens[i + 1] not in STOPWORDS
    ]

    phrase_counts = Counter(bigrams)
    word_counts = Counter(tokens)

    phrases = [phrase for phrase, _ in phrase_counts.most_common(limit)]
    if len(phrases) < limit:
        for word, _ in word_counts.most_common(limit):
            if word not in phrases:
                phrases.append(word)
            if len(phrases) >= limit:
                break

    return phrases[:limit]

def extractive_summary(text: str, tokens: List[str], max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text.strip()[:500] + ("..." if len(text.strip()) > 500 else "")

    important_words = {word for word, _ in Counter(tokens).most_common(20)}
    scored: List[Tuple[int, int, str]] = []

    for idx, sentence in enumerate(sentences):
        sentence_tokens = tokenize(sentence)
        score = sum(1 for token in sentence_tokens if token in important_words)
        score += min(len(sentence_tokens), 30) // 10
        scored.append((score, -idx, sentence))

    selected = sorted(scored, reverse=True)[:max_sentences]
    selected = sorted(selected, key=lambda x: -x[1])
    return " ".join(item[2] for item in selected)

def analyse_text(text: str) -> Dict:
    clean_text = re.sub(r"\s+", " ", text).strip()
    tokens = tokenize(clean_text)
    analyser = SentimentIntensityAnalyzer()
    sentiment = analyser.polarity_scores(clean_text)

    compound = sentiment["compound"]
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    words = re.findall(r"\b\w+\b", clean_text)
    phrases = top_phrases(tokens)

    return {
        "sentiment_score": round(compound, 3),
        "sentiment_label": label,
        "sentiment_breakdown": {
            "positive": round(sentiment["pos"], 3),
            "neutral": round(sentiment["neu"], 3),
            "negative": round(sentiment["neg"], 3)
        },
        "dominant_theme": phrases[0].title() if phrases else "Not enough text to detect a theme",
        "themes": [p.title() for p in phrases],
        "summary": extractive_summary(clean_text, tokens),
        "word_count": len(words),
        "estimated_reading_time_minutes": max(1, round(len(words) / 180)),
    }

def password_gate_is_enabled() -> bool:
    """Return True unless explicitly disabled with LOGIN_REQUIRED=false."""
    return os.getenv("LOGIN_REQUIRED", "true").lower() not in {"0", "false", "no", "off"}

@app.before_request
def require_password_login():
    """Protect the web app and analysis endpoint with a server-side password gate."""
    allowed_endpoints = {"login", "logout", "health", "static"}
    if request.endpoint in allowed_endpoints or request.path.startswith("/static/"):
        return None

    if not password_gate_is_enabled():
        return None

    if session.get("authenticated") is True:
        return None

    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    configured_password = os.getenv("APP_PASSWORD")
    error = None

    if not configured_password:
        error = "APP_PASSWORD has not been configured in Render Environment Variables."
        return render_template("login.html", error=error), 500

    if request.method == "POST":
        submitted_password = request.form.get("password", "")
        if hmac.compare_digest(submitted_password, configured_password):
            session.clear()
            session.permanent = True
            session["authenticated"] = True
            return redirect(url_for("index"))
        error = "Incorrect password. Please try again."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/analyse", methods=["POST"])
def analyse():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received. Please record or upload an audio file."}), 400

    audio = request.files["audio"]
    if audio.filename == "":
        return jsonify({"error": "No selected audio file."}), 400

    filename = secure_filename(audio.filename)
    if not allowed_file(filename):
        return jsonify({
            "error": "Unsupported file type. Please use mp3, mp4, mpeg, mpga, m4a, wav, webm, ogg."
        }), 400

    suffix = "." + filename.rsplit(".", 1)[1].lower()

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            audio.save(temp_file.name)
            temp_path = temp_file.name

        transcript = transcribe_audio(temp_path)
        if not transcript:
            return jsonify({"error": "The transcription was empty. Please try a clearer recording."}), 422

        analysis = analyse_text(transcript)

        return jsonify({
            "transcript": transcript,
            "analysis": analysis
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.errorhandler(413)
def file_too_large(_):
    return jsonify({"error": "The audio file is too large. Please upload a file below 25 MB."}), 413

if __name__ == "__main__":
    app.run(debug=True)