from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from flask_cors import CORS
from openai import OpenAI
import time
import json
import os
import re
import base64
import tempfile
import threading
import cv2
import numpy as np
import pytesseract
import httpx
from verification.proctoring_service import (
    analyze_frame_objects_and_pose,
    get_proctoring_session,
    clear_proctoring_session,
    MAX_WARNINGS
)

app = Flask(__name__)
CORS(app)

# NVIDIA API configuration with custom httpx client to eliminate SSL verification retries on Windows
http_client = httpx.Client(verify=False, timeout=httpx.Timeout(90.0, connect=15.0))
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY", ""),
    http_client=http_client
)
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")

# System prompt for the student learning assistant
SYSTEM_PROMPT = """You are an expert AI tutor for students. Your role is to:
1. Explain concepts clearly and simply, adapting to the student's level
2. Break down complex topics into digestible chunks
3. Provide examples and analogies to aid understanding
4. Ask clarifying questions when needed
5. Encourage critical thinking and problem-solving
6. Provide practice questions and quizzes when appropriate
7. Be patient, encouraging, and supportive

STRUCTURING YOUR RESPONSES (critical):
- Always use markdown headings (## for major sections, ### for subsections) to organize content
- Use bullet points (-) or numbered lists for key concepts and step-by-step explanations
- Use **bold** to highlight important terms, definitions, and key takeaways
- Always include a real-world example or analogy when explaining a concept
- Start with a clear, concise answer, then expand with deeper detail
- End with a brief summary or key takeaway
- Keep paragraphs short (2-3 lines max) for readability
- Use at most 1-2 emojis per response, only for friendliness — never as decoration

Always maintain a friendly, educational tone."""

# --- Secure Exam Configuration ---
REQUIRE_FULLSCREEN = True
REQUIRE_CAMERA = True
REQUIRE_MICROPHONE = True
AUTO_SUBMIT_ON_TAB_SWITCH = True
AUTO_SUBMIT_ON_FULLSCREEN_EXIT = True

submitted_quizzes = set()

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'REQUIRE_FULLSCREEN': REQUIRE_FULLSCREEN,
        'REQUIRE_CAMERA': REQUIRE_CAMERA,
        'REQUIRE_MICROPHONE': REQUIRE_MICROPHONE,
        'AUTO_SUBMIT_ON_TAB_SWITCH': AUTO_SUBMIT_ON_TAB_SWITCH,
        'AUTO_SUBMIT_ON_FULLSCREEN_EXIT': AUTO_SUBMIT_ON_FULLSCREEN_EXIT
    })

@app.route('/api/verification/face', methods=['POST'])
def verify_face():
    """Verify face from camera snapshot for exam entry"""
    try:
        data = request.json
        image_base64 = data.get('image', '')
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        image_bytes = base64.b64decode(image_base64.split(',')[-1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        from verification.opencv_service import _detect_face_box
        box, engine = _detect_face_box(img)
        
        if box is not None:
            return jsonify({'success': True, 'message': 'Face verified'})
        else:
            return jsonify({'success': False, 'error': 'No face detected. Please ensure your face is clearly visible.'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/verification/head_pose', methods=['POST'])
def verify_head_pose():
    """Verify head pose (up/down) from camera snapshot"""
    try:
        data = request.json
        image_base64 = data.get('image', '')
        if not image_base64:
            return jsonify({'error': 'No image provided'}), 400
        
        import base64
        image_bytes = base64.b64decode(image_base64.split(',')[-1])
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        from verification.opencv_service import _get_face_detector
        height, width = img.shape[:2]
        detector = _get_face_detector(width, height)
        if detector is not None:
            _, faces = detector.detect(img)
            if faces is not None and len(faces) > 0:
                face = faces[0]
                re_x, re_y = face[4], face[5]
                le_x, le_y = face[6], face[7]
                nose_x, nose_y = face[8], face[9]
                rm_x, rm_y = face[10], face[11]
                lm_x, lm_y = face[12], face[13]
                
                eye_y = (re_y + le_y) / 2.0
                mouth_y = (rm_y + lm_y) / 2.0
                
                face_h = mouth_y - eye_y
                pose = 'center'
                
                if face_h > 0:
                    pitch_ratio = (nose_y - eye_y) / face_h
                    if pitch_ratio < 0.25:
                        pose = 'up'
                    elif pitch_ratio > 0.75:
                        pose = 'down'
                
                if pose == 'center':
                    eye_w = le_x - re_x
                    if eye_w > 0:
                        yaw_ratio = (nose_x - re_x) / eye_w
                        if yaw_ratio < 0.25:
                            pose = 'right' # Looking to their right (screen left)
                        elif yaw_ratio > 0.75:
                            pose = 'left'  # Looking to their left (screen right)
                
                return jsonify({'success': True, 'pose': pose})
                
        return jsonify({'success': False, 'pose': 'unknown'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/verification/anomaly', methods=['POST'])
def record_anomaly():
    data = request.json
    app.logger.warning(f"ANOMALY DETECTED: {data}")
    return jsonify({'success': True})

@app.route('/api/proctoring/analyze', methods=['POST'])
def proctoring_analyze():
    """Real-time AI proctoring detection endpoint.
    Runs pretrained YOLO (phone + person count) & MediaPipe (head pose).
    Returns warning status, debounced anomalies, and auto-submit flag.
    """
    try:
        data = request.json or {}
        quiz_id = data.get('quiz_id', 'default_quiz')
        image_data = data.get('image', '')

        if not image_data:
            return jsonify({'error': 'No image provided'}), 400

        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]

        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode image'}), 400

        detection = analyze_frame_objects_and_pose(frame)
        session = get_proctoring_session(quiz_id)
        result = session.process_detection(detection)

        return jsonify(result)
    except Exception as e:
        app.logger.exception('Proctoring analysis error')
        return jsonify({'error': str(e)}), 500

@app.route('/api/proctoring/anomalies', methods=['GET'])
def get_proctoring_anomalies():
    quiz_id = request.args.get('quiz_id', 'default_quiz')
    session = get_proctoring_session(quiz_id)
    return jsonify({
        'quiz_id': quiz_id,
        'warning_count': session.warning_count,
        'max_warnings': session.max_warnings,
        'anomaly_history': session.anomaly_history
    })

@app.route('/api/quiz/save', methods=['POST'])
def quiz_save():
    data = request.json
    app.logger.info(f"Quiz progress saved: {data.get('quiz_id')}")
    return jsonify({'success': True})

@app.route('/api/quiz/submit', methods=['POST'])
def quiz_submit():
    data = request.json or {}
    quiz_id = data.get('quiz_id')
    student_id = data.get('student_id', 'student_1')
    
    if not quiz_id:
        return jsonify({'error': 'Missing quiz ID'}), 400
        
    submission_key = f"{student_id}_{quiz_id}"
    if submission_key in submitted_quizzes:
        return jsonify({'error': 'Duplicate submission rejected.'}), 400
        
    submitted_quizzes.add(submission_key)
    
    # Attach proctoring anomaly history to submission log
    session = get_proctoring_session(quiz_id)
    anomaly_history = list(session.anomaly_history) if session else []
    data['proctoring_anomalies'] = anomaly_history
    data['proctoring_warnings'] = session.warning_count if session else 0
    
    app.logger.info(f"QUIZ SUBMITTED: quiz_id={quiz_id} warnings={data.get('proctoring_warnings')} anomalies={len(anomaly_history)}")
    return jsonify({'success': True, 'message': 'Quiz submitted successfully.', 'anomalies_recorded': len(anomaly_history)})


# Server-side cache for generated misconception explanations
_misconception_server_cache = {}


@app.route('/api/misconception/explain', methods=['POST'])
def explain_misconception():
    """Generate a focused explanation for a student's specific misconception.
    Uses existing AI client and model.
    Contains the 6 required sections:
      1. What You Misunderstood
      2. Correct Concept
      3. Why This Understanding Is Incorrect
      4. Simple Explanation
      5. Example
      6. Key Takeaway
    """
    try:
        data = request.json or {}
        topic = data.get('topic', '')
        question = data.get('question', '')
        expected_representation = data.get('expected_representation', '')
        student_answer = data.get('student_answer', '')
        misconception = data.get('misconception', {})

        statement = misconception.get('statement', '')
        expected_understanding = misconception.get('expected_understanding', '')
        context_explanation = misconception.get('explanation', '')

        if not statement and not expected_understanding and not context_explanation:
            return jsonify({'error': 'Misconception details required'}), 400

        cache_key = f"{topic.strip().lower()}::{statement.strip().lower()}::{expected_understanding.strip().lower()}"
        if cache_key in _misconception_server_cache:
            return jsonify({
                'success': True,
                'topic': topic,
                'statement': statement,
                'expected_understanding': expected_understanding,
                'explanation': _misconception_server_cache[cache_key]
            })

        system_prompt = (
            "You are an expert, supportive AI educator. Your role is to help a student "
            "overcome a specific misconception identified in their quiz response.\n"
            "RULES:\n"
            "- Do NOT generate the complete original lesson again.\n"
            "- Do NOT explain unrelated concepts.\n"
            "- Focus strictly and specifically on the detected misconception.\n"
            "- You MUST structure your entire response using the following 6 exact markdown headings (##):\n\n"
            "## What You Misunderstood\n"
            "Explain clearly and directly what was misunderstood based on the student's actual statement.\n\n"
            "## Correct Concept\n"
            "Explain the accurate concept related to this misconception.\n\n"
            "## Why This Understanding Is Incorrect\n"
            "Clearly explain the fundamental difference between the student's misconception and the correct concept.\n\n"
            "## Simple Explanation\n"
            "Explain the concept in beginner-friendly, plain language that anyone can easily grasp.\n\n"
            "## Example\n"
            "Provide a simple, concrete real-world example or analogy illustrating the correct concept.\n\n"
            "## Key Takeaway\n"
            "Give a concise, memorable takeaway point to help the student remember the correct concept permanently."
        )

        user_prompt = (
            f"Topic: {topic or 'General'}\n"
            f"Quiz Question: {question or 'N/A'}\n"
            f"Reference / Ideal Answer: {expected_representation or 'N/A'}\n"
            f"Student's Actual Answer: {student_answer or 'N/A'}\n\n"
            "Detected Misconception:\n"
            f"- What the student stated: \"{statement}\"\n"
            f"- Expected understanding: {expected_understanding}\n"
            f"- Analysis notes: {context_explanation}\n\n"
            "Please generate the focused 6-part misconception explanation now."
        )

        want_stream = data.get('stream', False) or request.headers.get('Accept') == 'text/event-stream'
        if want_stream:
            def generate_stream():
                t_llm_start = time.perf_counter()
                first_token_logged = False
                print("[LLM] Request started")
                accumulated = []
                try:
                    completion = client.chat.completions.create(
                        model=DEFAULT_LLM_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3,
                        top_p=0.8,
                        max_tokens=500,
                        stream=True
                    )
                    for chunk in completion:
                        if not chunk.choices:
                            continue
                        content = chunk.choices[0].delta.content
                        if content:
                            if not first_token_logged:
                                ttft = round((time.perf_counter() - t_llm_start) * 1000)
                                print(f"[LLM] First token: {ttft} ms")
                                print("[LLM] Streaming...")
                                first_token_logged = True
                            accumulated.append(content)
                            yield f"data: {json.dumps({'content': content})}\n\n"

                    full_text = "".join(accumulated).strip()
                    _misconception_server_cache[cache_key] = full_text
                    dur = time.perf_counter() - t_llm_start
                    print(f"[LLM] Completed: {dur:.2f} s")
                    yield f"data: {json.dumps({'done': True, 'explanation': full_text})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"

            return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

        t_llm_start = time.perf_counter()
        print("[LLM] Request started")
        completion = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            top_p=0.8,
            max_tokens=500,
            stream=False
        )

        explanation = completion.choices[0].message.content.strip()
        dur = time.perf_counter() - t_llm_start
        print(f"[LLM] Completed: {dur:.2f} s")
        _misconception_server_cache[cache_key] = explanation

        return jsonify({
            'success': True,
            'topic': topic,
            'statement': statement,
            'expected_understanding': expected_understanding,
            'explanation': explanation
        })

    except Exception as e:
        app.logger.warning(f"Misconception LLM call failed ({e}), generating resilient fallback explanation")
        fallback_explanation = (
            f"## What You Misunderstood\n"
            f"You stated: \"{statement}\". This indicates an inaccurate understanding of how {topic or 'this concept'} functions.\n\n"
            f"## Correct Concept\n"
            f"{expected_understanding or 'A complete understanding requires focusing on the core principles and relationships of the concept.'}\n\n"
            f"## Why This Understanding Is Incorrect\n"
            f"{context_explanation or f'The statement diverges from the accurate mechanism because {expected_understanding or topic}.'}\n\n"
            f"## Simple Explanation\n"
            f"In simple terms, {topic or 'the subject'} should be understood through {expected_understanding or 'its primary definition and mechanisms'}.\n\n"
            f"## Example\n"
            f"Consider a real-world scenario where {topic or 'the system'} operates according to {expected_understanding or 'its established rules'}, contrasting with the mistaken assumption.\n\n"
            f"## Key Takeaway\n"
            f"Remember: {expected_understanding or f'Always associate {topic} with its verified foundational principles.'}"
        )
        return jsonify({
            'success': True,
            'topic': topic,
            'statement': statement,
            'expected_understanding': expected_understanding,
            'explanation': fallback_explanation
        })



def _extract_relevant_context(user_message, study_context, selected_concept="", max_len=1200):
    if not study_context:
        return ""
    
    paragraphs = [p.strip() for p in study_context.split('\n\n') if p.strip()]
    if not paragraphs:
        return study_context[:max_len]
    
    msg_words = set(re.findall(r'\w+', (user_message + " " + selected_concept).lower()))
    stopwords = {'what', 'is', 'a', 'the', 'how', 'why', 'does', 'do', 'can', 'you', 'explain', 'tell', 'me', 'about', 'in', 'of', 'and', 'or', 'to', 'for', 'with', 'on', 'this', 'that', 'it'}
    keywords = msg_words - stopwords

    if not keywords:
        return "\n\n".join(paragraphs[:2])[:max_len]

    scored = []
    for idx, p in enumerate(paragraphs):
        p_lower = p.lower()
        score = sum(2 if kw in p_lower else 0 for kw in keywords)
        if score > 0:
            scored.append((score, idx, p))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = []
        curr_len = 0
        for score, idx, p in scored:
            if curr_len + len(p) <= max_len:
                selected.append(p)
                curr_len += len(p)
            elif not selected:
                selected.append(p[:max_len])
                break
        return "\n\n".join(selected)

    return "\n\n".join(paragraphs[:2])[:max_len]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    data = request.json or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])
    
    # Build messages with system prompt and conversation history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Add conversation history (last 10 messages to keep context manageable)
    for msg in history[-10:]:
        messages.append(msg)
    
    # Add current user message
    current_topic = data.get('topic', '').strip()
    study_context = data.get('study_context', '').strip()
    selected_concept = data.get('selected_concept', '').strip()

    if not current_topic and not study_context:
        system_content = """You are an AI Study Tutor for students.
The student has not searched for a study topic on the main page yet.
Answer their doubt politely and briefly, and remind them to search a topic on the main page to get full study materials and interactive quizzes."""
    else:
        relevant_material = _extract_relevant_context(user_message, study_context, selected_concept=selected_concept)
        system_content = f"""You are an AI Study Tutor assisting a student.
The student is currently studying:

CURRENT TOPIC: {current_topic}
{"SELECTED CONCEPT: " + selected_concept if selected_concept else ""}

RELEVANT STUDY MATERIAL:
{relevant_material if relevant_material else "(Refer to the current topic)"}

INSTRUCTIONS:
1. Answer the student's exact doubt concisely and clearly.
2. Use the current Study Material as your primary context.
3. Maintain continuity with previous chatbot messages.
4. Explain difficult concepts using simple language with small examples when helpful.
5. Keep responses concise as this is a small side-chat window.
6. If the question is completely unrelated to {current_topic}, politely remind the student you are currently studying {current_topic} and ask if they would like to ask about this topic or start a new search."""

    messages = [{"role": "system", "content": system_content}]

    for msg in history[-6:]:
        if isinstance(msg, dict) and msg.get('content'):
            messages.append({"role": msg.get('role', 'user'), "content": msg.get('content')})

    messages.append({"role": "user", "content": user_message})
    

    def generate():
        try:
            completion = client.chat.completions.create(
                model=DEFAULT_LLM_MODEL,
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                max_tokens=600,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}, "reasoning_budget": 0},
                stream=True
            )

            for chunk in completion:
                if not chunk.choices:
                    continue
                # Skip internal reasoning content — it adds latency and is
                # not useful to display to students.
                reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                if reasoning:
                    continue
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/api/quiz', methods=['POST'])
def generate_quiz():
    data = request.json
    topic = data.get('topic', '')
    num_questions = data.get('num_questions', 5)
    
    prompt = f"""Create a quiz with {num_questions} open-ended questions about {topic}.
    These are verbal response questions - NO multiple choice options allowed.
    
    IMPORTANT FORMATTING RULES:
    - Each question must start with "Q1:", "Q2:", etc.
    - Questions should require critical thinking and verbal explanation
    - No options (A, B, C, D) should be provided
    - No answers should be shown - the student will answer verbally
    - Make questions educational and appropriate for a student learning level
    - Write each question on its own line
    
    Example format:
    Q1: Explain the process of photosynthesis in your own words.
    Q2: What are the main causes of World War II?
    
    Focus on questions that encourage students to explain concepts in their own words,
    demonstrate understanding of relationships between ideas, and apply knowledge to
    new situations."""
    
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a quiz generator for students. Create clear, educational open-ended questions that require verbal answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=600,
            stream=False
        )
        
        quiz_text = completion.choices[0].message.content
        try:
            from verification.study_knowledge import get_study_knowledge, store_study_knowledge
            knowledge = get_study_knowledge(topic)
            if knowledge:
                questions = []
                for line in (quiz_text or "").split('\n'):
                    line = line.strip()
                    m = re.match(r"^Q\d+[:.]?\s*(.+)$", line, re.IGNORECASE)
                    if m:
                        questions.append(m.group(1).strip())
                knowledge["questions"] = questions
                store_study_knowledge(knowledge)
        except Exception:
            pass

        try:
            threading.Thread(
                target=_prewarm_quiz_questions,
                args=(topic, quiz_text),
                daemon=True
            ).start()
        except Exception as prewarm_err:
            app.logger.warning(f"Failed to start pre-warming: {prewarm_err}")
        return jsonify({'quiz': quiz_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/store-answer', methods=['POST'])
def store_answer():
    data = request.json
    topic = data.get('topic', '')
    question = data.get('question', '')
    answer = data.get('answer', '')
    timestamp = data.get('timestamp', '')
    
    answer_data = {
        'topic': topic,
        'question': question,
        'answer': answer,
        'timestamp': timestamp
    }
    
    try:
        # Store answers in a JSON file
        filename = 'quiz_answers.json'
        try:
            with open(filename, 'r') as f:
                all_answers = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_answers = []
        
        all_answers.append(answer_data)
        
        with open(filename, 'w') as f:
            json.dump(all_answers, f, indent=2)
        
        return jsonify({'success': True, 'message': 'Answer stored successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/explain', methods=['POST'])
def explain_topic():
    data = request.json
    topic = data.get('topic', '')
    level = data.get('level', 'beginner')
    
    prompt = f"""Explain the topic '{topic}' at a {level} level.

Structure your explanation with these sections, using markdown headings:
## Definition
A simple, clear one-to-two sentence definition.

## Key Concepts
Break down the main ideas into digestible bullet points, explaining each one.

## How It Works
A step-by-step or detailed breakdown of the mechanism or process.

## Real-World Example
A concrete example or analogy that makes the concept relatable and easy to grasp.

## Common Misconceptions
Highlight 2-3 things people often get wrong about this topic and clarify them.

## Quick Summary
A brief recap of the most important takeaways.

Be thorough — provide enough detail for the student to truly understand the topic, but keep each section readable. Use bullet points, bold text, and short paragraphs."""
    
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert educator who explains topics clearly and engagingly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.7,
            max_tokens=650,
            stream=False
        )
        
        explanation = completion.choices[0].message.content
        try:
            from verification.study_knowledge import (
                build_study_knowledge_from_explanation,
                precompute_knowledge_embeddings,
                store_study_knowledge,
                get_json_safe_knowledge
            )
            knowledge = build_study_knowledge_from_explanation(topic, level, explanation)
            store_study_knowledge(knowledge)
            
            # Precompute embeddings in background thread so response returns instantly
            def _bg_embed(k):
                try:
                    precompute_knowledge_embeddings(k, client=client, config=verification_pipeline.config)
                    store_study_knowledge(k)
                    app.logger.info(f"Background embeddings precomputed for topic '{topic}'")
                except Exception as bg_err:
                    app.logger.warning(f"Background embedding precomputation failed: {bg_err}")
            
            threading.Thread(target=_bg_embed, args=(knowledge,), daemon=True).start()
            
            app.logger.info(f"Generated & stored study knowledge for topic '{topic}' with {len(knowledge.get('concepts', []))} concepts")
            return jsonify({'explanation': explanation, 'study_knowledge': get_json_safe_knowledge(knowledge)})
        except Exception as sk_err:
            app.logger.warning(f"Error building study knowledge: {sk_err}")
            return jsonify({'explanation': explanation})
    except Exception as e:
        app.logger.exception(f"Error explaining topic '{topic}': {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Whisper speech-to-text endpoint using local faster-whisper"""
    try:
        data = request.json or {}
        audio_base64 = data.get('audio', '')
        
        if not audio_base64:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            from verification.whisper_service import transcribe, SpeechRecognitionError
            whisper_cfg = verification_config.whisper_cfg
            result = transcribe(
                tmp_path,
                llm_client=client,
                model_size=whisper_cfg.get("model_size", "base"),
                api_model=whisper_cfg.get("api_model", "whisper-1"))
            return jsonify({'text': result.get('transcript', '')})
        except SpeechRecognitionError as sre:
            return jsonify({'text': '', 'error': sre.user_message})
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ocr', methods=['POST'])
def ocr_image():
    """OpenCV OCR endpoint"""
    try:
        data = request.json
        image_base64 = data.get('image', '')
        
        if not image_base64:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Preprocess image with OpenCV for better OCR
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to enhance text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
        
        # Use Tesseract OCR (which uses OpenCV for preprocessing)
        text = pytesseract.image_to_string(denoised, config='--psm 6')
        
        return jsonify({'text': text.strip()})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

from verification.pipeline import UnderstandingVerificationPipeline
from verification.config import VerificationConfig
from verification.opencv_service import (
    MediaProcessingError,
    load_transcript as load_recording_meta,
    process_video,
    secure_save_video,
    store_transcript,
)
from verification.whisper_service import (
    SpeechRecognitionError,
    transcribe,
    warm_up_local_whisper,
)

# Student Understanding Verification module (AUVA pipeline)
verification_config = VerificationConfig()
verification_pipeline = UnderstandingVerificationPipeline(
    client=client, config=verification_config)


def _preload_models():
    """Pre-load models in a background thread so the first request does not pay model-loading cost."""
    try:
        whisper_cfg = verification_config.whisper_cfg
        model_size = whisper_cfg.get("model_size", "base")
        if warm_up_local_whisper(model_size):
            app.logger.info("Whisper model pre-loaded (%s)", model_size)
    except Exception:
        app.logger.warning("Whisper pre-load skipped; will load lazily", exc_info=True)

    try:
        from verification.semantic import _st_model
        _st_model("all-MiniLM-L6-v2")
        app.logger.info("SentenceTransformer pre-loaded")
    except Exception:
        app.logger.warning("SentenceTransformer pre-load skipped; will load lazily", exc_info=True)


# Fire-and-forget background warm-up so the server stays responsive.
threading.Thread(target=_preload_models, daemon=True).start()


class _FriendlyError(Exception):
    """API-level error carrying a USER-FRIENDLY message only."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.user_message = message
        self.status = status


@app.route('/api/recording/upload', methods=['POST'])
def upload_recording():
    """Receive the student's recorded explanation.

    Pipeline: uploaded video -> OpenCV secure storage -> OpenCV frame-by-frame
    visual processing -> audio extraction -> Whisper -> stored transcript.

    Returns {recording_id, transcript, language, segments, video_analysis}.
    The transcript is persisted so it can be passed to the verification AI.
    """
    try:
        video_file = request.files.get('video')
        if video_file is None or not video_file.filename:
            raise _FriendlyError(
                'No recording was received. Please record your explanation '
                'and submit again.')

        # 1) OpenCV service: secure storage
        video_path = secure_save_video(video_file, video_file.filename)

        # 2) OpenCV service: frame-by-frame visual processing
        video_analysis = process_video(video_path)

        # 3) Whisper: extract audio -> speech-to-text
        whisper_cfg = verification_config.whisper_cfg
        whisper_result = transcribe(
            video_path,
            llm_client=client,
            model_size=whisper_cfg.get("model_size", "base"),
            api_model=whisper_cfg.get("api_model", "whisper-1"))

        # 4) Store the transcript so it can feed the verification AI
        store_transcript(video_path, whisper_result, video_analysis)

        return jsonify({
            'recording_id': os.path.basename(video_path),
            'transcript': whisper_result['transcript'],
            'language': whisper_result.get('language', ''),
            'segments': whisper_result.get('segments', []),
            'audio_quality': whisper_result.get('audio_quality'),
            'video_analysis': video_analysis,
        })
    except (MediaProcessingError, SpeechRecognitionError) as exc:
        return jsonify({'error': exc.user_message}), 400
    except _FriendlyError as exc:
        return jsonify({'error': exc.user_message}), exc.status
    except Exception:
        app.logger.exception('recording upload failed')
        return jsonify({'error':
                        "We couldn't process your recording right now. "
                        'Please try again in a moment.'}), 500


@app.route('/api/transcribe/live', methods=['POST'])
def transcribe_live():
    """Whisper speech-to-text for an IN-PROGRESS recording (live preview).

    Receives the audio/video captured by the browser so far and returns the
    transcript of everything spoken up to this point so the student can watch
    their answer appear as text while they explain.

    PREVIEW ONLY - the authoritative transcript used by the verification/AUVA
    pipeline is always regenerated from the complete uploaded recording by
    POST /api/recording/upload.
    """
    try:
        video_file = request.files.get('video')
        if video_file is None or not video_file.filename:
            return jsonify({'text': '', 'speech_detected': False})

        ext = os.path.splitext(video_file.filename)[1] or '.webm'
        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            video_file.save(tmp_path)
            whisper_cfg = verification_config.whisper_cfg
            result = transcribe(
                tmp_path,
                llm_client=client,
                model_size=whisper_cfg.get("model_size", "base"),
                api_model=whisper_cfg.get("api_model", "whisper-1"))
            quality = result.get('audio_quality') or {}
            return jsonify({'text': result.get('transcript', ''),
                            'language': result.get('language', ''),
                            'speech_detected': True,
                            'rms': quality.get('rms'),
                            'quality_label': quality.get('quality_label')})
        except SpeechRecognitionError as exc:
            # Expected mid-recording (e.g. silence while the student thinks).
            # The live preview stays quiet instead of showing an error, but we
            # pass back the mic level so the UI can warn when no sound reaches
            # the microphone at all (a common cause of empty transcripts).
            quality = getattr(exc, 'audio_quality', {}) or {}
            return jsonify({'text': '', 'speech_detected': False,
                            'hint': exc.user_message,
                            'rms': quality.get('rms'),
                            'quality_label': quality.get('quality_label'),
                            'silent': quality.get('quality_label') in
                                       ('poor', 'empty', 'unreadable')})
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    except Exception:
        app.logger.exception('live transcription failed')
        return jsonify({'text': '', 'speech_detected': False})


@app.route('/api/verify-understanding', methods=['POST'])
def verify_understanding():
    """Student Understanding Verification & Analysis (AUVA) endpoint.

    Accepts JSON with:
      subject, topic, expected_representation, transcript,
      video_analysis   (optional visual-analysis results),
      audio_analysis   (optional pre-computed acoustic metrics),
      emotion_analysis (optional facial-expression results),
      audio            (optional base64 WAV for live acoustic analysis),
      transcript_quality (optional ASR quality signal, e.g. Whisper avg_logprob),
      audio_quality    (optional 0-100 recording-quality score)
    """
    try:
        data = request.json or {}
        topic = data.get('topic', '')
        expected = data.get('expected_representation', '')

        # Stored Whisper transcript (from /api/recording/upload) is used as
        # the primary textual input when provided via recording_id.
        stored_transcript, stored_video_analysis = _resolve_recording(data)
        transcript = data.get('transcript', '') or stored_transcript or ''
        video_analysis = data.get('video_analysis') or stored_video_analysis

        if not topic or not expected or not transcript:
            return jsonify({'error': 'topic, expected_representation and '
                                     'transcript are required'}), 400

        audio_bytes = None
        if data.get('audio'):
            import base64 as _base64
            audio_bytes = _base64.b64decode(data['audio'])

        result = verification_pipeline.run(
            subject=data.get('subject', ''),
            topic=topic,
            expected_representation=expected,
            transcript=transcript,
            video_analysis=video_analysis,
            audio_analysis=data.get('audio_analysis'),
            emotion_analysis=data.get('emotion_analysis'),
            audio_bytes=audio_bytes,
            transcript_quality=data.get('transcript_quality'),
            audio_quality=data.get('audio_quality'),
        )
        return jsonify(result)
    except _FriendlyError as exc:
        return jsonify({'error': exc.user_message}), exc.status
    except Exception:
        app.logger.exception('verify-understanding failed')
        return jsonify({'error':
                        "We couldn't verify your understanding right now. "
                        'Please try again in a moment.'}), 500


def _resolve_recording(data):
    """Attach a stored Whisper transcript / visual metrics to a request.

    If ``recording_id`` is present, the persisted OpenCV + Whisper results are
    returned so the verification AI uses the stored transcript as its primary
    textual input.
    """
    recording_id = data.get('recording_id')
    if not recording_id:
        return None, None
    meta = load_recording_meta(recording_id)
    if not meta:
        raise _FriendlyError(
            'We could not find the submitted recording. Please record and '
            'submit your explanation again.')
    whisper = meta.get('whisper') or {}
    return (whisper.get('transcript', ''),
            meta.get('video_analysis'))


# Cache of auto-generated expected representations for quiz questions
_quiz_expected_cache = {}


def _generate_expected_representation(topic, question):
    """Generate (and cache) the expected concept representation for a quiz question.

    This is the reference description of what a correct understanding of the
    question must contain; it feeds the AUVA pipeline's semantic and concept
    coverage analysis.
    """
    key = (topic or '').strip().lower() + '::' + (question or '').strip()
    if key in _quiz_expected_cache:
        return _quiz_expected_cache[key]

    prompt = f"""Topic: {topic}
Quiz question: {question}

Write the EXPECTED CONCEPT REPRESENTATION for this question - that is, what a
correct student understanding must contain. Describe the key concepts,
mechanisms and relationships that a complete, correct answer should cover.
Be concise (3-6 sentences). Output only the description - do not address
the student and do not answer as if you were the student."""
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system",
                 "content": "You are an expert educator who defines what a correct student understanding looks like."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            top_p=0.7,
            max_tokens=300,
            stream=False
        )
        expected = completion.choices[0].message.content.strip()
    except Exception:
        # Fallback: the question itself acts as a weak lexical representation
        expected = f"{topic}. {question}".strip('. ')

    _quiz_expected_cache[key] = expected
    return expected


def _prewarm_quiz_questions(topic, quiz_text):
    """Pre-warm expected representations and extracted concepts for generated quiz questions in the background."""
    questions = []
    for line in (quiz_text or "").split('\n'):
        line = line.strip()
        m = re.match(r"^Q\d+[:.]?\s*(.+)$", line, re.IGNORECASE)
        if m:
            questions.append(m.group(1).strip())
        elif line.startswith(("1.", "2.", "3.", "4.", "5.")):
            parts = line.split('.', 1)
            if len(parts) > 1:
                questions.append(parts[1].strip())

    for q in questions:
        try:
            expected = _generate_expected_representation(topic, q)
            # Pre-extract concepts into verification pipeline cache
            from verification.pipeline import _concept_cache, _concept_cache_key
            import verification.concepts as concept_mod
            cache_key = _concept_cache_key(expected, f"{topic}: {q}".strip(': ') if topic else q)
            if cache_key not in _concept_cache:
                concepts = concept_mod.extract_expected_concepts(
                    expected, f"{topic}: {q}".strip(': ') if topic else q,
                    client=client, config=verification_pipeline.config, subject=topic
                )
                if concepts:
                    _concept_cache[cache_key] = concepts
                    concept_mod.get_or_create_concept_embeddings(concepts, client=client, config=verification_pipeline.config)
        except Exception as e:
            app.logger.warning(f"Error pre-warming question '{q[:30]}...': {e}")


@app.route('/api/verify-quiz-answer', methods=['POST'])
def verify_quiz_answer():
    """Run the AUVA understanding-verification pipeline on a single quiz answer.

    Accepts JSON with:
      topic        (quiz subject/topic),
      question     (the quiz question that was answered),
      answer       (optional explicit transcript; normally omitted),
      recording_id (id returned by /api/recording/upload - the stored Whisper
                    transcript is used as the primary textual input and the
                    OpenCV visual metrics feed the attention analysis),
      audio        (optional base64 WAV recording of the spoken answer)
    """
    try:
        data = request.json or {}
        topic = data.get('topic', '')
        question = data.get('question', '')

        # The Whisper transcript stored with the submitted recording is the
        # primary textual input for verification.
        answer, video_analysis = _resolve_recording(data)
        if not answer:
            answer = data.get('answer', '')

        if not question or not answer:
            return jsonify({'error':
                            'No transcript was found for this answer. '
                            'Please record and submit your explanation '
                            'again.'}), 400

        audio_bytes = None
        if data.get('audio'):
            audio_bytes = base64.b64decode(data['audio'])

        # IMMEDIATE SHORT-CIRCUIT: non-speech / silence bypasses all LLM and embedding overhead
        clean_answer = str(answer or '').strip()
        is_non_speech = (not clean_answer or len(clean_answer) < 5 or
                         clean_answer.lower() in ('(no speech detected)', 'no speech', 'saved automatically'))

        if is_non_speech:
            result = verification_pipeline.run(
                subject=topic,
                topic=f"{topic}: {question}".strip(': ') if topic else question,
                expected_representation="",
                transcript=answer,
                audio_bytes=audio_bytes,
                video_analysis=video_analysis,
            )
            result['expected_representation'] = ""
            return jsonify(result)

        from verification.study_knowledge import get_study_knowledge
        study_knowledge = get_study_knowledge(topic)

        # FAST LOCAL PATH: If study knowledge exists, use stored ideal answer / definition
        # instead of calling LLM for _generate_expected_representation!
        if study_knowledge and study_knowledge.get("ideal_answers"):
            expected = study_knowledge["ideal_answers"][0]
        elif study_knowledge and study_knowledge.get("definition"):
            expected = study_knowledge["definition"]
        else:
            expected = _generate_expected_representation(topic, question)

        result = verification_pipeline.run(
            subject=topic,
            topic=f"{topic}: {question}".strip(': ') if topic else question,
            expected_representation=expected,
            transcript=answer,
            audio_bytes=audio_bytes,
            video_analysis=video_analysis,
            study_knowledge=study_knowledge
        )
        # Include the generated reference so the UI can show what was expected
        result['expected_representation'] = expected
        return jsonify(result)
    except _FriendlyError as exc:
        return jsonify({'error': exc.user_message}), exc.status
    except Exception:
        app.logger.exception('verify-quiz-answer failed')
        return jsonify({'error':
                        "We couldn't verify this answer right now. Please "
                        'try again in a moment.'}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)