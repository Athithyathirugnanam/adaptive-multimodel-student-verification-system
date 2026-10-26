// Main JavaScript for the Student Learning Assistant

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const chatBotIcon = document.getElementById('chat-bot-icon');
const chatWidget = document.getElementById('chat-widget');
const chatWidgetClose = document.getElementById('chat-widget-close');

// State
let chatHistory = [];
let isStreaming = false;
let currentStudyContext = {
    topic: '',
    level: 'beginner',
    explanation: '',
    selectedConcept: ''
};

// Chat Bot Icon Toggle
chatBotIcon.addEventListener('click', () => {
    chatWidget.classList.remove('hidden');
    chatBotIcon.classList.add('hidden');
    chatInput.focus();
});

chatWidgetClose.addEventListener('click', () => {
    chatWidget.classList.add('hidden');
    chatBotIcon.classList.remove('hidden');
});

// Chat functionality
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = type === 'bot' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMessage(content);
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return contentDiv;
}

function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.id = 'typing-indicator';
    
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Comprehensive Markdown -> HTML converter.
// Supports: headings (##/###), bullet lists (-, *, +), ordered lists (1.),
// bold (**text**), italic (*text*), inline code (`code`), code blocks
// (```...```), blockquotes (> text), horizontal rules (---), and paragraphs.
// Block-level structure is detected BEFORE HTML escaping so markers like
// "> " and "## " are never transformed by escaping.
function formatMessage(text) {
    if (!text) return '';

    // Step 1: extract fenced code blocks and replace them with placeholders
    // so their content is never parsed as markdown or escaped.
    const codeBlocks = [];
    text = text.replace(/```(?:\w+)?\n([\s\S]*?)```/g, (m, code) => {
        codeBlocks.push(code);
        return '\x00CODEBLOCK' + (codeBlocks.length - 1) + '\x00';
    });

    // Step 2: process block-level structures line by line
    const lines = text.split('\n');
    const out = [];
    let listType = null;       // 'ul' or 'ol' while collecting consecutive items

    function flushList() {
        if (listType) {
            out.push('</' + listType + '>');
            listType = null;
        }
    }

    for (let i = 0; i < lines.length; i++) {
        const raw = lines[i];
        const trimmed = raw.trim();

        // Horizontal rules (---, ***, ___)
        if (/^( {0,3})(-{3,}|\*{3,}|_{3,})\s*$/.test(raw)) {
            flushList();
            out.push('<hr>');
            continue;
        }

        // Headings
        const h = raw.match(/^(#{1,6})\s+(.+)$/);
        if (h) {
            flushList();
            out.push('<h' + h[1].length + '>' + inlineFormat(h[2]) + '</h' + h[1].length + '>');
            continue;
        }

        // Blockquotes (can span multiple lines)
        const bq = raw.match(/^>\s?(.*)$/);
        if (bq) {
            flushList();
            out.push('<blockquote>' + inlineFormat(bq[1]) + '</blockquote>');
            continue;
        }

        // Unordered list items (-, *, +)
        const ul = raw.match(/^\s*[*+\-]\s+(.+)$/);
        // Ordered list items (1. 2. ...)
        const ol = raw.match(/^\s*(\d+)[\.\)]\s+(.+)$/);

        if (ol && !ul) {
            if (listType !== 'ol') { flushList(); out.push('<ol>'); listType = 'ol'; }
            out.push('<li>' + inlineFormat(ol[2]) + '</li>');
            continue;
        } else if (ul) {
            if (listType !== 'ul') { flushList(); out.push('<ul>'); listType = 'ul'; }
            out.push('<li>' + inlineFormat(ul[1]) + '</li>');
            continue;
        }

        // Plain paragraph text
        flushList();
        if (trimmed) {
            out.push('<p>' + inlineFormat(trimmed) + '</p>');
        } else {
            out.push('<p></p>'); // preserve paragraph breaks
        }
    }
    flushList();

    // Step 3: restore code blocks (escaped content, wrapped in <pre>)
    let html = out.join('\n');
    html = html.replace(/\x00CODEBLOCK(\d+)\x00/g, (m, i) => {
        return '<pre><code>' + escapeHtml(codeBlocks[parseInt(i, 10)]) + '</code></pre>';
    });

    // Collapse consecutive empty paragraphs used as section breaks
    while (html.indexOf('<p></p>\n<p></p>') !== -1) {
        html = html.replace('<p></p>\n<p></p>', '<p></p>');
    }
    html = html.replace(/<p><\/p>\n/g, '');

    return html;
}

// Inline formatting: escape HTML, then apply bold / italic / inline code.
function inlineFormat(text) {
    let html = escapeHtml(text);
    // Bold **text** or __text__
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/__(.+?)__/g, '<strong>$1</strong>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Italic *text* (but not inside ** or __)
    html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    return html;
}

// escapeHtml is defined near the end of this file (used by both the quiz
// feedback and inlineFormat). It escapes &, <, > so LLM content can never
// inject raw HTML.

// Fast, minimal formatter for streaming chunks — keeps the UI responsive
// by avoiding expensive regex passes on the full accumulated text.
function formatMessageStream(text) {
    if (!text) return '';
    let html = escapeHtml(text);

    // Headings
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Horizontal rules
    html = html.replace(/^( {0,3})(-{3,}|\*{3,}|_{3,})\s*$/gm, '<hr>');

    // Bold
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Newlines -> <br>
    html = html.replace(/\n/g, '<br>');

    return html;
}

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isStreaming) return;
    
    // Add user message
    addMessage(message, 'user');
    chatHistory.push({ role: 'user', content: message });
    
    // Clear input
    chatInput.value = '';
    chatInput.style.height = 'auto';
    
    // Add typing indicator
    addTypingIndicator();
    isStreaming = true;
    sendBtn.disabled = true;
    
    try {
        const activeTopic = currentStudyContext.topic || searchedTopic || '';
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory.slice(0, -1), // Send history without current message
                topic: activeTopic,
                study_context: currentStudyContext.explanation || '',
                selected_concept: currentStudyContext.selectedConcept || ''
            })
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Create bot message container
        const botMessageDiv = document.createElement('div');
        botMessageDiv.className = 'message bot';
        
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = '<i class="fas fa-robot"></i>';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        botMessageDiv.appendChild(avatar);
        botMessageDiv.appendChild(contentDiv);
        chatMessages.appendChild(botMessageDiv);
        
        // Stream the response with throttled DOM updates for responsiveness
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullResponse = '';
        let lastUpdate = 0;
        const throttleMs = 50;

        const finalizeResponse = () => {
            contentDiv.innerHTML = formatMessage(fullResponse);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));

                    if (data.error) {
                        contentDiv.innerHTML = `<p style="color: #ef4444;">Error: ${data.error}</p>`;
                        break;
                    }

                    if (data.done) {
                        finalizeResponse();
                        break;
                    }

                    if (data.content) {
                        fullResponse += data.content;
                        const now = Date.now();
                        if (now - lastUpdate >= throttleMs) {
                            contentDiv.innerHTML = formatMessageStream(fullResponse);
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                            lastUpdate = now;
                        }
                    }
                }
            }
        }

        chatHistory.push({ role: 'assistant', content: fullResponse });
        
    } catch (error) {
        removeTypingIndicator();
        addMessage('Sorry, there was an error connecting to the AI. Please try again.', 'bot');
        console.error('Error:', error);
    } finally {
        isStreaming = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

// Event listeners for chat
sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 100) + 'px';
});

// Quiz functionality - Text-based quiz
const generateQuizBtn = document.getElementById('generate-quiz-btn');
const quizResult = document.getElementById('quiz-result');
const quizContent = document.getElementById('quiz-content');
const newQuizBtn = document.getElementById('new-quiz-btn');
const startQuizBtn = document.getElementById('start-quiz-btn');
const nextQuestionBtn = document.getElementById('next-question-btn');
const transcribedAnswer = document.getElementById('transcribed-answer');
const answerDisplay = document.querySelector('.answer-display');
// Camera recorder elements (replaces the old text-answer box)
const answerRecorderEl = document.getElementById('answer-recorder');
const cameraPreview = document.getElementById('camera-preview');
const recordingPlayback = document.getElementById('recording-playback');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const enableMediaBtn = document.getElementById('enable-media-btn');
const recordingIndicator = document.getElementById('recording-indicator');
const micStatusEl = document.getElementById('mic-status');
const camStatusEl = document.getElementById('cam-status');
const recTimerEl = document.getElementById('rec-timer');
const recorderMessageEl = document.getElementById('recorder-message');
const micLevelEl = document.getElementById('mic-level');
const micLevelFillEl = document.getElementById('mic-level-fill');
const micLevelTextEl = document.getElementById('mic-level-text');
const micDeviceSelect = document.getElementById('mic-device');
// Live transcript ("Your Answer") panel shown below the camera preview
const liveTranscriptPanelEl = document.getElementById('live-transcript-panel');
const liveTranscriptText = document.getElementById('live-transcript-text');
const liveTranscriptStatus = document.getElementById('live-transcript-status');
const startRecordingBtn = document.getElementById('start-recording-btn');
const pauseRecordingBtn = document.getElementById('pause-recording-btn');
const stopRecordingBtn = document.getElementById('stop-recording-btn');
const rerecordBtn = document.getElementById('rerecord-btn');
const submitAnswerBtn = document.getElementById('submit-answer-btn');
const currentQuestionSpan = document.getElementById('current-question');
const questionCountSpan = document.getElementById('question-count');
const quizTopicDisplay = document.getElementById('quiz-topic-display');
const proctoringStatusBar = document.getElementById('proctoring-status-bar');
const proctoringWarningBadge = document.getElementById('proctoring-warning-badge');

// Quiz state
let quizQuestions = [];
let currentQuestionIndex = 0;
let quizTopic = '';
let isQuizActive = false;
let currentAnswer = '';
let searchedTopic = '';

// Quiz answer verification (AUVA) state
const answerFeedbackEl = document.getElementById('answer-feedback');
const quizSummaryEl = document.getElementById('quiz-summary');
let quizVerificationResults = [];

// Parse quiz questions from text
function parseQuizQuestions(quizText) {
    const questions = [];
    const lines = quizText.split('\n');
    let currentQuestion = '';
    
    for (const line of lines) {
        if (line.match(/^Q\d+:/i)) {
            if (currentQuestion) {
                questions.push(currentQuestion.trim());
            }
            currentQuestion = line.replace(/^Q\d+:\s*/i, '');
        } else if (currentQuestion) {
            currentQuestion += ' ' + line.trim();
        }
    }
    
    if (currentQuestion) {
        questions.push(currentQuestion.trim());
    }
    
    return questions;
}

// Display current question
function displayQuestion() {
    if (currentQuestionIndex >= quizQuestions.length) {
        // Quiz complete - show aggregate understanding summary
        quizContent.innerHTML = '<h3>🎉 Quiz Complete!</h3><p>Your answers have been recorded and verified below.</p>';
        startQuizBtn.classList.add('hidden');
        nextQuestionBtn.classList.add('hidden');
        answerDisplay.classList.add('hidden');
        shutdownRecorder();
        // The quiz is done - hide the recorder so the aggregate score
        // summary (renderQuizSummary) is the focus.
        answerRecorderEl.classList.add('hidden');
        isQuizActive = false;
        examState = 'SUBMITTED';
        if (window.headMovementCheckInterval) clearInterval(window.headMovementCheckInterval);
        stopProctoringLoop();
        renderQuizSummary();
        
        // Restore normal flow
        const studySectionEl = document.getElementById('study-section');
        if (studySectionEl) studySectionEl.classList.remove('hidden');
        const chatBotIconEl = document.getElementById('chat-bot-icon');
        if (chatBotIconEl) chatBotIconEl.classList.remove('hidden');
        
        return;
    }
    
    currentQuestionSpan.textContent = `Question ${currentQuestionIndex + 1}`;
    questionCountSpan.textContent = `of ${quizQuestions.length}`;
    quizContent.innerHTML = `<p><strong>Q${currentQuestionIndex + 1}:</strong> ${quizQuestions[currentQuestionIndex]}</p>`;
    
    // Reset answer display and the camera recorder (no text-answer box)
    transcribedAnswer.textContent = '';
    currentAnswer = '';
    answerDisplay.classList.add('hidden');

    // Reset per-answer verification feedback for the new question
    if (answerFeedbackEl) {
        answerFeedbackEl.classList.add('hidden');
        answerFeedbackEl.innerHTML = '';
    }

    prepareRecorderForQuestion();
    // Reveal the camera recording interface so the student can record and
    // submit their answer for this question (and then move on to the next).
    answerRecorderEl.classList.remove('hidden');

    // Show next question button
    nextQuestionBtn.classList.add('hidden');
}

// Start quiz
let examState = 'NOT_STARTED'; // NOT_STARTED, VERIFYING, READY, FULLSCREEN, ACTIVE, ANOMALY_DETECTED, SUBMITTING, SUBMITTED
let examConfig = {
    REQUIRE_FULLSCREEN: true,
    REQUIRE_CAMERA: true,
    REQUIRE_MICROPHONE: true,
    AUTO_SUBMIT_ON_TAB_SWITCH: true,
    AUTO_SUBMIT_ON_FULLSCREEN_EXIT: true
};
let currentQuizId = '';
let examSubmitted = false;

// Fetch config on load
fetch('/api/config')
    .then(res => res.json())
    .then(config => {
        if (!config.error) {
            Object.assign(examConfig, config);
        }
    })
    .catch(console.error);

let _proctoringCanvas = null;

async function captureFaceImage() {
    if (!recorderState.stream) return null;
    const videoTrack = recorderState.stream.getVideoTracks()[0];
    if (!videoTrack || videoTrack.readyState !== 'live') return null;
    
    // Fast path: draw directly from the active cameraPreview video element
    const width = cameraPreview.videoWidth || 640;
    const height = cameraPreview.videoHeight || 480;
    if (width === 0 || height === 0) return null;

    // Constrain resolution to max 640x480 for sub-5ms capture and lightweight network transmission
    const targetWidth = Math.min(640, width);
    const targetHeight = Math.round((targetWidth / width) * height);

    if (!_proctoringCanvas) {
        _proctoringCanvas = document.createElement('canvas');
    }
    if (_proctoringCanvas.width !== targetWidth || _proctoringCanvas.height !== targetHeight) {
        _proctoringCanvas.width = targetWidth;
        _proctoringCanvas.height = targetHeight;
    }

    const ctx = _proctoringCanvas.getContext('2d', { alpha: false });
    ctx.drawImage(cameraPreview, 0, 0, targetWidth, targetHeight);
    return _proctoringCanvas.toDataURL('image/jpeg', 0.65);
}


startQuizBtn.addEventListener('click', async () => {
    if (quizQuestions.length === 0) {
        alert('No questions available. Please generate a quiz first.');
        return;
    }
    
    examState = 'VERIFYING';
    currentQuizId = 'quiz_' + Date.now();
    examSubmitted = false;
    
    quizContent.innerHTML = '<h3><i class="fas fa-shield-alt"></i> Security Verification</h3><p>Verifying permissions and environment...</p>';
    startQuizBtn.classList.add('hidden');
    
    try {
        if (examConfig.REQUIRE_CAMERA || examConfig.REQUIRE_MICROPHONE) {
            quizContent.innerHTML += '<p>Requesting camera and microphone access...</p>';
            if (!recorderState.stream) {
                await enableCameraAndMic();
            }
            if (!recorderState.stream) {
                throw new Error("Camera or microphone permission denied.");
            }
            if (examConfig.REQUIRE_CAMERA && !recorderState.stream.getVideoTracks().length) {
                throw new Error("Camera is required but not found.");
            }
            if (examConfig.REQUIRE_MICROPHONE && !recorderState.stream.getAudioTracks().length) {
                throw new Error("Microphone is required but not found.");
            }
        }
        
        if (examConfig.REQUIRE_CAMERA) {
            quizContent.innerHTML += '<p>Verifying face presence...</p>';
            // Small delay to allow camera exposure to adjust only if just enabled
            await new Promise(r => setTimeout(r, 200)); 
            const faceImage = await captureFaceImage();
            if (faceImage) {
                const faceRes = await fetch('/api/verification/face', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ image: faceImage })
                });
                const faceData = await faceRes.json();
                if (!faceData.success) {
                    throw new Error(faceData.error || "Face verification failed.");
                }
            }
        }
        
        examState = 'READY';
        
        if (examConfig.REQUIRE_FULLSCREEN) {
            quizContent.innerHTML += '<p>Requesting fullscreen...</p>';
            try {
                await document.documentElement.requestFullscreen();
                await new Promise((resolve, reject) => {
                    const checkFullscreen = () => {
                        if (document.fullscreenElement) resolve();
                        else setTimeout(checkFullscreen, 100);
                    };
                    checkFullscreen();
                    setTimeout(() => reject(new Error("Fullscreen request timed out.")), 3000);
                });
            } catch (e) {
                throw new Error("Could not enter fullscreen mode. " + e.message);
            }
        }
        
        examState = 'ACTIVE';
        isQuizActive = true;
        currentQuestionIndex = 0;
        
        // Hide the study section completely during the quiz
        const studySectionEl = document.getElementById('study-section');
        if (studySectionEl) studySectionEl.classList.add('hidden');
        
        // Hide the chat bot to prevent cheating
        const chatBotIconEl = document.getElementById('chat-bot-icon');
        if (chatBotIconEl) chatBotIconEl.classList.add('hidden');
        const chatWidgetEl = document.getElementById('chat-widget');
        if (chatWidgetEl) chatWidgetEl.classList.add('hidden');
        
        resetQuizVerificationUI();
        displayQuestion();
        
        // Start Real-Time AI Proctoring Detection (YOLO Phone + Person Count + MediaPipe Head Pose)
        startProctoringLoop();
        
    } catch (e) {
        examState = 'NOT_STARTED';
        quizContent.innerHTML = `<h3><i class="fas fa-exclamation-triangle"></i> Verification Failed</h3>
                                 <p>${e.message}</p>`;
        startQuizBtn.classList.remove('hidden');
        startQuizBtn.innerHTML = '<i class="fas fa-redo"></i> Retry Verification';
    }
});

let isProctoringBusy = false;
let isProctoringPaused = false;

function showProctoringWarning(title, message) {
    const existing = document.getElementById('proctoring-warning-banner');
    if (existing) existing.remove();

    const banner = document.createElement('div');
    banner.id = 'proctoring-warning-banner';
    banner.className = 'proctoring-warning-banner';
    banner.innerHTML = `<i class="fas fa-exclamation-triangle" style="font-size:1.4rem;"></i>
        <div>
            <div style="font-size:1.05rem; font-weight:800; margin-bottom:2px;">${escapeHtml(title)}</div>
            <div style="font-weight:600; font-size:0.95rem;">${escapeHtml(message)}</div>
        </div>`;
    document.body.appendChild(banner);

    setTimeout(() => {
        if (banner.parentNode) banner.parentNode.removeChild(banner);
    }, 5000);
}

function startProctoringLoop() {
    stopProctoringLoop();
    if (proctoringStatusBar) proctoringStatusBar.classList.remove('hidden');
    if (proctoringWarningBadge) {
        proctoringWarningBadge.textContent = 'Warnings: 0 / 3';
        proctoringWarningBadge.classList.remove('danger');
    }

    // Sampling rate of ~1.6 FPS (600ms) prevents server CPU/network bottlenecks while maintaining responsive detection
    window.proctoringInterval = setInterval(async () => {
        if (examState !== 'ACTIVE') {
            stopProctoringLoop();
            return;
        }
        if (isProctoringBusy || isProctoringPaused) return;

        isProctoringBusy = true;
        try {
            const faceImage = await captureFaceImage();
            if (!faceImage) {
                isProctoringBusy = false;
                return;
            }

            const res = await fetch('/api/proctoring/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    quiz_id: currentQuizId,
                    image: faceImage,
                    timestamp: Date.now()
                })
            });

            if (!res.ok) {
                isProctoringBusy = false;
                return;
            }

            const data = await res.json();
            if (proctoringWarningBadge && typeof data.warning_count === 'number') {
                proctoringWarningBadge.textContent = `Warnings: ${data.warning_count} / ${data.max_warnings || 3}`;
                if (data.warning_count >= 2) {
                    proctoringWarningBadge.classList.add('danger');
                } else {
                    proctoringWarningBadge.classList.remove('danger');
                }
            }

            // Display visual warning when confirmed anomaly occurs
            if (data.new_warning && data.warning_title) {
                showProctoringWarning(data.warning_title, data.warning_message || 'Proctoring anomaly detected.');
            }

            // 3 Warnings -> Automatic Test Submission
            if (data.auto_submit || (typeof data.warning_count === 'number' && data.warning_count >= 3)) {
                stopProctoringLoop();
                submitExam("Maximum number of proctoring warnings reached.", "PROCTORING_VIOLATION");
            }
        } catch (err) {
            console.warn("Proctoring analysis tick failed:", err);
        } finally {
            isProctoringBusy = false;
        }
    }, 600);
}

function stopProctoringLoop() {
    if (window.proctoringInterval) {
        clearInterval(window.proctoringInterval);
        window.proctoringInterval = null;
    }
    isProctoringBusy = false;
    isProctoringPaused = false;
    if (proctoringStatusBar) {
        proctoringStatusBar.classList.add('hidden');
    }
    const banner = document.getElementById('proctoring-warning-banner');
    if (banner) banner.remove();
}

async function submitExam(reason, eventType) {
    if (examSubmitted || examState !== 'ACTIVE') return;
    examSubmitted = true;
    examState = 'ANOMALY_DETECTED';
    if (window.headMovementCheckInterval) clearInterval(window.headMovementCheckInterval);
    stopProctoringLoop();
    
    shutdownRecorder();
    if (answerRecorderEl) answerRecorderEl.classList.add('hidden');
    if (answerDisplay) answerDisplay.classList.add('hidden');
    if (nextQuestionBtn) nextQuestionBtn.classList.add('hidden');
    
    quizContent.innerHTML = '<h3><i class="fas fa-spinner fa-spin"></i> Automatically Submitting Exam...</h3>';
    
    const anomalyData = {
        quiz_id: currentQuizId,
        student_id: 'student_123',
        event_type: eventType,
        timestamp: new Date().toISOString(),
        current_question: currentQuestionIndex + 1,
        severity: 'HIGH',
        reason: reason
    };
    
    try {
        const anomalyPromise = fetch('/api/verification/anomaly', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(anomalyData)
        });
        
        const savePromise = fetch('/api/quiz/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                quiz_id: currentQuizId,
                answers: currentAnswer || 'Saved automatically'
            })
        });
        
        const submitData = {
            quiz_id: currentQuizId,
            student_id: 'student_123',
            answers: {},
            submission_type: 'AUTO',
            reason: reason,
            event_type: eventType,
            timestamp: new Date().toISOString()
        };
        
        const submitPromise = fetch('/api/quiz/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(submitData)
        });
        
        await Promise.allSettled([anomalyPromise, savePromise, submitPromise]);
        
        examState = 'SUBMITTED';
        showResultPage(reason, eventType);
    } catch (e) {
        console.error("Error auto-submitting:", e);
        showResultPage(reason, eventType);
    }
}

function showResultPage(reason, eventType) {
    let reasonText = reason;
    let titleText = 'Exam Automatically Submitted';
    if (eventType === 'TAB_SWITCH') {
        reasonText = 'Tab switching detected';
    } else if (eventType === 'FULLSCREEN_EXIT') {
        reasonText = 'Fullscreen mode was exited during the examination.';
    } else if (eventType === 'PROCTORING_VIOLATION') {
        titleText = 'TEST SUBMITTED';
        reasonText = 'Maximum number of proctoring warnings reached.';
    }
    
    quizContent.innerHTML = `
        <div class="anomaly-result" style="border: 2px solid var(--danger, #ef4444); padding: 20px; border-radius: 8px; margin-top: 20px; background-color: #fef2f2; color: #991b1b;">
            <h3 style="color: #991b1b; margin-top: 0;"><i class="fas fa-exclamation-triangle"></i> ${titleText}</h3>
            <p><strong>Reason:</strong><br>${reasonText}</p>
            <p><strong>Verification Event:</strong><br>${eventType}</p>
            <p>Your answers have been saved and submitted.</p>
            <p><strong>Status:</strong><br>Review Required</p>
        </div>
    `;
    
    // Restore normal flow
    const studySectionEl = document.getElementById('study-section');
    if (studySectionEl) studySectionEl.classList.remove('hidden');
    const chatBotIconEl = document.getElementById('chat-bot-icon');
    if (chatBotIconEl) chatBotIconEl.classList.remove('hidden');
    
    if (document.fullscreenElement) {
        document.exitFullscreen().catch(err => console.error(err));
    }

    // Display summary of answered questions and detected misconceptions
    renderQuizSummary();
}

document.addEventListener('fullscreenchange', () => {
    if (examState === 'ACTIVE' && examConfig.AUTO_SUBMIT_ON_FULLSCREEN_EXIT) {
        if (!document.fullscreenElement) {
            submitExam("Fullscreen mode was exited during the examination.", "FULLSCREEN_EXIT");
        }
    }
});

document.addEventListener('visibilitychange', () => {
    if (examState === 'ACTIVE' && examConfig.AUTO_SUBMIT_ON_TAB_SWITCH) {
        if (document.visibilityState === 'hidden') {
            submitExam("Tab switch detected", "TAB_SWITCH");
        }
    }
});

// Periodic Autosave
setInterval(() => {
    if (examState === 'ACTIVE' && currentQuizId) {
        fetch('/api/quiz/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                quiz_id: currentQuizId,
                answers: currentAnswer || 'Autosave heartbeat'
            })
        }).catch(err => console.debug("Autosave failed", err));
    }
}, 30000); // Autosave every 30 seconds


// ============================================
// Camera Recorder (browser capture -> OpenCV + Whisper backend)
//
// The BROWSER owns permission handling (getUserMedia) and recording.
// The BACKEND owns OpenCV frame processing, secure storage and Whisper
// speech-to-text. The student never types their answer.
// ============================================

const recorderState = {
    stream: null,
    mediaRecorder: null,
    chunks: [],
    blob: null,
    state: 'idle',          // idle | ready | recording | paused | stopped | submitting
    seconds: 0,
    audioDeviceId: '',      // selected input device ('' = browser default mic)
    timerId: null,
    liveTimerId: null,      // interval id for periodic live Whisper passes
    liveBusy: false,        // a live-preview transcription request is in flight
    liveLastSize: 0,        // bytes of the recording already sent for preview
    audioContext: null,
    audioSource: null,
    analyser: null,
    audioMeterId: null,
};

function showRecorderMessage(text, type) {
    if (!recorderMessageEl) return;
    if (!text) {
        recorderMessageEl.classList.add('hidden');
        recorderMessageEl.textContent = '';
        return;
    }
    recorderMessageEl.textContent = text;
    recorderMessageEl.className = 'recorder-message ' + (type || 'info');
}

function updateDeviceStatusPills() {
    const stream = recorderState.stream;
    const audioTrack = stream && stream.getAudioTracks().find(t => t.enabled);
    const micOn = !!audioTrack && audioTrack.readyState === 'live';
    const camOn = !!(stream && stream.getVideoTracks().some(t => t.enabled && t.readyState === 'live'));
    micStatusEl.textContent = micOn ? '🎙 Microphone: Active' : '🎙 Microphone: Inactive';
    micStatusEl.className = 'status-pill ' + (micOn ? 'status-active' : 'status-inactive');
    if (micOn && audioTrack && audioTrack.label && micStatusEl.textContent === '🎙 Microphone: Active') {
        micStatusEl.textContent = `🎙 Microphone: ${audioTrack.label}`;
    }
    camStatusEl.textContent = camOn ? '🎥 Camera: Active' : '🎥 Camera: Inactive';
    camStatusEl.className = 'status-pill ' + (camOn ? 'status-active' : 'status-inactive');
}

// Real-time microphone audio visualizer using the Web Audio API
function startAudioMeter(stream) {
    stopAudioMeter();
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx || !stream || !stream.getAudioTracks().length) return;

        recorderState.audioContext = new AudioCtx();
        recorderState.audioSource = recorderState.audioContext.createMediaStreamSource(stream);
        recorderState.analyser = recorderState.audioContext.createAnalyser();
        recorderState.analyser.fftSize = 512;
        recorderState.audioSource.connect(recorderState.analyser);

        const bufferLength = recorderState.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        let silentFrames = 0;

        function checkAudio() {
            if (!recorderState.analyser) return;
            recorderState.analyser.getByteTimeDomainData(dataArray);

            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const val = (dataArray[i] - 128) / 128;
                sum += val * val;
            }
            const rms = Math.sqrt(sum / bufferLength);
            const level = Math.min(100, Math.round(rms * 400));

            if (micLevelFillEl) {
                micLevelFillEl.style.width = Math.max(0, level) + '%';
                micLevelFillEl.classList.toggle('low', level < 8);
            }

            if (level > 4) {
                silentFrames = 0;
                if (micLevelTextEl) {
                    micLevelTextEl.textContent = 'Microphone working ✓ (Voice detected)';
                    micLevelTextEl.classList.add('ok');
                    micLevelTextEl.classList.remove('warn');
                }
            } else {
                silentFrames++;
                if (silentFrames > 30 && micLevelTextEl) {
                    micLevelTextEl.textContent = '⚠️ Low/No Sound - check Windows mic volume or select device above';
                    micLevelTextEl.classList.add('warn');
                    micLevelTextEl.classList.remove('ok');
                }
            }

            recorderState.audioMeterId = setTimeout(checkAudio, 60);
        }

        checkAudio();
    } catch (err) {
        console.warn('Real-time audio meter could not be initialized:', err);
    }
}

function stopAudioMeter() {
    if (recorderState.audioMeterId) {
        clearTimeout(recorderState.audioMeterId);
        recorderState.audioMeterId = null;
    }
    if (recorderState.audioSource) {
        try { recorderState.audioSource.disconnect(); } catch (e) {}
        recorderState.audioSource = null;
    }
    if (recorderState.audioContext && recorderState.audioContext.state !== 'closed') {
        try { recorderState.audioContext.close(); } catch (e) {}
        recorderState.audioContext = null;
    }
    recorderState.analyser = null;
}

// List the microphone input devices available to the browser so the student
// can pick a working one (the default is sometimes silent/muted).
async function enumerateAudioDevices() {
    if (!micDeviceSelect || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const mics = devices.filter((d) => d.type === 'audioinput');
        if (!mics.length) return;
        const currentVal = micDeviceSelect.value;
        micDeviceSelect.innerHTML = '<option value=""' + (currentVal === '' ? ' selected' : '') + '>Default microphone</option>';
        mics.forEach((d) => {
            const label = (d.label || d.deviceId || 'Microphone').trim();
            const opt = document.createElement('option');
            opt.value = d.deviceId || '';
            opt.textContent = label;
            if (d.deviceId && d.deviceId === currentVal) {
                opt.selected = true;
            }
            micDeviceSelect.appendChild(opt);
        });
    } catch (err) {
        // Device listing is best-effort - never block recording setup.
        console.warn('Could not enumerate audio devices:', err);
    }
}

function formatRecorderTime(totalSeconds) {
    const m = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const s = String(totalSeconds % 60).padStart(2, '0');
    return `${m}:${s}`;
}

function updateRecorderButtons() {
    const st = recorderState.state;
    startRecordingBtn.disabled = !(st === 'ready' || st === 'stopped');
    pauseRecordingBtn.disabled = !(st === 'recording' || st === 'paused');
    pauseRecordingBtn.innerHTML = st === 'paused'
        ? '<i class="fas fa-play"></i> Resume'
        : '<i class="fas fa-pause"></i> Pause';
    stopRecordingBtn.disabled = !(st === 'recording' || st === 'paused');
    rerecordBtn.classList.toggle('hidden', st !== 'stopped');
    submitAnswerBtn.disabled = st !== 'stopped';
    recordingIndicator.classList.toggle('hidden', st !== 'recording');
    // Live preview while capturing; playback preview once stopped.
    cameraPreview.classList.toggle('hidden', st === 'stopped');
    recordingPlayback.classList.toggle('hidden', st !== 'stopped');
    cameraPlaceholder.classList.toggle('hidden', !!recorderState.stream);
}

async function enableCameraAndMic() {
    showRecorderMessage('');
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showRecorderMessage(
            'Your browser does not support camera recording. Please use a modern browser over HTTPS.',
            'error');
        return;
    }

    try {
        const audioConstraints = recorderState.audioDeviceId
            ? {
                deviceId: { ideal: recorderState.audioDeviceId },
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
              }
            : {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
              };

        try {
            recorderState.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: audioConstraints,
            });
        } catch (innerErr) {
            // Fallback to basic audio constraint if enhanced constraints fail
            recorderState.stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 } },
                audio: true,
            });
        }
    } catch (err) {
        recorderState.stream = null;
        const friendly = {
            NotAllowedError: 'Camera/microphone permission was denied. Please allow access in your browser settings and try again.',
            PermissionDeniedError: 'Camera/microphone permission was denied. Please allow access in your browser settings and try again.',
            NotFoundError: 'No camera or microphone was found. Please connect a device and try again.',
            DevicesNotFoundError: 'No camera or microphone was found. Please connect a device and try again.',
            NotReadableError: 'Your camera or microphone is already in use by another application. Close it and try again.',
            TrackStartError: 'Your camera or microphone could not be started. Please check your devices and try again.',
            OverconstrainedError: 'The selected camera settings are not supported. Please try a different device.',
            SecurityError: 'Camera access requires a secure (HTTPS) connection.',
        };
        showRecorderMessage(friendly[err.name] ||
            'We could not access your camera or microphone. Please check your device settings and try again.',
            'error');
        updateRecorderButtons();
        updateDeviceStatusPills();
        return;
    }

    cameraPreview.srcObject = recorderState.stream;
    await cameraPreview.play().catch(() => {});
    updateDeviceStatusPills();

    // Start live browser-side audio visualizer
    startAudioMeter(recorderState.stream);

    // Reflect live mute/unmute of individual tracks on the status pills
    recorderState.stream.getTracks().forEach(track => {
        track.addEventListener('ended', () => { updateDeviceStatusPills(); updateRecorderButtons(); });
        track.addEventListener('mute', updateDeviceStatusPills);
        track.addEventListener('unmute', updateDeviceStatusPills);
    });

    recorderState.state = 'ready';
    showRecorderMessage('');
    updateRecorderButtons();
    // Populate the mic device dropdown with the browser's available inputs.
    enumerateAudioDevices();
}

function pickRecorderMimeType() {
    const candidates = [
        'video/webm;codecs=vp8,opus',
        'video/webm;codecs=vp9,opus',
        'video/webm',
        'video/mp4',
    ];
    for (const type of candidates) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported(type)) return type;
    }
    return '';
}

function resetTimer() {
    clearInterval(recorderState.timerId);
    recorderState.timerId = null;
    recorderState.seconds = 0;
    recTimerEl.textContent = '00:00';
}

function tickTimer() {
    recorderState.seconds += 1;
    recTimerEl.textContent = formatRecorderTime(recorderState.seconds);
}

// --- Live transcript (Whisper preview while recording) ----------------------
// While the student speaks, the recording captured so far is periodically sent
// to POST /api/transcribe/live and the returned running transcript is shown in
// the "Your Answer" panel below the camera preview. PREVIEW ONLY: the
// transcript actually sent to the verification pipeline is always regenerated
// from the complete upload (POST /api/recording/upload) and synced back here.
const LIVE_TRANSCRIBE_INTERVAL_MS = 5000;

function setLiveTranscriptStatus(text, tone) {
    if (!liveTranscriptStatus) return;
    liveTranscriptStatus.textContent = text;
    liveTranscriptStatus.className =
        'live-transcript-status' + (tone ? ` ${tone}` : '');
}

function updateLiveTranscriptText(text) {
    if (!liveTranscriptText) return;
    liveTranscriptText.textContent = text;
    liveTranscriptText.classList.remove('placeholder');
    // Keep the newest words visible as the answer grows
    liveTranscriptText.scrollTop = liveTranscriptText.scrollHeight;
}

function resetLiveTranscript() {
    if (liveTranscriptText) {
        liveTranscriptText.textContent =
            'Your spoken words will appear here while you explain...';
        liveTranscriptText.classList.add('placeholder');
        liveTranscriptText.scrollTop = 0;
    }
    setLiveTranscriptStatus('Waiting...', '');
    resetMicLevel();
}

// Update the live microphone level meter from the backend audio analysis.
// A silent mic (rms near 0 while recording) is the most common reason a
// student speaks but no transcript appears - this makes it visible.
function resetMicLevel() {
    if (micLevelFillEl) micLevelFillEl.style.width = '0%';
    if (micLevelFillEl) micLevelFillEl.classList.remove('low');
    if (micLevelTextEl) {
        micLevelTextEl.textContent = 'Waiting...';
        micLevelTextEl.classList.remove('warn', 'ok');
    }
}

function updateMicLevel(rms, qualityLabel, silent) {
    if (!micLevelFillEl || !micLevelTextEl) return;
    const level = Math.max(0, Math.min(100, Math.round((rms || 0) * 10000)));
    micLevelFillEl.style.width = level + '%';
    micLevelFillEl.classList.toggle('low', level < 8);
    if (silent) {
        micLevelTextEl.textContent =
            '⚠ No sound - try a different microphone in the device list above, '
            + 'and check Windows microphone settings.';
        micLevelTextEl.classList.add('warn');
        micLevelTextEl.classList.remove('ok');
    } else if (qualityLabel && qualityLabel !== 'good') {
        micLevelTextEl.textContent = 'Low sound level - speak closer to the microphone.';
        micLevelTextEl.classList.add('warn');
        micLevelTextEl.classList.remove('ok');
    } else {
        micLevelTextEl.textContent = 'Microphone working ✓';
        micLevelTextEl.classList.add('ok');
        micLevelTextEl.classList.remove('warn');
    }
}

function stopLiveTranscription(statusText, tone) {
    if (recorderState.liveTimerId) {
        clearInterval(recorderState.liveTimerId);
        recorderState.liveTimerId = null;
    }
    recorderState.liveBusy = false;
    if (statusText !== undefined) setLiveTranscriptStatus(statusText, tone || '');
}

async function requestLiveTranscription(blob, filename) {
    const formData = new FormData();
    formData.append('video', blob, filename || 'live-preview.webm');
    const response = await fetch('/api/transcribe/live', {
        method: 'POST',
        body: formData,
    });
    return response.json();
}

async function transcribeLiveTick() {
    if (!recorderState.chunks.length ||
        recorderState.state !== 'recording' ||
        recorderState.liveBusy) return;
    const totalBytes = recorderState.chunks.reduce((s, c) => s + c.size, 0);
    if (!totalBytes || totalBytes === recorderState.liveLastSize) return;

    recorderState.liveBusy = true;
    setLiveTranscriptStatus('Transcribing...', 'processing');
    try {
        // Cumulative blob = every chunk captured so far. The first chunk holds
        // the webm header, so the whole slice stays decodable by ffmpeg.
        const partialBlob = new Blob(recorderState.chunks.slice(),
            { type: (recorderState.mediaRecorder &&
                      recorderState.mediaRecorder.mimeType) || 'video/webm' });
        const data = await requestLiveTranscription(partialBlob);
        recorderState.liveLastSize = totalBytes;
        if (data.text) {
            updateLiveTranscriptText(data.text);
            setLiveTranscriptStatus('Live', 'listening');
            updateMicLevel(data.rms, data.quality_label, false);
        } else {
            // No speech yet (student thinking / pausing) - keep listening.
            // If the backend reports a silent mic, surface a clear warning
            // instead of just "Listening..." so the student knows to check
            // their microphone and speak closer.
            updateMicLevel(data.rms, data.quality_label, !!data.silent);
            if (data.silent) {
                setLiveTranscriptStatus('No sound - switch mic device', 'processing');
            } else {
                setLiveTranscriptStatus('Listening...', 'listening');
            }
        }
    } catch (err) {
        // Preview-only failure - never blocks recording or submission
        console.warn('Live transcription preview failed:', err);
        setLiveTranscriptStatus('Listening...', 'listening');
    } finally {
        recorderState.liveBusy = false;
    }
}

async function finalizeLiveTranscript() {
    stopLiveTranscription();
    if (!recorderState.blob || !recorderState.blob.size) return;
    setLiveTranscriptStatus('Finishing transcription...', 'processing');
    try {
        const data = await requestLiveTranscription(
            recorderState.blob,
            'final-preview.' +
                ((recorderState.blob.type.includes('mp4')) ? 'mp4' : 'webm'));
        if (data.text) {
            updateLiveTranscriptText(data.text);
            setLiveTranscriptStatus('Transcription complete', 'complete');
        } else if (data.hint) {
            setLiveTranscriptStatus('No speech detected', '');
        } else {
            setLiveTranscriptStatus('Transcription unavailable', '');
        }
    } catch (err) {
        console.warn('Final live transcription failed:', err);
        setLiveTranscriptStatus('Transcription unavailable', '');
    }
}

function discardRecordedBlob() {
    if (recordingPlayback.src) URL.revokeObjectURL(recordingPlayback.src);
    recorderState.blob = null;
    recorderState.chunks = [];
    recordingPlayback.removeAttribute('src');
}

async function requestRecorderFullscreen() {
    if (!document.fullscreenElement && document.documentElement.requestFullscreen) {
        try {
            await document.documentElement.requestFullscreen();
        } catch (err) {
            console.warn('Recorder fullscreen request was blocked or unavailable:', err);
        }
    }
}

async function startRecording() {
    if (!recorderState.stream) { await enableCameraAndMic(); }
    if (!recorderState.stream) return;

    try {
        await requestRecorderFullscreen();
        discardRecordedBlob();
        const mimeType = pickRecorderMimeType();
        recorderState.mediaRecorder = mimeType
            ? new MediaRecorder(recorderState.stream, { mimeType })
            : new MediaRecorder(recorderState.stream);

        recorderState.mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) recorderState.chunks.push(event.data);
        };
        recorderState.mediaRecorder.onstop = () => {
            recorderState.blob = new Blob(recorderState.chunks,
                { type: recorderState.mediaRecorder.mimeType || 'video/webm' });
            recorderState.chunks = [];
            recordingPlayback.src = URL.createObjectURL(recorderState.blob);
            recorderState.state = 'stopped';       // preview state - explicit submit only
            finalizeLiveTranscript();              // final Whisper pass on the full take
            updateRecorderButtons();
        };

        recorderState.mediaRecorder.start(1000);   // 1s chunks for smooth pause
        recorderState.state = 'recording';
        resetTimer();
        recorderState.timerId = setInterval(tickTimer, 1000);

        // Start the live Whisper preview of the spoken answer
        stopLiveTranscription();
        resetLiveTranscript();
        setLiveTranscriptStatus('Listening...', 'listening');
        recorderState.liveLastSize = 0;
        recorderState.liveTimerId = setInterval(
            transcribeLiveTick, LIVE_TRANSCRIBE_INTERVAL_MS);

        updateRecorderButtons();
    } catch (err) {
        showRecorderMessage(
            'Recording could not be started with the current camera settings. Please re-enable the camera and try again.',
            'error');
        updateRecorderButtons();
    }
}

function togglePauseRecording() {
    const mr = recorderState.mediaRecorder;
    if (!mr) return;
    if (recorderState.state === 'recording' && mr.state === 'recording') {
        mr.pause();
        clearInterval(recorderState.timerId);
        recorderState.timerId = null;
        stopLiveTranscription('Paused', '');
        recorderState.state = 'paused';
    } else if (recorderState.state === 'paused' && mr.state === 'paused') {
        mr.resume();
        recorderState.timerId = setInterval(tickTimer, 1000);
        setLiveTranscriptStatus('Listening...', 'listening');
        recorderState.liveTimerId = setInterval(
            transcribeLiveTick, LIVE_TRANSCRIBE_INTERVAL_MS);
        recorderState.state = 'recording';
    }
    updateRecorderButtons();
}

function stopRecording() {
    const mr = recorderState.mediaRecorder;
    if (!mr || mr.state === 'inactive') return;
    clearInterval(recorderState.timerId);
    recorderState.timerId = null;
    stopLiveTranscription();   // no more periodic passes; onstop does a final one
    mr.stop();   // onstop builds the blob and switches to preview
}

function reRecord() {
    discardRecordedBlob();
    recorderState.state = 'ready';
    resetTimer();
    stopLiveTranscription();
    resetLiveTranscript();
    showRecorderMessage('');
    updateRecorderButtons();
}

function prepareRecorderForQuestion() {
    // Reset per-question recording state but keep an existing stream so the
    // browser does not re-prompt for permission every question.
    discardRecordedBlob();
    recorderState.mediaRecorder = null;
    recorderState.state = recorderState.stream ? 'ready' : 'idle';
    resetTimer();
    stopLiveTranscription();
    resetLiveTranscript();
    showRecorderMessage('');
    if (!recorderState.stream) {
        enableCameraAndMic();
    } else {
        updateDeviceStatusPills();
    }
    updateRecorderButtons();
    enumerateAudioDevices();
}

function shutdownRecorder() {
    clearInterval(recorderState.timerId);
    recorderState.timerId = null;
    stopAudioMeter();
    if (recorderState.mediaRecorder && recorderState.mediaRecorder.state !== 'inactive') {
        try { recorderState.mediaRecorder.stop(); } catch (err) { /* ignore */ }
    }
    if (recorderState.stream) {
        recorderState.stream.getTracks().forEach(track => track.stop());
        recorderState.stream = null;
    }
    cameraPreview.srcObject = null;
    discardRecordedBlob();
    recorderState.state = 'idle';
    resetTimer();
    stopLiveTranscription();
    resetLiveTranscript();
    updateDeviceStatusPills();
    updateRecorderButtons();
}

async function submitRecording() {
    const question = quizQuestions[currentQuestionIndex];
    if (!recorderState.blob || recorderState.state !== 'stopped') {
        showRecorderMessage('Please record your explanation before submitting.', 'warning');
        return;
    }

    recorderState.state = 'submitting';
    updateRecorderButtons();
    showRecorderMessage('Uploading your recording and generating the Whisper transcript...', 'info');

    const formData = new FormData();
    const ext = (recorderState.blob.type.includes('mp4')) ? 'mp4' : 'webm';
    formData.append('video', recorderState.blob, `explanation.${ext}`);

    isProctoringPaused = true;
    try {
        const response = await fetch('/api/recording/upload', {
            method: 'POST',
            body: formData,
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || 'Upload failed.');
        }

        window.__quizLastRecording = data;
        currentAnswer = data.transcript;

        // Persist the transcribed answer record
        storeAnswer(quizTopic, question, data.transcript);

        // Show the Whisper-generated transcript (the student never types it)
        transcribedAnswer.textContent = data.transcript;
        answerDisplay.classList.remove('hidden');

        // Sync the "Your Answer" panel with the EXACT transcript that was
        // stored and sent to the verification/AUVA pipeline
        updateLiveTranscriptText(data.transcript);
        setLiveTranscriptStatus('Final transcript (sent for verification)', 'complete');

        showRecorderMessage(
            `Transcript generated (${(data.language || 'unknown language').toUpperCase()}). Verifying your understanding...`,
            'success');

        await verifyQuizAnswer(quizTopic, question);

        showRecorderMessage(
            'Verification complete. Review your feedback below, then continue to the next question.',
            'success');
        nextQuestionBtn.classList.remove('hidden');
        nextQuestionBtn.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch (error) {
        console.error('Recording submission error:', error);
        showRecorderMessage(error.message ||
            "We couldn't process your recording right now. Please try again.",
            'error');
        recorderState.state = 'stopped';
    } finally {
        isProctoringPaused = false;
        updateRecorderButtons();
    }
}

enableMediaBtn.addEventListener('click', enableCameraAndMic);
startRecordingBtn.addEventListener('click', startRecording);
pauseRecordingBtn.addEventListener('click', togglePauseRecording);
stopRecordingBtn.addEventListener('click', stopRecording);
rerecordBtn.addEventListener('click', reRecord);
submitAnswerBtn.addEventListener('click', submitRecording);

// Let the student switch microphone device. If recording is active, ask them
// to stop first so the switch is clean.
micDeviceSelect.addEventListener('change', async () => {
    const selected = micDeviceSelect.value || '';
    if (recorderState.state === 'recording' || recorderState.state === 'paused') {
        alert('Please stop recording before switching the microphone device.');
        micDeviceSelect.value = recorderState.audioDeviceId || '';
        return;
    }
    recorderState.audioDeviceId = selected;
    if (!recorderState.stream) {
        await enableCameraAndMic();
    } else if (selected !== '') {
        // Re-open the stream with the newly selected microphone device.
        shutdownRecorder();
        await enableCameraAndMic();
    } else {
        // Back to the browser default mic.
        shutdownRecorder();
        await enableCameraAndMic();
    }
});

// Next question
nextQuestionBtn.addEventListener('click', () => {
    currentQuestionIndex++;
    displayQuestion();
});

generateQuizBtn.addEventListener('click', async () => {
    if (!searchedTopic) {
        alert('Please search a topic first to generate a quiz');
        return;
    }
    
    const topic = searchedTopic;
    const numQuestions = 5; // Always 5 questions
    
    loadingOverlay.classList.remove('hidden');
    generateQuizBtn.disabled = true;
    
    try {
        const response = await fetch('/api/quiz', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                topic: topic,
                num_questions: numQuestions
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        quizTopic = topic;
        quizQuestions = parseQuizQuestions(data.quiz);
        
        if (quizQuestions.length === 0) {
            throw new Error('No questions were generated');
        }
        
        quizContent.innerHTML = `<p>Quiz generated with <strong>${quizQuestions.length}</strong> questions about <strong>${topic}</strong>.</p><p>Click "Start Quiz" to begin answering.</p>`;
        quizResult.classList.remove('hidden');
        document.querySelector('.quiz-form').style.display = 'none';
        startQuizBtn.classList.remove('hidden');
        
    } catch (error) {
        alert('Error generating quiz: ' + error.message);
        console.error('Error:', error);
    } finally {
        loadingOverlay.classList.add('hidden');
        generateQuizBtn.disabled = false;
    }
});

newQuizBtn.addEventListener('click', () => {
    quizResult.classList.add('hidden');
    document.querySelector('.quiz-form').style.display = 'block';
    quizQuestions = [];
    currentQuestionIndex = 0;
    isQuizActive = false;
    resetQuizVerificationUI();
});

// ============================================
// Quiz Answer -> Understanding Verification (AUVA)
// ============================================

function resetQuizVerificationUI() {
    quizVerificationResults = [];
    if (answerFeedbackEl) {
        answerFeedbackEl.classList.add('hidden');
        answerFeedbackEl.innerHTML = '';
    }
    if (quizSummaryEl) {
        quizSummaryEl.classList.add('hidden');
        quizSummaryEl.innerHTML = '';
    }
}

async function verifyQuizAnswer(topic, question) {
    if (!answerFeedbackEl) return;
    
    // The Whisper transcript of the submitted recording is the primary
    // textual input for verification.
    const recording = window.__quizLastRecording || null;
    const answer = currentAnswer ||
        (recording && recording.transcript) || '';
    
    answerFeedbackEl.classList.remove('hidden');
    answerFeedbackEl.innerHTML =
        '<div class="feedback-loading"><i class="fas fa-spinner fa-spin"></i> Analyzing your understanding of this answer...</div>';
    
    try {
        const response = await fetch('/api/verify-quiz-answer', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                topic: topic,
                question: question,
                answer: answer,
                recording_id: recording ? recording.recording_id : null
            })
        });
        const report = await response.json();
        if (report.error) throw new Error(report.error);
        
        const auva = report.auva || {};
        quizVerificationResults.push({
            question: question,
            answer: answer,
            auva: typeof auva.score === 'number' ? auva.score : null,
            confidence: auva.confidence,
            level: report.understanding_level || 'Unknown',
            report: report
        });
        
        renderAnswerFeedback(report);

        // Pre-fetch explanations for any detected misconceptions in the background immediately
        if (report.misconceptions && report.misconceptions.length > 0) {
            report.misconceptions.forEach(m => {
                prefetchMisconception({
                    topic: topic,
                    question: question,
                    expected_representation: report.expected_representation || '',
                    student_answer: answer,
                    misconception: m
                });
            });
        }
    } catch (error) {
        console.error('Quiz answer verification error:', error);
        answerFeedbackEl.innerHTML =
            '<div class="feedback-error"><i class="fas fa-exclamation-circle"></i> ' +
            'Verification unavailable for this answer: ' + escapeHtml(error.message) + '</div>';
    }
}
async function storeAnswer(topic, question, answer) {
    try {
        const response = await fetch('/api/store-answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                topic: topic,
                question: question,
                answer: answer,
                timestamp: new Date().toISOString()
            })
        });
        
        const data = await response.json();
        if (!data.success) {
            console.error('Failed to store answer:', data.error);
        }
    } catch (error) {
        console.error('Error storing answer:', error);
    }
}

function renderAnswerFeedback(r) {
    const auva = r.auva || {};
    const cs = auva.component_scores || {};
    const score = typeof auva.score === 'number' ? Math.round(auva.score) : null;
    const conf = typeof auva.confidence === 'number' ? Math.round(auva.confidence) : null;
    const gaps = r.knowledge_gaps || [];
    const misconceptions = r.misconceptions || [];
    // The student's spoken answer and the reference answer the AUVA algorithm
    // used to grade it (shown side-by-side so the comparison is visible).
    const studentAnswer = (r.transcript || '').trim();
    const referenceAnswer = (r.expected_representation || '').trim();
    
    window.__lastVerificationReport = r;
    
    const gapItems = gaps.map((g) =>
        `<li><strong>${escapeHtml(g.concept)}</strong>
         <span class="sev-badge sev-${g.severity}">${g.severity}</span>
         <span class="muted">${escapeHtml(g.reason)}</span></li>`).join('');
    const misItems = misconceptions.map((m, idx) =>
        `<li class="misconception-item">
            <div class="misconception-item-content">
                <div><strong>Stated:</strong> "${escapeHtml(m.statement)}" &rarr; <strong>${escapeHtml(m.expected_understanding)}</strong></div>
                ${m.explanation ? `<div class="muted" style="margin-top:0.25rem;">${escapeHtml(m.explanation)}</div>` : ''}
            </div>
            <button type="button" class="read-misconception-btn feedback-read-mis-btn" data-mis-index="${idx}">
                <i class="fas fa-book-open"></i> Read About Misconception
            </button>
        </li>`).join('');
    
    answerFeedbackEl.innerHTML = `
        <h4><i class="fas fa-clipboard-check"></i> Understanding Feedback</h4>
        <div class="feedback-header">
            <div class="feedback-score" style="color:${score !== null ? scoreColor(score) : 'var(--text-light)'}">${score !== null ? score : '--'}</div>
            <div class="feedback-meta">
                <span class="level-badge">${escapeHtml(r.understanding_level || 'Unknown')}</span>
                <span class="muted">Verification confidence: ${conf !== null ? conf : '--'}</span>
            </div>
        </div>
        <p class="feedback-explanation">${escapeHtml(r.overall_explanation || '')}</p>
        ${studentAnswer ? `<div class="feedback-answer-box">
            <h5><i class="fas fa-microphone"></i> Your answer</h5>
            <p>${escapeHtml(studentAnswer)}</p>
        </div>` : ''}
        ${referenceAnswer ? `<div class="feedback-reference-box">
            <h5><i class="fas fa-robot"></i> Given algorithm (reference answer used to grade)</h5>
            <p>${escapeHtml(referenceAnswer)}</p>
        </div>` : ''}
        ${misItems ? `<h5><i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i> Misconceptions detected</h5>
            <ul class="feedback-list">${misItems}</ul>` : ''}
        ${gapItems ? `<h5><i class="fas fa-map-signs" style="color:var(--warning)"></i> Knowledge gaps</h5>
            <ul class="feedback-list">${gapItems}</ul>` : ''}
        <details class="feedback-details">
            <summary>Component scores</summary>
            <ul class="feedback-components">
                <li>Semantic similarity: ${cs.semantic_similarity ?? '--'}</li>
                <li>Concept coverage: ${cs.concept_coverage ?? '--'}</li>
                <li>Attention: ${cs.attention ?? '--'} <em class="muted">(supplementary)</em></li>
                <li>Speech confidence: ${cs.speech_confidence ?? '--'} <em class="muted">(supplementary)</em></li>
                <li>Knowledge gap penalty: -${cs.knowledge_gap_penalty ?? '--'}</li>
                <li>Misconception penalty: -${cs.misconception_penalty ?? '--'}</li>
            </ul>
        </details>`;
}

function renderQuizSummary() {
    if (!quizSummaryEl) return;
    
    const verified = quizVerificationResults.filter((v) => v.auva !== null);
    if (!verified.length) {
        quizSummaryEl.classList.add('hidden');
        return;
    }
    
    const avg = verified.reduce((s, v) => s + v.auva, 0) / verified.length;
    const levelCounts = {};
    verified.forEach((v) => { levelCounts[v.level] = (levelCounts[v.level] || 0) + 1; });
    const levelText = Object.entries(levelCounts).map(([l, c]) => `${l} x${c}`).join(' · ');
    
    const rows = verified.map((v, i) => `
        <tr>
            <td>Q${quizVerificationResults.indexOf(v) + 1}</td>
            <td class="summary-q">${escapeHtml(v.question)}</td>
            <td style="color:${scoreColor(v.auva)};font-weight:600">${Math.round(v.auva)}</td>
            <td>${escapeHtml(v.level)}</td>
        </tr>`).join('');

    // Collect all misconceptions across the quiz questions
    const allMisconceptions = [];
    quizVerificationResults.forEach((v, qIdx) => {
        const misList = (v.report && v.report.misconceptions) || [];
        misList.forEach((m) => {
            allMisconceptions.push({
                questionNumber: qIdx + 1,
                question: v.question,
                expected_representation: (v.report && v.report.expected_representation) || '',
                student_answer: v.answer || '',
                topic: quizTopic,
                misconception: m
            });
        });
    });
    window.__summaryMisconceptions = allMisconceptions;

    // Automatically pre-warm all detected misconceptions in the background so "Read About Misconception" is instant
    if (allMisconceptions && allMisconceptions.length > 0) {
        allMisconceptions.forEach(item => prefetchMisconception(item));
    }
    
    quizSummaryEl.innerHTML = `
        <h4><i class="fas fa-chart-line"></i> Understanding Summary</h4>
        <div class="feedback-header">
            <div class="feedback-score" style="color:${scoreColor(avg)}">${Math.round(avg)}</div>
            <div class="feedback-meta">
                <span class="muted">Average AUVA across ${verified.length} answered question(s)</span>
                <span class="muted">${escapeHtml(levelText)}</span>
            </div>
        </div>
        <table class="summary-table">
            <thead><tr><th>#</th><th>Question</th><th>AUVA</th><th>Level</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
        ${allMisconceptions.length > 0 ? `
            <div class="summary-misconceptions" style="margin-top:1.5rem;">
                <h5 style="margin-bottom:0.6rem;"><i class="fas fa-exclamation-triangle" style="color:var(--danger)"></i> Misconceptions Detected in Your Quiz</h5>
                <p class="muted" style="margin-bottom:0.75rem;font-size:0.875rem;">Click below to read a focused explanation for each detected misconception:</p>
                <ul class="feedback-list">
                    ${allMisconceptions.map((item, idx) => `
                        <li class="misconception-item">
                            <div class="misconception-item-content">
                                <div><span class="sev-badge sev-medium" style="margin-right:0.4rem;">Q${item.questionNumber}</span><strong>Stated:</strong> "${escapeHtml(item.misconception.statement)}" &rarr; <strong>${escapeHtml(item.misconception.expected_understanding)}</strong></div>
                                ${item.misconception.explanation ? `<div class="muted" style="margin-top:0.25rem;">${escapeHtml(item.misconception.explanation)}</div>` : ''}
                            </div>
                            <button type="button" class="read-misconception-btn summary-read-mis-btn" data-summary-idx="${idx}">
                                <i class="fas fa-book-open"></i> Read About Misconception
                            </button>
                        </li>
                    `).join('')}
                </ul>
            </div>
        ` : ''}`;
    quizSummaryEl.classList.remove('hidden');
}

// Misconception Explanation Logic, Pending Tracker & Cache
const _misconceptionExplanationCache = {};
const _misconceptionPendingFetches = {};

function prefetchMisconception(contextData) {
    if (!contextData) return;
    const mis = contextData.misconception || {};
    const cacheKey = (contextData.topic || quizTopic || '') + '::' + (mis.statement || '') + '::' + (mis.expected_understanding || '');
    if (_misconceptionExplanationCache[cacheKey] || _misconceptionPendingFetches[cacheKey]) {
        return;
    }

    _misconceptionPendingFetches[cacheKey] = fetch('/api/misconception/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            topic: contextData.topic || quizTopic || '',
            question: contextData.question || '',
            expected_representation: contextData.expected_representation || '',
            student_answer: contextData.student_answer || '',
            misconception: mis
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data && data.success && data.explanation) {
            _misconceptionExplanationCache[cacheKey] = data.explanation;
        }
        return data;
    })
    .catch(err => {
        console.warn('Background misconception prefetch error:', err);
    })
    .finally(() => {
        delete _misconceptionPendingFetches[cacheKey];
    });
}

async function openMisconceptionModal(contextData) {
    const modal = document.getElementById('misconception-modal');
    const modalBody = document.getElementById('misconception-modal-body');
    if (!modal || !modalBody) return;

    const mis = contextData.misconception || {};
    const cacheKey = (contextData.topic || quizTopic || '') + '::' + (mis.statement || '') + '::' + (mis.expected_understanding || '');

    modal.classList.remove('hidden');

    if (_misconceptionExplanationCache[cacheKey]) {
        renderMisconceptionContent(modalBody, mis, _misconceptionExplanationCache[cacheKey]);
        return;
    }

    modalBody.innerHTML = `
        <div class="misconception-loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Loading focused explanation...</p>
        </div>
    `;

    try {
        let explanation = null;
        if (_misconceptionPendingFetches[cacheKey]) {
            const data = await _misconceptionPendingFetches[cacheKey];
            if (data && data.success && data.explanation) {
                explanation = data.explanation;
            }
        }

        if (!explanation) {
            const response = await fetch('/api/misconception/explain', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'text/event-stream'
                },
                body: JSON.stringify({
                    topic: contextData.topic || quizTopic || '',
                    question: contextData.question || '',
                    expected_representation: contextData.expected_representation || '',
                    student_answer: contextData.student_answer || '',
                    misconception: mis,
                    stream: true
                })
            });

            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('text/event-stream')) {
                // Token streaming path: display tokens progressively as they arrive
                modalBody.innerHTML = `
                    <div style="background:#f1f5f9; padding:0.9rem 1.1rem; border-radius:8px; margin-bottom:1.25rem; font-size:0.92rem; border-left:4px solid var(--danger);">
                        <div><strong>What was stated:</strong> "${escapeHtml(mis.statement || '')}"</div>
                        <div style="margin-top:0.3rem;"><strong>Expected understanding:</strong> ${escapeHtml(mis.expected_understanding || '')}</div>
                    </div>
                    <div id="misconception-stream-text" class="formatted-content" style="min-height:120px;"></div>
                `;
                const streamContainer = document.getElementById('misconception-stream-text');

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let accumulated = '';
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            try {
                                const parsed = JSON.parse(line.slice(6));
                                if (parsed.content) {
                                    accumulated += parsed.content;
                                    if (streamContainer) {
                                        streamContainer.innerHTML = formatMessageStream(accumulated);
                                    }
                                }
                                if (parsed.done && parsed.explanation) {
                                    explanation = parsed.explanation;
                                }
                            } catch (e) {}
                        }
                    }
                }
                if (!explanation) {
                    explanation = accumulated;
                }
            } else {
                const data = await response.json();
                if (!data.success) {
                    throw new Error(data.error || 'Failed to generate explanation');
                }
                explanation = data.explanation;
            }
        }

        _misconceptionExplanationCache[cacheKey] = explanation;
        renderMisconceptionContent(modalBody, mis, explanation);
    } catch (err) {
        console.error('Misconception explanation error:', err);
        modalBody.innerHTML = `
            <div class="feedback-error">
                <i class="fas fa-exclamation-circle"></i>
                Could not load explanation: ${escapeHtml(err.message)}
            </div>
        `;
    }
}

function renderMisconceptionContent(container, mis, rawMarkdown) {
    let html = `
        <div style="background:#f1f5f9; padding:0.9rem 1.1rem; border-radius:8px; margin-bottom:1.25rem; font-size:0.92rem; border-left:4px solid var(--danger);">
            <div><strong>What was stated:</strong> "${escapeHtml(mis.statement || '')}"</div>
            <div style="margin-top:0.3rem;"><strong>Expected understanding:</strong> ${escapeHtml(mis.expected_understanding || '')}</div>
        </div>
    `;

    const sectionConfigs = [
        { key: 'misunderstood', regex: /##\s*What You Misunderstood\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'What You Misunderstood', icon: 'fa-times-circle', cardClass: 'card-misunderstood' },
        { key: 'correct', regex: /##\s*Correct Concept\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'Correct Concept', icon: 'fa-check-circle', cardClass: 'card-correct' },
        { key: 'why', regex: /##\s*Why This Understanding Is Incorrect\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'Why This Understanding Is Incorrect', icon: 'fa-exclamation-circle', cardClass: 'card-why' },
        { key: 'simple', regex: /##\s*Simple Explanation\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'Simple Explanation', icon: 'fa-book-open', cardClass: 'card-simple' },
        { key: 'example', regex: /##\s*Example\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'Example', icon: 'fa-cube', cardClass: 'card-example' },
        { key: 'takeaway', regex: /##\s*Key Takeaway\s*\n([\s\S]*?)(?=\n##|$)/i, title: 'Key Takeaway', icon: 'fa-key', cardClass: 'card-takeaway' }
    ];

    let matchedCount = 0;
    let sectionsHtml = '';

    for (const sec of sectionConfigs) {
        const match = rawMarkdown.match(sec.regex);
        if (match && match[1].trim()) {
            matchedCount++;
            sectionsHtml += `
                <div class="misconception-card ${sec.cardClass}">
                    <h4><i class="fas ${sec.icon}"></i> ${sec.title}</h4>
                    <div>${formatMessage(match[1].trim())}</div>
                </div>
            `;
        }
    }

    if (matchedCount >= 3) {
        html += sectionsHtml;
    } else {
        html += `<div class="formatted-content">${formatMessage(rawMarkdown)}</div>`;
    }

    container.innerHTML = html;
}

function closeMisconceptionModal() {
    const modal = document.getElementById('misconception-modal');
    if (modal) {
        modal.classList.add('hidden');
    }
}

// Global click delegation for Read About Misconception and Modal close
document.addEventListener('click', (e) => {
    const btn = e.target.closest('.feedback-read-mis-btn');
    if (btn) {
        const idx = parseInt(btn.getAttribute('data-mis-index'), 10);
        const report = window.__lastVerificationReport;
        if (report && report.misconceptions && report.misconceptions[idx]) {
            openMisconceptionModal({
                topic: report.topic || quizTopic || '',
                question: report.question || (quizQuestions[currentQuestionIndex] || ''),
                expected_representation: report.expected_representation || '',
                student_answer: report.transcript || currentAnswer || '',
                misconception: report.misconceptions[idx]
            });
        }
        return;
    }

    const summaryBtn = e.target.closest('.summary-read-mis-btn');
    if (summaryBtn) {
        const idx = parseInt(summaryBtn.getAttribute('data-summary-idx'), 10);
        const items = window.__summaryMisconceptions || [];
        if (items[idx]) {
            openMisconceptionModal(items[idx]);
        }
        return;
    }

    if (e.target.closest('#close-misconception-modal-btn') || 
        e.target.closest('#done-misconception-btn') || 
        e.target.id === 'misconception-modal-backdrop') {
        closeMisconceptionModal();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeMisconceptionModal();
    }
});

// Study functionality
const explainBtn = document.getElementById('explain-btn');
const explainResult = document.getElementById('explain-result');
const explainContent = document.getElementById('explain-content');
const newExplainBtn = document.getElementById('new-explain-btn');

explainBtn.addEventListener('click', async () => {
    const topic = document.getElementById('explain-topic').value.trim();
    const level = document.getElementById('explain-level').value;
    
    if (!topic) {
        alert('Please enter a topic to study');
        return;
    }
    
    // Store the searched topic for quiz generation
    searchedTopic = topic;
    quizTopicDisplay.innerHTML = `Quiz will be generated for: <strong>${topic}</strong> (5 questions)`;
    
    loadingOverlay.classList.remove('hidden');
    explainBtn.disabled = true;
    
    try {
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                topic: topic,
                level: level
            })
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        explainContent.innerHTML = formatMessage(data.explanation);
        explainResult.classList.remove('hidden');
        document.querySelector('.study-form').style.display = 'none';
        
        // Update AI Tutor context and reset chat history for new topic
        currentStudyContext = {
            topic: topic,
            level: level,
            explanation: data.explanation,
            selectedConcept: '',
            studyKnowledge: data.study_knowledge || null
        };
        chatHistory = [];
        chatMessages.innerHTML = '';
        addMessage(`I'm ready to help you with **${topic}**! Ask me any doubt about this material or highlight any section to discuss it. 🎓`, 'bot');
        
    } catch (error) {
        alert('Error generating explanation: ' + error.message);
        console.error('Error:', error);
    } finally {
        loadingOverlay.classList.add('hidden');
        explainBtn.disabled = false;
    }
});

newExplainBtn.addEventListener('click', () => {
    explainResult.classList.add('hidden');
    document.querySelector('.study-form').style.display = 'block';
    document.getElementById('explain-topic').value = '';
    searchedTopic = '';
    quizTopicDisplay.innerHTML = 'Search a topic above to generate a quiz';

    // Reset AI Tutor context
    currentStudyContext = {
        topic: '',
        level: 'beginner',
        explanation: '',
        selectedConcept: ''
    };
    chatHistory = [];
    chatMessages.innerHTML = '';
    addMessage("Hello! I'm your AI tutor. Ask me anything! 🎓", 'bot');
});

// Track highlighted text inside Study Material for context-aware doubt solving
document.addEventListener('selectionchange', () => {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
    if (selectedText && explainContent && explainContent.contains(selection.anchorNode)) {
        currentStudyContext.selectedConcept = selectedText;
    }
});

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
}

// Initialize
chatInput.focus();

// ============================================
// Understanding Verification (AUVA) Section

// ============================================
// Shared UI helpers
// ============================================

function escapeHtml(text) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(String(text == null ? '' : text)));
    return div.innerHTML;
}

function scoreColor(score) {
    if (score >= 75) return 'var(--success)';
    if (score >= 60) return '#22c55e';
    if (score >= 40) return 'var(--warning)';
    return 'var(--danger)';
}
