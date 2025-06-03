const messageInput = document.getElementById('coachGMessageInput');
const sendButton = document.getElementById('coachGSendButton');
const chatMessages = document.getElementById('coachGChatMessages');
const typingIndicator = document.getElementById('coachGTypingIndicator');

// Auto-resize textarea
messageInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// Send message on Enter (but allow Shift+Enter for new lines)
messageInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

sendButton.addEventListener('click', sendMessage);

function sendMessage() {
    const message = messageInput.value.trim();
    const personality = document.getElementById('coachGPersonality').value;
    if (!message) return;

    // Add user message
    addMessage(message, 'coach_g_user');
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Show typing indicator
    showTypingIndicator();

    // Make API call to Coach G
    fetch('/api/coach-g/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            message: message,
            personality: personality
        })
    })
        .then(response => response.json())
        .then(data => {
            hideTypingIndicator();
            addCoachResponse(data.response);
        })
        .catch(error => {
            hideTypingIndicator();
            addCoachResponse("Sorry, I'm having trouble connecting right now. Please try again.");
            console.error('Error:', error);
        });
}

function sendQuickQuestion(question) {
    messageInput.value = question;
    sendMessage();
}

function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `coach_g_message ${sender}`;

    const avatar = document.createElement('div');
    avatar.className = 'coach_g_message_avatar';

    if (sender === 'coach_g_user') {
        avatar.innerHTML = '<i class="fas fa-user"></i>';
    } else {
        // For Coach G - you can replace this with an actual image
        avatar.innerHTML = '<img src="static/images/coach_g_profile_pic.png" alt="Coach G" class="coach_g_coach_image" onerror="this.parentElement.innerHTML=\'<i class=&quot;fas fa-user-tie&quot;></i>\'">';
    }

    const messageContent = document.createElement('div');
    messageContent.className = 'coach_g_message_content';
    messageContent.textContent = content;

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);

    // Remove welcome message if it exists
    const welcomeMessage = chatMessages.querySelector('.coach_g_welcome_message');
    if (welcomeMessage) {
        welcomeMessage.remove();
    }

    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addCoachResponse(response) {
    addMessage(response, 'coach_g_coach');
}

function showTypingIndicator() {
    typingIndicator.style.display = 'flex';
    sendButton.disabled = true;
}

function hideTypingIndicator() {
    typingIndicator.style.display = 'none';
    sendButton.disabled = false;
}

// Initialize
document.addEventListener('DOMContentLoaded', function () {
    messageInput.focus();
});