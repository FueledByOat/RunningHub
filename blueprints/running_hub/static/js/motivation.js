const profileData = JSON.parse(document.getElementById("profile-data").textContent);
const races = profileData.races;

function createRaceCard(race, index) {
    const imageUrl = race.imageUrl;

    return `<div class="race-poster" data-index="${index}">
                <div class="poster-image-container">
                    <img src="${imageUrl}" alt="${race.name}" class="race-image" />
                </div>
                <div class="poster-text-container">
                    <a href="${race.link}" target="_blank" rel="noopener noreferrer">
                    <h2 class="poster-title">${race.name}</h2>
                    </a>
                    <p class="poster-goal">${race.goal}</p>
                    <div class="countdown" id="countdown-${index}">Loading...</div>
                </div>
            </div>`;
}

function renderCarousel() {
    const carousel = document.getElementById('raceCarousel');
    if (races && carousel) {
        carousel.innerHTML = races.map(createRaceCard).join('');
        races.forEach((race, i) => startCountdown(race.date, `countdown-${i}`));
    }
}

function startCountdown(raceDate, elementId) {
    const countdownElement = document.getElementById(elementId);
    if (!countdownElement) return;

    const intervalId = setInterval(() => {
        const now = new Date();
        const raceTime = new Date(raceDate);
        const totalSeconds = (raceTime - now) / 1000;

        if (totalSeconds <= 0) {
            countdownElement.innerText = "RACE DAY!";
            clearInterval(intervalId);
            return;
        }

        const days = Math.floor(totalSeconds / 3600 / 24);
        const hours = Math.floor(totalSeconds / 3600) % 24;
        const minutes = Math.floor(totalSeconds / 60) % 60;
        const seconds = Math.floor(((raceTime - now) / 1000) % 60);

        countdownElement.innerText = `${days}D : ${String(hours).padStart(2, '0')}H : ${String(minutes).padStart(2, '0')}M : ${String(seconds).padStart(2, '0')}S`;
    }, 1000);
}

document.addEventListener("DOMContentLoaded", () => {
    renderCarousel();

    const carousel = document.getElementById('raceCarousel');
    const nextBtn = document.getElementById('nextBtn');
    const prevBtn = document.getElementById('prevBtn');

    const scrollCarousel = (direction) => {
        const poster = carousel.querySelector('.race-poster');
        if (poster) {
            const scrollAmount = poster.offsetWidth + 30; // 30px for left/right margins
            carousel.scrollBy({ left: scrollAmount * direction, behavior: 'smooth' });
        }
    };

    nextBtn.addEventListener('click', () => scrollCarousel(1));
    prevBtn.addEventListener('click', () => scrollCarousel(-1));

    const newMessageBtn = document.getElementById('newMessageBtn');
    const messagePersonality = document.getElementById('messagePersonality');
    const dailyMessageContainer = document.getElementById('dailyMessage');

    async function fetchDailyMessage() {
        const personality = messagePersonality.value;
        dailyMessageContainer.innerHTML = '<p>Generating your message...</p>';

        try {
            // The URL includes the '/hub' prefix from your blueprint
            const response = await fetch('/hub/api/daily_motivation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ personality: personality })
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            if (data.error) {
                dailyMessageContainer.innerHTML = `<p>${data.error}</p>`;
            } else {
                dailyMessageContainer.innerHTML = data.response;
            }

        } catch (error) {
            console.error('Error fetching daily message:', error);
            dailyMessageContainer.innerHTML = '<p>Could not generate a message. Please try again later.</p>';
        }
    }

    newMessageBtn.addEventListener('click', fetchDailyMessage);
    fetchDailyMessage();
});