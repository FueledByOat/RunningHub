const races = [
    {
        name: "Raspberry Festival 1 Mile",
        date: "2025-07-26T10:00:00",
        goal: "Break 5 minutes!",
        link: "https://www.raspberryrun.com/"
    },
    {
        name: "Twin Cities 10 Mile",
        date: "2025-10-05T07:00:00",
        goal: "Break 60 minutes and practice negative splits",
        link: "https://www.tcmevents.org/alleventsandraces/medtronictwincitiesmarathonweekend/tc10mile"
    },
    {
        name: "ORRRC Turkey Trot",
        date: "2025-11-26T08:30:00",
        goal: "5:30 splits!",
        link: "https://www.miamisburgtrot.com/"
    },
        {
        name: "La Crosse Marathon",
        date: "2026-05-01T07:00:00",
        goal: "For Funsies",
        link: "https://lacrossemarathon.com/"
    },
    {
        name: "Fukuoka Marathon",
        date: "2026-11-09T09:00:00",
        goal: "Run a sub-2:44 marathon and stay smooth through 35km",
        link: "https://www.f-marathon.jp/en/"
    }
];

let currentIndex = 0;

function createRaceCard(race, index) {
  return `
    <div class="race-poster ${index === 0 ? 'active' : ''}" data-index="${index}">
      <h2 class="poster-title">${race.name}</h2>
      <p class="poster-goal">🎯 Goal: ${race.goal}</p>
      <div class="countdown" id="countdown-${index}">Loading...</div>
      <a class="poster-link" href="${race.link}" target="_blank">View Race Info</a>
    </div>`;
}

function renderCarousel() {
  const carousel = document.getElementById('raceCarousel');
  carousel.innerHTML = races.map(createRaceCard).join('');
  races.forEach((race, i) => startCountdown(race, `countdown-${i}`));
}

function startCountdown(race, elementId) {
  function update() {
    const now = new Date();
    const raceTime = new Date(race.date);
    const total = raceTime - now;

    if (total <= 0) {
      document.getElementById(elementId).innerText = "Race Day!";
      return;
    }

    const days = Math.floor(total / (1000 * 60 * 60 * 24));
    const hours = Math.floor((total / (1000 * 60 * 60)) % 24);
    const minutes = Math.floor((total / (1000 * 60)) % 60);
    const seconds = Math.floor((total / 1000) % 60);

    document.getElementById(elementId).innerText =
      `${days}d ${hours}h ${minutes}m ${seconds}s`;
  }

  update();
  setInterval(update, 1000);
}

function showCard(index) {
  const cards = document.querySelectorAll('.race-poster');
  cards.forEach(card => card.classList.remove('active'));
  cards[index].classList.add('active');
  currentIndex = index;
}

function nextCard() {
  const next = (currentIndex + 1) % races.length;
  showCard(next);
}

function prevCard() {
  const prev = (currentIndex - 1 + races.length) % races.length;
  showCard(prev);
}

document.addEventListener("DOMContentLoaded", renderCarousel);