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

function createCountdownElements(container, id) {
    const unitNames = ['Days', 'Hours', 'Minutes', 'Seconds'];
    const units = unitNames.map(unit => `
      <div class="flip-unit">
        <div class="flip-number" id="${id}-${unit.toLowerCase()}">00</div>
        <div class="flip-unit-label">${unit}</div>
      </div>
    `).join('');

    container.innerHTML = `<div class="flip-clock">${units}</div>`;
}

function startCountdown(race, container, id) {
    let previous = {
        days: null,
        hours: null,
        minutes: null,
        seconds: null
    };

    function flipTo(id, newValue) {
        const el = document.getElementById(id);
        if (!el) return;

        el.classList.remove("flip");
        void el.offsetWidth; // force reflow
        el.classList.add("flip");
        el.textContent = newValue;
    }

    function update() {
        const now = new Date();
        const raceTime = new Date(race.date);
        const total = raceTime - now;

        if (total <= 0) {
            container.innerHTML = "<p>It's race day!</p>";
            return;
        }

        const days = String(Math.floor(total / (1000 * 60 * 60 * 24))).padStart(2, '0');
        const hours = String(Math.floor((total / (1000 * 60 * 60)) % 24)).padStart(2, '0');
        const minutes = String(Math.floor((total / (1000 * 60)) % 60)).padStart(2, '0');
        const seconds = String(Math.floor((total / 1000) % 60)).padStart(2, '0');

        if (previous.days !== days) {
            flipTo(`${id}-days`, days);
            previous.days = days;
        }
        if (previous.hours !== hours) {
            flipTo(`${id}-hours`, hours);
            previous.hours = hours;
        }
        if (previous.minutes !== minutes) {
            flipTo(`${id}-minutes`, minutes);
            previous.minutes = minutes;
        }
        if (previous.seconds !== seconds) {
            flipTo(`${id}-seconds`, seconds);
            previous.seconds = seconds;
        }
    }

    update();
    setInterval(update, 1000);
}

function createCountdownElements(container, idPrefix) {
    container.innerHTML = `
      <div class="flip-clock">
        <div class="flip-unit"><div id="${idPrefix}-days" class="flip-number" data-value="">--</div><div class="flip-unit-label">Days</div></div>
        <div class="flip-unit"><div id="${idPrefix}-hours" class="flip-number" data-value="">--</div><div class="flip-unit-label">Hours</div></div>
        <div class="flip-unit"><div id="${idPrefix}-minutes" class="flip-number" data-value="">--</div><div class="flip-unit-label">Minutes</div></div>
        <div class="flip-unit"><div id="${idPrefix}-seconds" class="flip-number" data-value="">--</div><div class="flip-unit-label">Seconds</div></div>
      </div>
    `;
}

const raceList = document.getElementById("race-list");

races.forEach((race, index) => {
    const id = `race${index}`;
    const wrapper = document.createElement("div");
    wrapper.className = "race-item";
    wrapper.innerHTML = `
  <div class="race-card">
    <div class="race-info">
      <h4>${race.name} – ${new Date(race.date).toDateString()}</h4>
      <p class="race-goal">
        🎯 <strong>Goal:</strong> ${race.goal}<br>
        🔗 <a href="${race.link}" target="_blank" rel="noopener">Race Info</a>
      </p>
    </div>
    <div class="race-countdown" id="${id}"></div>
  </div>
`;
    raceList.appendChild(wrapper);

    const clockContainer = document.getElementById(id);
    createCountdownElements(clockContainer, id);
    startCountdown(race, clockContainer, id);
});