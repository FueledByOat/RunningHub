// Mock data that simulates your Python backend response
const mockFatigueData = {
    overall_fatigue: 72,
    muscle_fatigue: [
        {
            muscle_group: 'Quadriceps',
            last_trained: '2025-06-03',
            volume_7day: 2400,
            volume_14day: 4200,
            recovery_score: 35,
            fatigue_level: 65
        },
        {
            muscle_group: 'Glutes',
            last_trained: '2025-06-02',
            volume_7day: 1800,
            volume_14day: 3600,
            recovery_score: 15,
            fatigue_level: 85
        },
        {
            muscle_group: 'Hamstrings',
            last_trained: '2025-06-01',
            volume_7day: 1200,
            volume_14day: 2800,
            recovery_score: 65,
            fatigue_level: 35
        },
        {
            muscle_group: 'Calves',
            last_trained: '2025-05-31',
            volume_7day: 800,
            volume_14day: 1600,
            recovery_score: 45,
            fatigue_level: 55
        },
        {
            muscle_group: 'Core',
            last_trained: '2025-06-04',
            volume_7day: 900,
            volume_14day: 1800,
            recovery_score: 30,
            fatigue_level: 70
        }
    ],
    weekly_stress: [60, 80, 45, 90, 30, 75, 85]
};

function getFatigueClass(fatigueLevel) {
    if (fatigueLevel < 40) return 'fatigue-low';
    if (fatigueLevel < 70) return 'fatigue-moderate';
    return 'fatigue-high';
}

function getRecoveryStatus(recoveryScore) {
    if (recoveryScore > 60) return { class: 'recovery-fresh', text: 'Fresh' };
    if (recoveryScore > 30) return { class: 'recovery-ready', text: 'Ready' };
    return { class: 'recovery-fatigued', text: 'Fatigued' };
}

function calculateDaysSince(dateString) {
    const lastTrained = new Date(dateString);
    const today = new Date();
    const diffTime = Math.abs(today - lastTrained);
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
}

function updateFatigueData(data) {
    console.log('Updating dashboard with data:', data);

    // Update overall score
    document.getElementById('overall-score').textContent = data.overall_fatigue;

    // Update muscle group fatigue
    const muscleContainer = document.getElementById('muscle-fatigue-container');
    muscleContainer.innerHTML = '';

    data.muscle_fatigue.forEach(muscle => {
        const fatigueClass = getFatigueClass(muscle.fatigue_level);
        const recovery = getRecoveryStatus(muscle.recovery_score);

        const muscleItem = document.createElement('div');
        muscleItem.className = 'muscle-group-item';
        muscleItem.innerHTML = `
                    <span class="muscle-name">${muscle.muscle_group}</span>
                    <div class="fatigue-indicator">
                        <div class="fatigue-bar">
                            <div class="fatigue-fill ${fatigueClass}" style="width: ${muscle.fatigue_level}%"></div>
                        </div>
                        <span class="fatigue-score">${muscle.fatigue_level}</span>
                        <span class="recovery-badge ${recovery.class}">${recovery.text}</span>
                    </div>
                `;
        muscleContainer.appendChild(muscleItem);
    });

    // Update stress bars
    const stressBarsContainer = document.getElementById('stress-bars');
    stressBarsContainer.innerHTML = '';

    data.weekly_stress.forEach(stress => {
        const bar = document.createElement('div');
        bar.className = 'stress-bar';
        bar.style.height = `${stress}%`;
        stressBarsContainer.appendChild(bar);
    });

    // Update recovery time
    const recoveryContainer = document.getElementById('recovery-time-container');
    recoveryContainer.innerHTML = '';

    data.muscle_fatigue.slice(0, 8).forEach(muscle => {
        const daysSince = calculateDaysSince(muscle.last_trained);
        const recoveryItem = document.createElement('div');
        recoveryItem.className = 'days-item';
        recoveryItem.innerHTML = `
                    <div class="days-number">${daysSince}</div>
                    <div class="days-label">${muscle.muscle_group}</div>
                `;
        recoveryContainer.appendChild(recoveryItem);
    });
}

function refreshDashboard() {
    console.log('Refreshing dashboard...');
    // In a real app, this would fetch from your API:
    fetch('/strong/api/fatigue-data')
        .then(r => r.json())
        .then(updateFatigueData)
        .catch(error => {
            console.log('API call failed, using mock data');
            updateFatigueData(mockFatigueData);
        });
}

// For now, simulate API call with mock data
// setTimeout(() => {
//     // Add some randomness to simulate changing data
//     const simulatedData = JSON.parse(JSON.stringify(mockFatigueData));
//     simulatedData.muscle_fatigue.forEach(muscle => {
//         muscle.fatigue_level += Math.floor(Math.random() * 10 - 5); // ±5 variation
//         muscle.fatigue_level = Math.max(0, Math.min(100, muscle.fatigue_level));
//         muscle.recovery_score = 100 - muscle.fatigue_level;
//     });

//     updateFatigueData(simulatedData);
// }, 500);


// Initialize dashboard on page load
document.addEventListener('DOMContentLoaded', function () {
    updateFatigueData(mockFatigueData);

    // Auto-refresh every 30 seconds
    setInterval(refreshDashboard, 30000);
});

// For testing - expose functions globally
window.updateFatigueData = updateFatigueData;
window.refreshDashboard = refreshDashboard;