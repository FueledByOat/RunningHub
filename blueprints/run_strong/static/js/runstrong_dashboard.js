// Global variables
let currentFilter = 'all';

// Enhanced mock data with broad muscle groups (kept as fallback)
const mockFatigueData = {
    overall_fatigue: 72,
    muscle_fatigue: [
        {
            muscle_group: 'Quadriceps',
            last_trained: '2025-06-03',
            volume_7day: 2400,
            volume_14day: 4200,
            recovery_score: 35,
            fatigue_level: 65,
            broad_group: 'Lower body'
        },
        {
            muscle_group: 'Glutes',
            last_trained: '2025-06-02',
            volume_7day: 1800,
            volume_14day: 3600,
            recovery_score: 15,
            fatigue_level: 85,
            broad_group: 'Lower body'
        },
        {
            muscle_group: 'Chest',
            last_trained: '2025-06-01',
            volume_7day: 2000,
            volume_14day: 3800,
            recovery_score: 45,
            fatigue_level: 55,
            broad_group: 'Upper body'
        },
        {
            muscle_group: 'Triceps',
            last_trained: '2025-06-04',
            volume_7day: 1200,
            volume_14day: 2400,
            recovery_score: 60,
            fatigue_level: 40,
            broad_group: 'Upper body'
        },
        {
            muscle_group: 'Hamstrings',
            last_trained: '2025-06-01',
            volume_7day: 1200,
            volume_14day: 2800,
            recovery_score: 65,
            fatigue_level: 35,
            broad_group: 'Lower body'
        },
        {
            muscle_group: 'Core',
            last_trained: '2025-06-03',
            volume_7day: 900,
            volume_14day: 1800,
            recovery_score: 30,
            fatigue_level: 70,
            broad_group: 'Core'
        }
    ],
    weekly_stress: [60, 80, 45, 90, 30, 75, 85],
    daily_training: [
        { day: 'Mon', intensity: 85, hasTraining: true, volume: 4250 },
        { day: 'Tue', intensity: 0, hasTraining: false, volume: 0 },
        { day: 'Wed', intensity: 70, hasTraining: true, volume: 3500 },
        { day: 'Thu', intensity: 90, hasTraining: true, volume: 4500 },
        { day: 'Fri', intensity: 0, hasTraining: false, volume: 0 },
        { day: 'Sat', intensity: 60, hasTraining: true, volume: 3000 },
        { day: 'Sun', intensity: 45, hasTraining: true, volume: 2250 }
    ],
    fatigue_trend: [
        { day: 'Mon', fatigue: 45 },
        { day: 'Tue', fatigue: 50 },
        { day: 'Wed', fatigue: 65 },
        { day: 'Thu', fatigue: 80 },
        { day: 'Fri', fatigue: 75 },
        { day: 'Sat', fatigue: 70 },
        { day: 'Sun', fatigue: 72 }
    ],
    active_filter: 'all',
    available_filters: ['All', 'Upper body', 'Lower body', 'Core']
};
// Utility Functions
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
    const diffTime = today - lastTrained;
    return Math.max(0, Math.ceil(diffTime / (1000 * 60 * 60 * 24)));
}

// Filtering and Data Handling
function setMuscleGroupFilter(filter) {
    currentFilter = filter;

    // Update filter buttons
    document.querySelectorAll('.filter-toggle').forEach(btn => {
        btn.classList.remove('active');
    });
    const activeBtn = document.querySelector(`[data-filter="${filter}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    // Update status text
    const status = filter === 'all' ? 'All Muscle Groups' :
        `${filter.charAt(0).toUpperCase() + filter.slice(1)} Only`;
    const filterStatusEl = document.getElementById('filter-status');
    if (filterStatusEl) filterStatusEl.textContent = status;

    refreshDashboard();
}

function filterMuscleData(data, filter) {
    if (filter === 'all') return data;

    const filtered = { ...data };
    filtered.muscle_fatigue = data.muscle_fatigue.filter(m =>
        m.broad_group?.toLowerCase() === filter.toLowerCase()
    );

    if (filtered.muscle_fatigue.length > 0) {
        const total = filtered.muscle_fatigue.reduce((sum, m) => sum + m.fatigue_level, 0);
        filtered.overall_fatigue = Math.round(total / filtered.muscle_fatigue.length);
    }

    return filtered;
}

// Visualization and DOM Manipulation
function updateTrainingIntensity(data) {
    const stressBars = document.getElementById('stress-bars');
    const dayLabels = document.getElementById('day-labels');
    if (!stressBars || !dayLabels) return;

    stressBars.innerHTML = '';
    dayLabels.innerHTML = '';

    data.daily_training.forEach(day => {
        const bar = document.createElement('div');
        bar.className = day.hasTraining ? 'stress-bar' : 'stress-bar rest-day';
        bar.style.height = `${day.intensity}%`;
        bar.title = `${day.day}: ${day.hasTraining ? `${day.intensity}% intensity` : 'Rest day'}`;
        stressBars.appendChild(bar);

        const label = document.createElement('div');
        label.textContent = day.day;
        dayLabels.appendChild(label);
    });
}

function updateFatigueTrend(data) {
    const svg = document.getElementById('fatigue-trend-svg');
    if (!svg || !data.fatigue_trend?.length) return;

    const width = 300, height = 120;
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.innerHTML = ''; // Clear all elements

    const points = data.fatigue_trend.map((item, index) => ({
        x: (index / (data.fatigue_trend.length - 1)) * width,
        y: height - (item.fatigue / 100) * height,
        fatigue: item.fatigue,
        day: item.day
    }));

    const pathData = points.map((pt, i) =>
        `${i === 0 ? 'M' : 'L'} ${pt.x},${pt.y}`
    ).join(' ');

    const areaData = `${pathData} L ${width},${height} L 0,${height} Z`;

    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('d', areaData);
    area.classList.add('trend-area');
    svg.appendChild(area);

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.classList.add('trend-path');
    svg.appendChild(path);

    points.forEach(pt => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', pt.x);
        circle.setAttribute('cy', pt.y);
        circle.setAttribute('r', 4);
        circle.classList.add('trend-point');

        const tooltip = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        tooltip.textContent = `${pt.day}: ${pt.fatigue}% fatigue`;
        circle.appendChild(tooltip);

        svg.appendChild(circle);
    });

    points.forEach(pt => {
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', pt.x);
        text.setAttribute('y', height - 5);
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('class', 'chart-label');
        text.textContent = pt.day;
        svg.appendChild(text);
    });
}

function updateFatigueData(data) {
    console.log('Updating dashboard with data:', data);

    document.getElementById('overall-score').textContent = data.overall_fatigue;
    document.getElementById('fatigue-description').textContent = currentFilter === 'all'
        ? 'Current fatigue level based on recent training volume and recovery'
        : `Current ${currentFilter} fatigue level based on recent training`;

    const container = document.getElementById('muscle-fatigue-container');
    if (!container) return;
    container.innerHTML = '';

    if (!data.muscle_fatigue.length) {
        container.innerHTML = `<p class="no-data">No data available for selected muscle group</p>`;
        return;
    }

    data.muscle_fatigue.forEach(muscle => {
        const fatigueClass = getFatigueClass(muscle.fatigue_level);
        const recovery = getRecoveryStatus(muscle.recovery_score);

        const div = document.createElement('div');
        div.className = 'muscle-group-item';
        div.innerHTML = `
            <span class="muscle-name">${muscle.muscle_group}</span>
            <div class="fatigue-indicator">
                <div class="fatigue-bar">
                    <div class="fatigue-fill ${fatigueClass}" style="width: ${muscle.fatigue_level}%"></div>
                </div>
                <span class="fatigue-score">${muscle.fatigue_level}</span>
                <span class="recovery-badge ${recovery.class}">${recovery.text}</span>
            </div>
        `;
        container.appendChild(div);
    });

    updateTrainingIntensity(data);
    updateFatigueTrend(data);

    const recoveryContainer = document.getElementById('recovery-time-container');
    if (recoveryContainer) {
        recoveryContainer.innerHTML = '';
        data.muscle_fatigue.slice(0, 8).forEach(muscle => {
            const days = calculateDaysSince(muscle.last_trained);
            const div = document.createElement('div');
            div.className = 'days-item';
            div.innerHTML = `<div class="days-number">${days}</div><div class="days-label">${muscle.muscle_group}</div>`;
            recoveryContainer.appendChild(div);
        });
    }
}

// Dashboard Refresh Logic
function refreshDashboard() {
    console.log(`Refreshing dashboard with filter: ${currentFilter}`);
    const url = currentFilter === 'all'
        ? '/strong/api/fatigue-data'
        : `/strong/api/fatigue-data?muscle_group=${encodeURIComponent(currentFilter)}`;

    fetch(url)
        .then(r => r.json())
        .then(data => updateFatigueData(data))
        .catch(err => {
            console.error('Error fetching data, falling back to mock:', err);
            const fallback = filterMuscleData(mockFatigueData, currentFilter);
            updateFatigueData(fallback);
        });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    refreshDashboard();
    setInterval(refreshDashboard, 30000);
});

// Expose for debugging/testing
window.setMuscleGroupFilter = setMuscleGroupFilter;
window.refreshDashboard = refreshDashboard;
window.updateFatigueData = updateFatigueData;