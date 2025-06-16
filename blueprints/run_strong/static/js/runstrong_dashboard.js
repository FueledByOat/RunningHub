// runstrong_dashboard.js

// --- UTILITY FUNCTIONS ---
function getFatigueClass(score) {
    if (score < 50) return 'fatigue-low';
    else if (score < 75) return 'fatigue-moderate';
    else return 'fatigue-high';
}

function getRecoveryStatus(recoveryScore) {
    if (recoveryScore > 60) return { class: 'recovery-fresh', text: 'Fresh' };
    if (recoveryScore > 30) return { class: 'recovery-ready', text: 'Ready' };
    return { class: 'recovery-fatigued', text: 'Fatigued' };
}

// --- RENDER FUNCTIONS ---

/**
 * Main function to update all dashboard components with new data.
 * This is now the single entry point for rendering.
 */
function updateDashboardUI(data) {
    console.log('Updating dashboard with data:', data);

    renderOverallFatigue(data.overall_fatigue, data.recommendation);
    renderTrainingIntensity(data.daily_training || []);
    renderFatigueTrend(data.fatigue_trend || []);
    renderMuscleFatigueGroups(data.muscle_fatigue || []);
    renderInsights(data.muscle_fatigue || [], data.least_used_muscles || []);
}

function renderFilterButtons(filters = ['All', 'Upper body', 'Lower body', 'Core'], activeFilter = 'all') {
    const container = document.getElementById('filter-buttons-container');
    if (!container) return;
    container.innerHTML = filters.map(filter => `
        <button 
            class="filter-btn ${activeFilter.toLowerCase() === filter.toLowerCase() ? 'active' : ''}" 
            onclick="setMuscleGroupFilter('${filter.toLowerCase()}')">
            ${filter}
        </button>
    `).join('');
}

function renderOverallFatigue(score, recommendation) {
    const statusBlock = document.getElementById('fatigue-status-block');
    const scoreDisplay = document.getElementById('fatigue-score-display');
    const descriptionEl = document.getElementById('fatigue-description');
    const recommendationEl = document.getElementById('recommendation-container');

    // Guard clause to handle null or invalid data
    if (score === null || score === undefined || isNaN(score)) {
        if (statusBlock) statusBlock.className = 'fatigue-status-block'; // Reset to default color
        if (scoreDisplay) scoreDisplay.textContent = '--';
        if (descriptionEl) descriptionEl.textContent = 'No Data';
        if (recommendationEl) recommendationEl.textContent = "Not enough data to generate a recommendation.";
        return;
    }
    
    // Determine status text and fatigue class
    const fatigueClass = getFatigueClass(score); // Uses your existing utility function
    let statusText = 'High';
    if (score < 50) statusText = 'Fresh';
    else if (score < 75) statusText = 'Moderate';

    // Update the UI
    if (statusBlock) statusBlock.className = `fatigue-status-block ${fatigueClass}`;
    if (scoreDisplay) scoreDisplay.textContent = Math.round(score);
    if (descriptionEl) descriptionEl.textContent = statusText;
    if (recommendationEl) recommendationEl.textContent = recommendation || "No recommendation available.";
}

function renderMuscleFatigueGroups(muscleData) {
    const container = document.getElementById('muscle-fatigue-container');
    if (!container) return;
    container.innerHTML = '';

    const grouped = muscleData.reduce((acc, muscle) => {
        const group = muscle.broad_group || 'Other';
        if (!acc[group]) acc[group] = [];
        acc[group].push(muscle);
        return acc;
    }, {});

    ['Upper body', 'Lower body', 'Core', 'Other'].forEach(groupName => {
        if (grouped[groupName]) {
            const groupDiv = document.createElement('div');
            groupDiv.className = 'fatigue-group';

            const title = document.createElement('h4');
            title.className = 'fatigue-group-title';
            title.textContent = groupName;
            groupDiv.appendChild(title);

            grouped[groupName].forEach(muscle => {
                const fatigueClass = getFatigueClass(muscle.fatigue_level);
                const recovery = getRecoveryStatus(muscle.recovery_score);
                const itemDiv = document.createElement('div');
                itemDiv.className = 'muscle-group-item';
                itemDiv.innerHTML = `
                    <span class="muscle-name">${muscle.muscle_group}</span>
                    <div class="fatigue-indicator">
                        <div class="fatigue-bar">
                            <div class="fatigue-fill ${fatigueClass}" style="width: ${muscle.fatigue_level}%"></div>
                        </div>
                        <span class="fatigue-score">${muscle.fatigue_level}</span>
                        <span class="recovery-badge ${recovery.class}">${recovery.text}</span>
                    </div>`;
                groupDiv.appendChild(itemDiv);
            });
            container.appendChild(groupDiv);
        }
    });

    if (container.innerHTML === '') {
        container.innerHTML = `<p class="no-data">No data available for selected filter.</p>`;
    }
}

function renderInsights(muscleData, leastUsedMuscles) {
    const priorityContainer = document.getElementById('recovery-priority-container');
    const neglectedContainer = document.getElementById('neglected-muscles-container');

    // Render Recovery Priority (Top 3 most fatigued)
    if (priorityContainer) {
        const mostFatigued = [...muscleData].sort((a, b) => b.fatigue_level - a.fatigue_level).slice(0, 5);
        priorityContainer.innerHTML = mostFatigued.map(m => `
            <div class="insight-item">
                <span class="muscle-name">${m.muscle_group}</span>
                <span class="fatigue-value ${getFatigueClass(m.fatigue_level)}">${m.fatigue_level}% Fatigued</span>
            </div>
        `).join('');
    }

    // Render Neglected Muscles (from backend or calculated)
    if (neglectedContainer) {
        neglectedContainer.innerHTML = leastUsedMuscles.map(m => {
            const days = Math.round((new Date() - new Date(m.last_trained)) / (1000 * 60 * 60 * 24));
            return `
            <div class="insight-item">
                <span class="muscle-name">${m.muscle_group}</span>
                <span class="days-value">Rested ${days} days</span>
            </div>
            `
        }).join('');
    }
}

function renderTrainingIntensity(dailyTraining) {
    const stressBars = document.getElementById('stress-bars');
    const dayLabels = document.getElementById('day-labels');
    if (!stressBars || !dayLabels) return;

    stressBars.innerHTML = '';
    dayLabels.innerHTML = '';

    dailyTraining.forEach(day => {
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

function renderFatigueTrend(fatigueTrend) {
    const svg = document.getElementById('fatigue-trend-svg');
    if (!svg || !fatigueTrend?.length) return;

    // Get the actual container dimensions
    const container = svg.parentElement;
    const containerRect = container.getBoundingClientRect();
    const width = containerRect.width || 400; // fallback width
    const height = 140; // Keep your desired height

    const padding = { top: 10, right: 10, bottom: 25, left: 30 };
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Set SVG to fill container
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', height);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('preserveAspectRatio', 'none');
    svg.innerHTML = '';

    // Create main group for chart area
    const chartGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    chartGroup.setAttribute('transform', `translate(${padding.left}, ${padding.top})`);
    svg.appendChild(chartGroup);

    // Scale data points
    const maxFatigue = Math.max(...fatigueTrend.map(d => d.fatigue), 100);
    const points = fatigueTrend.map((item, index) => ({
        x: (index / (fatigueTrend.length - 1)) * chartWidth,
        y: chartHeight - (item.fatigue / maxFatigue) * chartHeight,
        fatigue: item.fatigue,
        day: item.day
    }));

    // Add horizontal grid lines and labels
    [0, 25, 50, 75, 100].forEach(value => {
        const y = chartHeight - (value / maxFatigue) * chartHeight;

        // Grid line
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', 0);
        line.setAttribute('x2', chartWidth);
        line.setAttribute('y1', y);
        line.setAttribute('y2', y);
        line.setAttribute('stroke', 'rgba(255,255,255,0.1)');
        line.setAttribute('stroke-width', '1');
        chartGroup.appendChild(line);

        // Y-axis label
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', -5);
        label.setAttribute('y', y + 3);
        label.setAttribute('text-anchor', 'end');
        label.setAttribute('font-size', '10');
        label.setAttribute('fill', 'rgba(255,255,255,0.6)');
        label.textContent = value;
        chartGroup.appendChild(label);
    });

    // Create path for trend line
    const pathData = points.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x},${pt.y}`).join(' ');
    const areaData = `${pathData} L ${chartWidth},${chartHeight} L 0,${chartHeight} Z`;

    // Add area fill
    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('d', areaData);
    area.classList.add('trend-area');
    chartGroup.appendChild(area);

    // Add trend line
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', pathData);
    path.classList.add('trend-path');
    chartGroup.appendChild(path);

    // Add data points with hover effects
    points.forEach((point, index) => {
        const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        circle.setAttribute('cx', point.x);
        circle.setAttribute('cy', point.y);
        circle.setAttribute('r', '4');
        circle.setAttribute('fill', 'var(--rs-highlight, #ff4ec7)');
        circle.setAttribute('stroke', '#fff');
        circle.setAttribute('stroke-width', '2');
        circle.style.cursor = 'pointer';

        // Add tooltip
        circle.addEventListener('mouseenter', () => {
            circle.setAttribute('r', '6');
        });
        circle.addEventListener('mouseleave', () => {
            circle.setAttribute('r', '4');
        });

        chartGroup.appendChild(circle);
    });

    // Add day labels on x-axis
    points.forEach((point, index) => {
        if (index % Math.ceil(points.length / 4) === 0) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', point.x);
            label.setAttribute('y', chartHeight + 15);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('font-size', '9');
            label.setAttribute('fill', 'rgba(255,255,255,0.6)');
            label.textContent = fatigueTrend[index].day;
            chartGroup.appendChild(label);
        }
    });

    // Add trend indicator
    const currentFatigue = points[points.length - 1]?.fatigue || 0;
    const previousFatigue = points[points.length - 2]?.fatigue || currentFatigue;
    const trend = currentFatigue - previousFatigue;

    const trendText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    trendText.setAttribute('x', chartWidth);
    trendText.setAttribute('y', 15);
    trendText.setAttribute('text-anchor', 'end');
    trendText.setAttribute('font-size', '11');
    trendText.setAttribute('font-weight', 'bold');

    if (trend > 5) {
        trendText.setAttribute('fill', '#dc3545');
        trendText.textContent = '↗ Rising';
    } else if (trend < -5) {
        trendText.setAttribute('fill', '#4ade80');
        trendText.textContent = '↘ Improving';
    } else {
        trendText.setAttribute('fill', '#ffc107');
        trendText.textContent = '→ Stable';
    }

    chartGroup.appendChild(trendText);
}


// --- API AND EVENT HANDLERS ---
let currentFilter = 'all';

function setMuscleGroupFilter(filter) {
    currentFilter = filter;
    fetchDashboardData();
}

function fetchDashboardData() {
    console.log(`Refreshing dashboard with filter: ${currentFilter}`);
    const url = `/strong/api/fatigue-data?muscle_group=${encodeURIComponent(currentFilter)}`;

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            renderFilterButtons(data.available_filters, currentFilter);
            updateDashboardUI(data);
        })
        .catch(err => {
            console.error('Error fetching dashboard data:', err);
            // You can display an error message to the user here
            document.getElementById('muscle-fatigue-column').innerHTML =
                `<div class="dashboard-card"><p class="error">Could not load dashboard data. Please try again later.</p></div>`;
        });
}

function triggerManualUpdate() {
    const btn = document.getElementById('update-btn');
    btn.disabled = true;
    btn.textContent = 'Updating...';

    fetch('/strong/api/update-fatigue')
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                console.log('Manual update triggered successfully.');
                fetchDashboardData(); // Refresh the dashboard with new data
            } else {
                throw new Error(data.error || 'Unknown update error');
            }
        })
        .catch(err => {
            console.error('Error triggering manual update:', err);
            alert('Failed to update data.');
        })
        .finally(() => {
            btn.disabled = false;
            btn.innerHTML = `<i class="fas fa-sync"></i> Update Now`;
        });
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardData();
});
window.addEventListener('resize', () => {
    // Redraw chart with new dimensions
    fetchDashboardData();
});