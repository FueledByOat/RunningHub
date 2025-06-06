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
                {day: 'Mon', intensity: 85, hasTraining: true, volume: 4250},
                {day: 'Tue', intensity: 0, hasTraining: false, volume: 0},
                {day: 'Wed', intensity: 70, hasTraining: true, volume: 3500},
                {day: 'Thu', intensity: 90, hasTraining: true, volume: 4500},
                {day: 'Fri', intensity: 0, hasTraining: false, volume: 0},
                {day: 'Sat', intensity: 60, hasTraining: true, volume: 3000},
                {day: 'Sun', intensity: 45, hasTraining: true, volume: 2250}
            ],
            fatigue_trend: [
                {day: 'Mon', fatigue: 45},
                {day: 'Tue', fatigue: 50},
                {day: 'Wed', fatigue: 65},
                {day: 'Thu', fatigue: 80},
                {day: 'Fri', fatigue: 75},
                {day: 'Sat', fatigue: 70},
                {day: 'Sun', fatigue: 72}
            ],
            active_filter: 'all',
            available_filters: ['All', 'Upper body', 'Lower body', 'Core']
        };

        // Utility functions
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

        // Filter functions
        function setMuscleGroupFilter(filter) {
            currentFilter = filter;
            
            // Update UI toggles
            document.querySelectorAll('.filter-toggle').forEach(btn => {
                btn.classList.remove('active');
            });
            document.querySelector(`[data-filter="${filter}"]`).classList.add('active');
            
            // Update filter status
            const statusText = filter === 'all' ? 'All Muscle Groups' : 
                              filter.charAt(0).toUpperCase() + filter.slice(1) + ' Only';
            document.getElementById('filter-status').textContent = statusText;
            
            // Refresh data with filter
            refreshDashboard();
        }

        function filterMuscleData(data, filter) {
            if (filter === 'all') return data;
            
            const filtered = {...data};
            filtered.muscle_fatigue = data.muscle_fatigue.filter(muscle => 
                muscle.broad_group && muscle.broad_group.toLowerCase() === filter.toLowerCase()
            );
            
            // Recalculate overall fatigue for filtered data
            if (filtered.muscle_fatigue.length > 0) {
                filtered.overall_fatigue = Math.round(
                    filtered.muscle_fatigue.reduce((sum, m) => sum + m.fatigue_level, 0) / 
                    filtered.muscle_fatigue.length
                );
            }
            
            return filtered;
        }

        // Chart update functions
        function updateTrainingIntensity(data) {
            const stressBarsContainer = document.getElementById('stress-bars');
            const dayLabelsContainer = document.getElementById('day-labels');

            stressBarsContainer.innerHTML = '';
            dayLabelsContainer.innerHTML = '';

            data.daily_training.forEach(day => {
                const bar = document.createElement('div');
                bar.className = day.hasTraining ? 'stress-bar' : 'stress-bar rest-day';
                bar.style.height = `${day.intensity}%`;
                bar.title = `${day.day}: ${day.hasTraining ? day.intensity + '% intensity' : 'Rest day'}`;
                stressBarsContainer.appendChild(bar);

                const label = document.createElement('div');
                label.textContent = day.day;
                dayLabelsContainer.appendChild(label);
            });
        }

        function updateFatigueTrend(data) {
            const svg = document.getElementById('fatigue-trend-svg');
            const width = 300;
            const height = 120;

            svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

            // Clear existing trend elements
            const existingTrend = svg.querySelectorAll('.trend-path, .trend-area, .trend-point, .trend-label');
            existingTrend.forEach(el => el.remove());

            const points = data.fatigue_trend.map((item, index) => ({
                x: (index / (data.fatigue_trend.length - 1)) * width,
                y: height - (item.fatigue / 100) * height,
                fatigue: item.fatigue,
                day: item.day
            }));

            // Create path string for line
            const pathData = points.map((point, index) =>
                `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`
            ).join(' ');

            // Create area path
            const areaData = `${pathData} L ${width} ${height} L 0 ${height} Z`;

            // Add area
            const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            area.setAttribute('d', areaData);
            area.setAttribute('class', 'trend-area');
            svg.appendChild(area);

            // Add line
            const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            path.setAttribute('d', pathData);
            path.setAttribute('class', 'trend-path');
            svg.appendChild(path);

            // Add points
            points.forEach(point => {
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', point.x);
                circle.setAttribute('cy', point.y);
                circle.setAttribute('class', 'trend-point');
                circle.innerHTML = `<title>${point.day}: ${point.fatigue}% fatigue</title>`;
                svg.appendChild(circle);
            });

            // Add day labels
            points.forEach(point => {
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', point.x);
                text.setAttribute('y', height - 5);
                text.setAttribute('class', 'chart-label');
                text.textContent = point.day;
                svg.appendChild(text);
            });
        }

        function updateFatigueData(data) {
            console.log('Updating dashboard with data:', data);

            // Update overall score
            document.getElementById('overall-score').textContent = data.overall_fatigue;
            
            // Update description based on filter
            const description = currentFilter === 'all' ? 
                'Current fatigue level based on recent training volume and recovery' :
                `Current ${currentFilter} fatigue level based on recent training`;
            document.getElementById('fatigue-description').textContent = description;

            // Update muscle group fatigue
            const muscleContainer = document.getElementById('muscle-fatigue-container');
            muscleContainer.innerHTML = '';

            if (data.muscle_fatigue.length === 0) {
                muscleContainer.innerHTML = '<p style="text-align: center; color: #999;">No data available for selected muscle group</p>';
                return;
            }

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

            // Update charts
            updateTrainingIntensity(data);
            updateFatigueTrend(data);

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
            console.log(`Refreshing dashboard with filter: ${currentFilter}`);
            
            const url = currentFilter === 'all' ? 
                '/strong/api/fatigue-data' : 
                `/strong/api/fatigue-data?muscle_group=${encodeURIComponent(currentFilter)}`;
                
            fetch(url)
                .then(r => r.json())
                .then(data => {
                    updateFatigueData(data);
                })
                .catch(error => {
                    console.error('Failed to fetch real data, using mock data:', error);
                    const filteredData = filterMuscleData(mockFatigueData, currentFilter);
                    updateFatigueData(filteredData);
                });
        }

        // Initialize dashboard on page load - NOW USES REAL DATA
        document.addEventListener('DOMContentLoaded', function () {
            // Load real data on page load instead of mock data
            refreshDashboard();

            // Auto-refresh every 30 seconds
            setInterval(refreshDashboard, 30000);
        });

        // Expose functions globally for testing
        window.updateFatigueData = updateFatigueData;
        window.refreshDashboard = refreshDashboard;
        window.setMuscleGroupFilter = setMuscleGroupFilter;