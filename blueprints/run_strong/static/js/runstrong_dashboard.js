document.addEventListener('DOMContentLoaded', () => {
    const toggleContainer = document.getElementById('category-toggle');
    const views = document.querySelectorAll('[data-category-view]');

    toggleContainer.addEventListener('click', (e) => {
        if (e.target.tagName === 'BUTTON') {
            const category = e.target.dataset.category;

            // Update active button
            toggleContainer.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');

            // Show/hide relevant views
            views.forEach(view => {
                view.classList.toggle('hidden', view.dataset.categoryView !== category);
            });
        }
    });

    // Render workload charts
    {% for category, details in data.items() %}
    const ctx_{{ category }} = document.getElementById('chart-{{ category }}');
    if (ctx_{{ category }}) {
        new Chart(ctx_{{ category }}, {
            type: 'line',
            data: {
                labels: {{ details.seven_day_workload | map(attribute='day') | list | tojson }},
                datasets: [{
                    label: 'Workload (units)',
                    data: {{ details.seven_day_workload | map(attribute='workload') | list | tojson }},
                    backgroundColor: 'rgba(255, 78, 199, 0.2)',
                    borderColor: 'rgba(255, 78, 199, 1)',
                    borderWidth: 2,
                    pointRadius: 4,
                    tension: 0.3
                }]
            },
            options: {
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
    {% endfor %}
});