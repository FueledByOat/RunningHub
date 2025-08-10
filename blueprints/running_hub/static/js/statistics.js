document.addEventListener('DOMContentLoaded', function () {
    // Get data from backend

    const weeklyDistances = JSON.parse(document.getElementById("weeklyDistances").textContent || '[0, 0, 0, 0, 0, 0, 0]');
    const paceDates = JSON.parse(document.getElementById("paceDates").textContent || '["Jan", "Feb", "Mar", "Apr", "May"]');
    const paceValues = JSON.parse(document.getElementById("paceValues").textContent || '[5.2, 5.1, 5.3, 5.0, 4.9]');


    const distanceCtx = document.getElementById('distanceChart');
    if (distanceCtx) {
        const distanceChart = new Chart(distanceCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Distance (km)',
                    data: weeklyDistances,
                    backgroundColor: '#0c1559',
                    borderColor: '#0c1559',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Distance (km)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Day of Week'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    // Pace Chart
    const paceCtx = document.getElementById('paceChart');
    if (paceCtx) {
        const paceChart = new Chart(paceCtx.getContext('2d'), {
            type: 'line',
            data: {
                labels: paceDates,
                datasets: [{
                    label: 'Pace (min/km)',
                    data: paceValues,
                    fill: false,
                    borderColor: '#ffcc00',
                    backgroundColor: 'rgba(255, 204, 0, 0.2)',
                    borderWidth: 2,
                    tension: 0.1,
                    pointBackgroundColor: '#0c1559',
                    pointBorderColor: '#0c1559',
                    pointRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: false,
                        title: {
                            display: true,
                            text: 'Pace (min/km)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Date'
                        }
                    }
                }
            }
        });
    }
});