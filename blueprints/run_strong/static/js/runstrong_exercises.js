// runstrong_exercises.js

async function loadExercises() {
    try {
        const response = await fetch('/strong/api/exercises');
        const result = await response.json(); // Get the full result object

        if (result.status !== 'success') {
            throw new Error(result.message);
        }

        const exercises = result.data; // The array is inside the 'data' property
        const container = document.getElementById('exerciseGrid');
        const searchInput = document.getElementById('searchInput');

        function formatList(value) {
            if (!value) return 'N/A';
            // Replace commas with a comma and a space for readability
            return value.replace(/,/g, ', ');
        }

        function createCard(ex) {
            return `
                <div class="exercise-card" data-name="${ex.name.toLowerCase()}">
                    <div class="exercise-title">${ex.name}</div>
                    <div class="exercise-type">${ex.compound_vs_isolation || 'N/A'} | ${ex.difficulty_rating || 'N/A'}</div>
                    <button class="expand-btn" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'block' ? 'none' : 'block';">
                        Show Details
                    </button>
                    <div class="exercise-details">
                        <strong>Primary Muscles:</strong> ${formatList(ex.primary_muscles)}<br>
                        <strong>Secondary:</strong> ${formatList(ex.secondary_muscles)}<br>
                        <strong>Equipment:</strong> ${formatList(ex.equipment_required)}<br>
                        <strong>Instructions:</strong> ${ex.instructions || 'No instructions provided.'}<br>
                        ${ex.video_url ? `<a href="${ex.video_url}" target="_blank" style="color: var(--rs-highlight);">Watch Video</a>` : ''}
                    </div>
                </div>
            `;
        }

        function render(exercises) {
            container.innerHTML = exercises.map(createCard).join('');
        }

        searchInput.addEventListener('input', () => {
            const value = searchInput.value.toLowerCase();
            const cards = Array.from(container.children);
            cards.forEach(card => {
                const name = card.getAttribute('data-name');
                card.style.display = name.includes(value) ? 'block' : 'none';
            });
        });

        render(exercises);
    } catch (error) {
        console.error('Error loading exercises:', error);
        document.getElementById('exerciseGrid').innerHTML = `<p class="error">Could not load exercise library.</p>`;
    }
}

window.onload = loadExercises;