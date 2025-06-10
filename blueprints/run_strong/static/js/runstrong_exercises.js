async function loadExercises() {
    const response = await fetch('/strong/exercises');
    const data = await response.json();
    const container = document.getElementById('exerciseGrid');
    const searchInput = document.getElementById('searchInput');

    function createCard(ex) {
        return `
                    <div class="exercise-card" data-name="${ex.name.toLowerCase()}">
                        <div class="exercise-title">${ex.name}</div>
                        <div class="exercise-type">${ex.exercise_type} | ${ex.movement_pattern}</div>
                        <button class="expand-btn" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'block' ? 'none' : 'block';">
                            Show Details
                        </button>
                        <div class="exercise-details">
                            <strong>Primary Muscles:</strong> ${ex.primary_muscles}<br>
                            <strong>Secondary:</strong> ${ex.secondary_muscles}<br>
                            <strong>Difficulty:</strong> ${ex.difficulty_rating}<br>
                            <strong>Equipment:</strong> ${ex.equipment_required}<br>
                            <strong>Instructions:</strong> ${ex.instructions}<br>
                            <strong>Common Mistakes:</strong> ${ex.common_mistakes}<br>
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

    render(data.exercises);
}

window.onload = loadExercises;