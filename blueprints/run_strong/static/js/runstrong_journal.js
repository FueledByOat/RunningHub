    document.addEventListener('DOMContentLoaded', () => {
        const setsContainer = document.getElementById('sets-container');
        const addSetBtn = document.getElementById('add-set-btn');
        const workoutForm = document.getElementById('workout-form');
        const setRowTemplate = document.getElementById('set-row-template');
        const formStatus = document.getElementById('form-status');

        const addSetRow = () => {
            const newRow = setRowTemplate.content.cloneNode(true);
            setsContainer.appendChild(newRow);
        };

        addSetRow();

        addSetBtn.addEventListener('click', addSetRow);

        setsContainer.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-set-btn')) {
                e.target.closest('.set-row').remove();
            }
        });

        workoutForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            formStatus.textContent = 'Saving...';

            const setRows = setsContainer.querySelectorAll('.set-row');
            const setsData = [];

            setRows.forEach(row => {
                const exerciseId = row.querySelector('select[name="exercise_id"]').value;
                const weight = row.querySelector('input[name="weight"]').value;
                const reps = row.querySelector('input[name="reps"]').value;
                const rpe = row.querySelector('input[name="rpe"]').value;

                if (exerciseId && weight && reps) {
                    setsData.push({
                        exercise_id: parseInt(exerciseId),
                        weight: parseFloat(weight),
                        reps: parseInt(reps),
                        rpe: rpe ? parseFloat(rpe) : null
                    });
                }
            });

            if (setsData.length === 0) {
                formStatus.textContent = 'Error: Please add at least one complete set.';
                formStatus.style.color = 'red';
                return;
            }

            const workoutPayload = {
                session_date: document.getElementById('session_date').value,
                notes: document.getElementById('notes').value,
                sets: setsData
            };

            try {
                const response = await fetch("{{ url_for('run_strong.log_workout_entry') }}", {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(workoutPayload)
                });

                if (response.ok) {
                    formStatus.textContent = 'Workout saved successfully!';
                    formStatus.style.color = 'green';
                    setTimeout(() => window.location.reload(), 1500);
                } else {
                    const errorData = await response.json();
                    throw new Error(errorData.message || 'Failed to save workout.');
                }
            } catch (error) {
                formStatus.textContent = `Error: ${error.message}`;
                formStatus.style.color = 'red';
            }
        });
    });