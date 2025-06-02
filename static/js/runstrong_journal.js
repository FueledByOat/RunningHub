let currentRoutine = null;
let routineExercises = [];

// Load routines on page load
document.addEventListener('DOMContentLoaded', function () {
    loadRoutines();
    // Set today's date as default
    document.getElementById('workout-date').value = new Date().toISOString().split('T')[0];
});

async function loadRoutines() {
    try {
        const response = await fetch('/api/routines');
        const routines = await response.json();
        displayRoutines(routines);
    } catch (error) {
        console.error('Error loading routines:', error);
    }
}

function displayRoutines(routines) {
    const routineList = document.getElementById('routine-list');

    if (routines.length === 0) {
        routineList.innerHTML = '<p style="color: rgba(255, 255, 255, 0.6);">No routines found. Create one in the Planner first.</p>';
        return;
    }

    routineList.innerHTML = routines.map(routine => `
                <div class="routine-card" onclick="selectRoutine(${routine.id})">
                    <h3>${routine.name}</h3>
                    <p>Created: ${new Date(routine.date_created).toLocaleDateString()}</p>
                    <p>${routine.exercise_count || 0} exercises</p>
                </div>
            `).join('');
}

async function selectRoutine(routineId) {
    try {
        const response = await fetch(`/api/routines/${routineId}/exercises`);
        const data = await response.json();

        currentRoutine = data.routine;
        routineExercises = data.exercises;

        document.getElementById('routine-title').textContent = currentRoutine.name;
        document.getElementById('routine-selection').style.display = 'none';
        document.getElementById('workout-form').style.display = 'block';

        displayExerciseEntries();
    } catch (error) {
        console.error('Error loading routine exercises:', error);
    }
}

function displayExerciseEntries() {
    const container = document.getElementById('exercises-container');

    container.innerHTML = routineExercises.map((exercise, index) => `
                <div class="exercise-entry">
                    <div class="exercise-header">
                        <div class="exercise-name">${exercise.name}</div>
                        <div class="planned-info">
                            Planned: ${exercise.sets} sets × ${exercise.reps} reps @ ${exercise.load_lbs} lbs
                        </div>
                    </div>
                    
                    <div class="status-selector">
                        <label style="margin-right: 1rem;">Status:</label>
                        <button type="button" class="status-btn active" onclick="setStatus(${index}, 'completed', this)">Completed</button>
                        <button type="button" class="status-btn" onclick="setStatus(${index}, 'partial', this)">Partial</button>
                        <button type="button" class="status-btn" onclick="setStatus(${index}, 'skipped', this)">Skipped</button>
                    </div>
                    
                    <div class="performance-grid">
                        <div class="form-group">
                            <label for="actual-sets-${index}">Actual Sets:</label>
                            <input type="number" id="actual-sets-${index}" min="0" value="${exercise.sets}">
                        </div>
                        <div class="form-group">
                            <label for="actual-reps-${index}">Actual Reps:</label>
                            <input type="number" id="actual-reps-${index}" min="0" value="${exercise.reps}">
                        </div>
                        <div class="form-group">
                            <label for="actual-load-${index}">Actual Load (lbs):</label>
                            <input type="number" id="actual-load-${index}" min="0" step="0.5" value="${exercise.load_lbs}">
                        </div>
                        <div class="form-group" style="grid-column: 1 / -1;">
                            <label for="notes-${index}">Notes:</label>
                            <textarea id="notes-${index}" placeholder="How did this exercise feel? Any observations?">${exercise.notes || ''}</textarea>
                        </div>
                    </div>
                </div>
            `).join('');
}

function setStatus(exerciseIndex, status, button) {
    // Remove active class from all status buttons for this exercise
    const exerciseEntry = button.closest('.exercise-entry');
    exerciseEntry.querySelectorAll('.status-btn').forEach(btn => btn.classList.remove('active'));

    // Add active class to clicked button
    button.classList.add('active');

    // Store status data
    routineExercises[exerciseIndex].status = status;
}

async function saveWorkout() {
    const workoutDate = document.getElementById('workout-date').value;

    if (!workoutDate) {
        alert('Please select a workout date.');
        return;
    }

    const performanceData = routineExercises.map((exercise, index) => ({
        routine_id: currentRoutine.id,
        exercise_id: exercise.exercise_id,
        workout_date: workoutDate,
        planned_sets: exercise.sets,
        actual_sets: parseInt(document.getElementById(`actual-sets-${index}`).value) || 0,
        planned_reps: exercise.reps,
        actual_reps: parseInt(document.getElementById(`actual-reps-${index}`).value) || 0,
        planned_load_lbs: exercise.load_lbs,
        actual_load_lbs: parseFloat(document.getElementById(`actual-load-${index}`).value) || 0,
        notes: document.getElementById(`notes-${index}`).value,
        completion_status: exercise.status || 'completed'
    }));

    try {
        const response = await fetch('/api/workout-performance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                routine_id: currentRoutine.id,
                workout_date: workoutDate,
                exercises: performanceData
            })
        });

        if (response.ok) {
            alert('Workout saved successfully!');
            goBack();
        } else {
            alert('Error saving workout. Please try again.');
        }
    } catch (error) {
        console.error('Error saving workout:', error);
        alert('Error saving workout. Please try again.');
    }
}

function goBack() {
    document.getElementById('workout-form').style.display = 'none';
    document.getElementById('routine-selection').style.display = 'block';
    currentRoutine = null;
    routineExercises = [];
}