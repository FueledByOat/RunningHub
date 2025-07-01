// runstrong_journal.js

let currentRoutine = null;
let routineExercises = [];
let allExercises = [];
let freestyleExercises = [];

// Load routines on page load
document.addEventListener('DOMContentLoaded', function () {
    loadRoutines();
    loadAllExercisesForFreestyle(); // New function call
    document.getElementById('workout-date').value = new Date().toISOString().split('T')[0];
    document.getElementById('freestyle-workout-date').value = new Date().toISOString().split('T')[0];
});


// New function to fetch all exercises
async function loadAllExercisesForFreestyle() {
    try {
        const response = await fetch('/strong/api/exercises');
        const result = await response.json();

        if (result.status === 'success') {
            // CORRECTED: Assign to the global 'allExercises' variable
            allExercises = result.data; 
            const select = document.getElementById('freestyle-exercise-select');
            select.innerHTML = '<option value="" disabled selected>Select an exercise</option>';
            // Now correctly references the global variable
            select.innerHTML += allExercises.map(ex => `<option value="${ex.id}">${ex.name}</option>`).join('');
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        console.error('Error loading exercises for freestyle logging:', error);
    }
}


// New function to show the freestyle form
function startFreestyleWorkout() {
    document.getElementById('routine-selection').style.display = 'none';
    document.getElementById('freestyle-form').style.display = 'block';
}

// New function to go back to routine selection
function cancelFreestyleWorkout() {
    document.getElementById('freestyle-form').style.display = 'none';
    document.getElementById('routine-selection').style.display = 'block';
    freestyleExercises = []; // Clear the session
    renderFreestyleExercises(); // Update display
}

// New function to add an exercise to the temporary freestyle session
function addFreestyleExercise() {
    const exerciseSelect = document.getElementById('freestyle-exercise-select');
    const exerciseId = parseInt(exerciseSelect.value);
    const selectedExercise = allExercises.find(ex => ex.id === exerciseId);

    if (!selectedExercise) return;

    const exerciseData = {
        exercise_id: selectedExercise.id,
        name: selectedExercise.name, // Store name for display
        actual_sets: parseInt(document.getElementById('freestyle-sets').value) || 0,
        actual_reps: parseInt(document.getElementById('freestyle-reps').value) || 0,
        actual_load_lbs: parseFloat(document.getElementById('freestyle-load').value) || 0,
    };

    freestyleExercises.push(exerciseData);
    renderFreestyleExercises();
}

// New function to display the list of exercises in the current freestyle session
function renderFreestyleExercises() {
    const container = document.getElementById('freestyle-exercises-container');
    if (freestyleExercises.length === 0) {
        container.innerHTML = '<p>No exercises added yet.</p>';
        return;
    }

    container.innerHTML = freestyleExercises.map((ex, index) => `
        <div class="exercise-entry">
            <div class="exercise-name">${ex.name}</div>
            <span>${ex.actual_sets} sets × ${ex.actual_reps} reps @ ${ex.actual_load_lbs} lbs</span>
            <button class="btn-danger-small" onclick="removeFreestyleExercise(${index})">Remove</button>
        </div>
    `).join('');
}

// New function to remove an exercise from the temporary list
function removeFreestyleExercise(index) {
    freestyleExercises.splice(index, 1);
    renderFreestyleExercises();
}

// New function to save the entire freestyle session
async function saveFreestyleSession() {
    const workoutDate = document.getElementById('freestyle-workout-date').value;
    if (freestyleExercises.length === 0) {
        alert('Please add at least one exercise to the session.');
        return;
    }

    try {
        // CORRECTED: Point to the correct freestyle endpoint
        const response = await fetch('/strong/api/workout-performance/freestyle', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            // CORRECTED: The payload for freestyle doesn't need a routine_id
            // It only needs the date and the exercises performed.
            body: JSON.stringify({
                workout_date: workoutDate,
                exercises: freestyleExercises // The backend expects this key
            })
        });
        const result = await response.json();
        if (response.ok && result.status === 'success') {
            alert('Freestyle workout saved successfully!');
            cancelFreestyleWorkout(); // Go back to selection screen
        } else {
            alert(`Error saving workout: ${result.message || 'Please try again.'}`);
        }
    } catch (error) {
        console.error('Error saving freestyle workout:', error);
        alert('Error saving workout. Please try again.');
    }
}

async function loadRoutines() {
    try {
        const response = await fetch('/strong/api/routines');
        const result = await response.json();

        if (result.status === 'success') {
            const routines = result.data;
            // CHANGE THIS LINE:
            displayRoutines(routines); // Use the correct function name 'displayRoutines'
        } else {
            throw new Error(result.message);
        }
    } catch (error) {
        console.error('Error loading routines:', error);
    }
}

function displayRoutines(routines) {
    const routineList = document.getElementById('routine-list');
    // ...
    // CORRECTED: The function name was mismatched in the original file
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
        const response = await fetch(`/strong/api/routines/${routineId}`); // Endpoint was also slightly off
        const result = await response.json();

        if (result.status === 'success') {
            const payload = result.data; // The object containing routine and exercises

            currentRoutine = payload.routine;
            routineExercises = payload.exercises;

            document.getElementById('routine-title').textContent = currentRoutine.name;
            document.getElementById('routine-selection').style.display = 'none';
            document.getElementById('workout-form').style.display = 'block';

            displayExerciseEntries();
        } else {
            throw new Error(result.message);
        }
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
        const response = await fetch('/strong/api/workout-performance', {
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