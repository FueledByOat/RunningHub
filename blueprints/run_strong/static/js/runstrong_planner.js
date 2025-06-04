let currentExercises = [];
let selectedExercise = null;
let routineExercises = [];
let editingIndex = -1;

// Load exercises on page load
document.addEventListener('DOMContentLoaded', function () {
  loadExercises();
});

async function loadExercises() {
  try {
    const response = await fetch('/strong/api/exercises');
    currentExercises = await response.json();
    displayExercises(currentExercises);
  } catch (error) {
    console.error('Error loading exercises:', error);
    // Fallback for demo purposes
    currentExercises = [
      { id: 1, name: 'Push-ups', primary_muscles: 'Chest, Triceps' },
      { id: 2, name: 'Squats', primary_muscles: 'Quadriceps, Glutes' },
      { id: 3, name: 'Pull-ups', primary_muscles: 'Latissimus Dorsi, Biceps' }
    ];
    displayExercises(currentExercises);
  }
}

function displayExercises(exercises) {
  const exerciseList = document.getElementById('exercise-list');
  exerciseList.innerHTML = '';

  if (exercises.length === 0) {
    exerciseList.innerHTML = '<p class="no-exercises-message">No exercises found.</p>';
    return;
  }

  exercises.forEach(exercise => {
    const exerciseItem = document.createElement('div');
    exerciseItem.className = 'exercise-item';
    exerciseItem.innerHTML = `
                    <div>
                        <strong>${exercise.name}</strong>
                        <br>
                        <small>${exercise.primary_muscles || 'N/A'}</small>
                    </div>
                    <button class="btn-secondary" onclick="selectExercise(${exercise.id})">Select</button>
                `;
    exerciseList.appendChild(exerciseItem);
  });
}

function selectExercise(exerciseId) {
  selectedExercise = currentExercises.find(ex => ex.id === exerciseId);
  editingIndex = -1; // Reset editing state
  document.getElementById('exercise-details').style.display = 'grid';

  // Reset form values
  document.getElementById('exercise-sets').value = 3;
  document.getElementById('exercise-reps').value = 10;
  document.getElementById('exercise-load').value = 0;
  document.getElementById('exercise-notes').value = '';

  // Scroll to exercise details
  document.getElementById('exercise-details').scrollIntoView({ behavior: 'smooth' });
}

function addExerciseToRoutine() {
  if (!selectedExercise) return;

  const sets = parseInt(document.getElementById('exercise-sets').value) || 1;
  const reps = parseInt(document.getElementById('exercise-reps').value) || 1;
  const load = parseFloat(document.getElementById('exercise-load').value) || 0;
  const notes = document.getElementById('exercise-notes').value.trim();

  const exerciseData = {
    exercise: selectedExercise,
    sets: sets,
    reps: reps,
    load_lbs: load,
    notes: notes,
    order_index: editingIndex >= 0 ? editingIndex : routineExercises.length
  };

  if (editingIndex >= 0) {
    // Update existing exercise
    routineExercises[editingIndex] = exerciseData;
    editingIndex = -1;
  } else {
    // Add new exercise
    routineExercises.push(exerciseData);
  }

  updateRoutineDisplay();
  cancelExerciseSelection();
}

function editExercise(index) {
  const exerciseData = routineExercises[index];
  selectedExercise = exerciseData.exercise;
  editingIndex = index;

  // Populate form with existing values
  document.getElementById('exercise-sets').value = exerciseData.sets;
  document.getElementById('exercise-reps').value = exerciseData.reps;
  document.getElementById('exercise-load').value = exerciseData.load_lbs;
  document.getElementById('exercise-notes').value = exerciseData.notes;

  // Show the exercise details form
  document.getElementById('exercise-details').style.display = 'grid';
  document.getElementById('exercise-details').scrollIntoView({ behavior: 'smooth' });
}

function cancelExerciseSelection() {
  selectedExercise = null;
  editingIndex = -1;
  document.getElementById('exercise-details').style.display = 'none';
}

function updateRoutineDisplay() {
  const container = document.getElementById('routine-exercises');

  if (routineExercises.length === 0) {
    container.innerHTML = '<p class="no-exercises-message">No exercises added yet.</p>';
    return;
  }

  container.innerHTML = routineExercises.map((item, index) => `
                <div class="routine-exercise">
                    <div class="routine-exercise-content">
                        <strong>${item.exercise.name}</strong>
                        <div style="font-size: 0.9rem; color: rgba(255, 255, 255, 0.8); margin-top: 0.25rem;">
                            ${item.sets} sets × ${item.reps} reps @ ${item.load_lbs} lbs
                            ${item.notes ? `<br><em>${item.notes}</em>` : ''}
                        </div>
                    </div>
                    <div class="routine-exercise-actions">
                        <button class="btn-edit" onclick="editExercise(${index})" title="Edit exercise">Edit</button>
                        <button class="btn-danger" onclick="removeExercise(${index})" title="Remove exercise">Remove</button>
                    </div>
                </div>
            `).join('');
}

function removeExercise(index) {
  if (confirm('Are you sure you want to remove this exercise from the routine?')) {
    routineExercises.splice(index, 1);
    updateRoutineDisplay();

    // If we were editing this exercise, cancel the editing
    if (editingIndex === index) {
      cancelExerciseSelection();
    } else if (editingIndex > index) {
      // Adjust editing index if needed
      editingIndex--;
    }
  }
}

async function saveRoutine() {
  const routineName = document.getElementById('routine-name').value.trim();

  if (!routineName) {
    alert('Please enter a routine name.');
    document.getElementById('routine-name').focus();
    return;
  }

  if (routineExercises.length === 0) {
    alert('Please add at least one exercise to the routine.');
    return;
  }

  try {
    const response = await fetch('/strong/api/routines', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: routineName,
        exercises: routineExercises
      })
    });

    if (response.ok) {
      alert('Routine saved successfully!');
      // Reset form
      document.getElementById('routine-name').value = '';
      routineExercises = [];
      updateRoutineDisplay();
      cancelExerciseSelection();
    } else {
      const errorData = await response.json();
      alert(`Error saving routine: ${errorData.message || 'Please try again.'}`);
    }
  } catch (error) {
    console.error('Error saving routine:', error);
    alert('Error saving routine. Please check your connection and try again.');
  }
}

// Search functionality
document.getElementById('exercise-search').addEventListener('input', function (e) {
  const searchTerm = e.target.value.toLowerCase().trim();

  if (searchTerm === '') {
    displayExercises(currentExercises);
    return;
  }

  const filteredExercises = currentExercises.filter(exercise =>
    exercise.name.toLowerCase().includes(searchTerm) ||
    (exercise.primary_muscles && exercise.primary_muscles.toLowerCase().includes(searchTerm))
  );
  displayExercises(filteredExercises);
});

// Keyboard shortcuts
document.addEventListener('keydown', function (e) {
  // ESC key to cancel exercise selection
  if (e.key === 'Escape') {
    cancelExerciseSelection();
  }

  // Ctrl+S to save routine
  if (e.ctrlKey && e.key === 's') {
    e.preventDefault();
    saveRoutine();
  }
});