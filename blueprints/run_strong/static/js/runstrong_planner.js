// runstrong_planner.js

let currentExercises = [];
let selectedExercise = null;
let routineExercises = [];
let editingIndex = -1;
let isEditingRoutine = false;
let currentRoutineId = null;
let exerciseMaxLoads = {}; // Cache for max loads

// Load exercises on page load
document.addEventListener('DOMContentLoaded', function () {
  loadExercises();
  loadExistingRoutines();
  loadExerciseMaxLoads();
});

async function loadExercises() {
  try {
    const response = await fetch('/strong/api/exercises');
    const result = await response.json(); // Get the full response object

    if (result.status !== 'success') {
      throw new Error(result.message);
    }

    currentExercises = result.data; // Use the .data property
    displayExercises(currentExercises);
  } catch (error) {
    console.error('Error loading exercises:', error);
  }
}

async function loadExistingRoutines() {
  try {
    const response = await fetch('/strong/api/routines');
    const routines = await response.json();

    // Transform the data to match expected format
    const transformedRoutines = routines.data.map(routine => ({
      id: routine.id || routine[0],
      name: routine.name || routine[1]
    }));

    displayRoutineSelector(transformedRoutines);
  } catch (error) {
    console.error('Error loading routines:', error);
  }
}

async function loadExerciseMaxLoads() {
  try {
    const response = await fetch('/strong/api/exercise-max-loads');
    exerciseMaxLoads = await response.json();
  } catch (error) {
    console.error('Error loading max loads:', error);
    exerciseMaxLoads = {};
  }
}

function displayRoutineSelector(routines) {
  const existingSelector = document.getElementById('routine-selector-container');

  if (routines.length > 0) {
    existingSelector.innerHTML = `
      <div class="form-group">
        <label for="existing-routine">Or Edit Existing Routine:</label>
        <select id="existing-routine" onchange="loadRoutineForEditing(this.value)">
          <option value="">Select a routine to edit...</option>
          ${routines.map(routine =>
      `<option value="${routine.id}">${routine.name}</option>`
    ).join('')}
        </select>
      </div>
    `;
    existingSelector.style.display = 'block';
  } else {
    existingSelector.style.display = 'none';
  }
}

async function loadRoutineForEditing(routineId) {
  if (!routineId) {
    resetForm();
    return;
  }

  try {
    const response = await fetch(`/strong/api/routines/${routineId}`);
    const data = await response.json();

    if (data.routine && data.exercises) {
      isEditingRoutine = true;
      currentRoutineId = routineId;

      // Populate routine name
      document.getElementById('routine-name').value = data.routine.name || data.routine[1];

      // Load routine exercises - handle different data formats
      routineExercises = data.exercises.map(ex => ({
        exercise: {
          id: ex.exercise_id || ex.id,
          name: ex.exercise_name || ex.name,
          primary_muscles: ex.primary_muscles || 'N/A'
        },
        sets: ex.sets || 3,
        reps: ex.reps || 10,
        load_lbs: ex.load_lbs || 0,
        notes: ex.notes || '',
        order_index: ex.order_index || 0
      }));

      // Sort by order_index
      routineExercises.sort((a, b) => a.order_index - b.order_index);

      updateRoutineDisplay();
      updateSaveButtonText();
    }
  } catch (error) {
    console.error('Error loading routine for editing:', error);
    alert('Error loading routine. Please try again.');
  }
}

function resetForm() {
  isEditingRoutine = false;
  currentRoutineId = null;
  routineExercises = [];
  document.getElementById('routine-name').value = '';

  const existingRoutineSelect = document.getElementById('existing-routine');
  if (existingRoutineSelect) {
    existingRoutineSelect.value = '';
  }

  updateRoutineDisplay();
  updateSaveButtonText();
  cancelExerciseSelection();
}

function updateSaveButtonText() {
  const saveButton = document.querySelector('.btn-primary');
  if (saveButton) {
    saveButton.textContent = isEditingRoutine ? 'Update Routine' : 'Save Routine';
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

  // Pre-populate load with max load if available
  const maxLoad = exerciseMaxLoads[exerciseId] || 0;
  document.getElementById('exercise-load').value = maxLoad;

  // Update placeholder text to show this is auto-populated
  const loadInput = document.getElementById('exercise-load');
  if (maxLoad > 0) {
    loadInput.title = `Auto-populated with your previous maximum load: ${maxLoad} lbs`;
  } else {
    loadInput.title = "No previous maximum load found";
  }

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

    // Update order indices
    routineExercises.forEach((item, idx) => {
      item.order_index = idx;
    });

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

  // Update order indices before saving
  routineExercises.forEach((item, index) => {
    item.order_index = index;
  });

  try {
    const url = isEditingRoutine
      ? `/strong/api/routines/${currentRoutineId}`
      : '/strong/api/routines';

    const method = isEditingRoutine ? 'PUT' : 'POST';

    const payloadExercises = routineExercises.map(item => ({
      exercise_id: item.exercise.id,
      sets: item.sets,
      reps: item.reps,
      load_lbs: item.load_lbs,
      notes: item.notes,
      order_index: item.order_index
    }));

    const response = await fetch(url, {
      method: method,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        name: routineName,
        exercises: payloadExercises
      })
    });

    if (response.ok) {
      const action = isEditingRoutine ? 'updated' : 'saved';
      alert(`Routine ${action} successfully!`);

      // Reset form and reload routines
      resetForm();
      loadExistingRoutines();
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

  // Ctrl+R to reset form
  if (e.ctrlKey && e.key === 'r') {
    e.preventDefault();
    if (confirm('Reset form and clear current routine?')) {
      resetForm();
    }
  }
});