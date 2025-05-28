// Load exercises from backend
function loadExercises() {
  fetch('/runstrong/exercises')
    .then(res => {
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      return res.json();
    })
    .then(data => {
      const pool = document.getElementById("exercise-pool");
      pool.innerHTML = "";
      data.exercises.forEach(ex => {
        const li = document.createElement("li");
        li.className = "draggable";
        li.dataset.id = ex['id'];  // ex[0] is id, ex[1] is name
        li.textContent = ex['name'];
        pool.appendChild(li);
      });
    })
    .catch(err => {
      console.error('Error loading exercises:', err);
      alert("Error loading exercises. Please refresh the page.");
    });
}

// Drag and drop setup
const exercisePool = Sortable.create(document.getElementById("exercise-pool"), {
  group: {
    name: "shared",
    pull: "clone",
    put: false
  },
  animation: 150,
  sort: false,
  ghostClass: "sortable-ghost",
  chosenClass: "sortable-chosen"
});

const routineList = Sortable.create(document.getElementById("routine-list"), {
  group: {
    name: "shared",
    pull: true,
    put: true
  },
  animation: 150,
  ghostClass: "sortable-ghost",
  chosenClass: "sortable-chosen",
  onAdd: function(evt) {
    // When cloning from exercise pool, ensure proper data attributes
    const item = evt.item;
    if (!item.classList.contains('from-routine')) {
      item.classList.add('draggable');
    }
    // Add remove button to each item in routine list
    addRemoveButton(item);
  }
});

// Function to add remove button to routine items
function addRemoveButton(item) {
  if (item.querySelector('.remove-btn')) return; // Don't add if already exists
  
  const removeBtn = document.createElement('span');
  removeBtn.innerHTML = ' ✕';
  removeBtn.className = 'remove-btn';
  removeBtn.style.cssText = 'margin-left: 10px; cursor: pointer; color:rgb(15, 2, 2); font-weight: bold; float: right;';
  removeBtn.title = 'Remove from routine';
  
  removeBtn.addEventListener('click', function(e) {
    e.stopPropagation();
    item.remove();
  });
  
  item.appendChild(removeBtn);
}

// Save routine
document.getElementById("save-routine").addEventListener("click", () => {
  const name = document.getElementById("routine-name").value.trim();
  const items = document.querySelectorAll("#routine-list li");
  const routine = Array.from(items).map((el, idx) => ({
    id: el.dataset.id,
    order: idx
  }));

  if (!name || routine.length === 0) {
    alert("Please enter a routine name and add at least one exercise.");
    return;
  }

  fetch("/runstrong/save-routine", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, routine })
  })
  .then(res => {
    if (!res.ok) {
      throw new Error(`HTTP error! status: ${res.status}`);
    }
    return res.json();
  })
  .then(data => {
    if (data.status === 'success') {
      alert("Routine saved successfully!");
      loadRoutines();
      // Clear the form
      document.getElementById("routine-name").value = '';
      document.getElementById("routine-list").innerHTML = '';
    } else {
      alert("Error saving routine: " + (data.message || 'Unknown error'));
    }
  })
  .catch(err => {
    console.error('Error saving routine:', err);
    alert("Error saving routine. Please try again.");
  });
});

// Load routines into dropdown
function loadRoutines() {
  fetch("/runstrong/load-routines")
    .then(res => {
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      return res.json();
    })
    .then(data => {
      const selector = document.getElementById("routine-selector");
      selector.innerHTML = '<option value="">Select a routine...</option>';
      
      data.routines.forEach(r => {
        const opt = document.createElement("option");
        opt.value = r['id'];  // r[0] is id, r[1] is name
        opt.textContent = r['name'];
        selector.appendChild(opt);
      });
    })
    .catch(err => {
      console.error('Error loading routines:', err);
      alert("Error loading routines. Please refresh the page.");
    });
}

// Load a selected routine
document.getElementById("load-routine").addEventListener("click", () => {
  const id = document.getElementById("routine-selector").value;
  if (!id) {
    alert("Please select a routine to load.");
    return;
  }

  fetch(`/runstrong/load-routine/${id}`)
    .then(res => {
      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }
      return res.json();
    })
    .then(data => {
      const list = document.getElementById("routine-list");
      list.innerHTML = "";
      
      data.exercises.forEach(ex => {
        const li = document.createElement("li");
        li.className = "draggable from-routine";
        li.dataset.id = ex['id'];  // ex[0] is id, ex[1] is name
        li.textContent = ex['name'];
        addRemoveButton(li);
        list.appendChild(li);
      });
    })
    .catch(err => {
      console.error('Error loading routine:', err);
      alert("Error loading routine. Please try again.");
    });
});

// Initial population
document.addEventListener("DOMContentLoaded", () => {
  loadRoutines();
  loadExercises();
});