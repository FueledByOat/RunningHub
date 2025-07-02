// runstrong_add_exercise.js

function addTagInput(sectionId, inputName) {
    const section = document.getElementById(sectionId);
    const newInput = document.createElement("input");
    newInput.type = "text";
    newInput.name = inputName;
    newInput.classList.add("tag-input");
    section.appendChild(newInput);
}

function gatherFormData() {
    const form = document.getElementById("exercise-form");
    const data = {};
    const formData = new FormData(form);

    // Iterate over all form elements
    for (const [key, value] of formData.entries()) {
        const cleanValue = value.trim();
        if (cleanValue === "") continue;

        // If the key indicates an array (e.g., 'primary_muscles[]')
        if (key.endsWith("[]")) {
            const realKey = key.slice(0, -2);
            if (!data[realKey]) {
                data[realKey] = []; // Initialize array if it doesn't exist
            }
            data[realKey].push(cleanValue);
        } else {
            data[key] = cleanValue;
        }
    }
    return data;
}
async function submitExerciseForm(event) {
    event.preventDefault();
    const form = event.target;
    // CORRECTED: gatherFormData now returns a clean JS object with arrays.
    // We don't need to JSON.stringify individual fields anymore.
    const data = gatherFormData();

    try {
        const response = await fetch(form.action, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            // The entire object is stringified here, which is correct.
            body: JSON.stringify(data)
        });

        const result = await response.json();
        const statusDiv = document.getElementById("form-status");
        if (response.ok) {
            form.reset();
            statusDiv.innerText = "Exercise added successfully!";
            statusDiv.style.color = "lime";
        } else {
            statusDiv.innerText = result.error || "Failed to add exercise.";
            statusDiv.style.color = "red";
        }
    } catch (error) {
        console.error("Submission error:", error);
        document.getElementById("form-status").innerText = "Submission error.";
    }
}