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

    new FormData(form).forEach((value, key) => {
        // Handle multiple input fields with the same name (tags)
        if (key.endsWith("[]")) {
            const realKey = key.slice(0, -2);
            data[realKey] = data[realKey] || [];
            if (value.trim() !== "") data[realKey].push(value.trim());
        } else {
            data[key] = value.trim() === "" ? null : value.trim();
        }
    });

    // JSON encode fields expected to be arrays
    const arrayFields = [
        'primary_muscles', 'secondary_muscles', 'progressions', 'regressions',
        'equipment_required', 'common_mistakes', 'training_style',
        'experience_level', 'goals', 'alternatives', 'supersets_well_with'
    ];
    arrayFields.forEach(field => {
        if (data[field]) {
            data[field] = JSON.stringify(data[field]);
        } else {
            data[field] = JSON.stringify([]);
        }
    });

    return data;
}

async function submitExerciseForm(event) {
    event.preventDefault();
    const form = event.target;
    const data = gatherFormData();

    try {
        const response = await fetch(form.action, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
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