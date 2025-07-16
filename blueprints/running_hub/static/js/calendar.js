document.addEventListener('DOMContentLoaded', function () {
    const calendarEl = document.getElementById('calendar');
    const modal = document.getElementById('eventModal');
    const form = document.getElementById('workoutForm');
    const deleteBtn = document.getElementById('deleteWorkoutBtn');

    const workoutDateInput = document.getElementById('workout_date');
    let isEditMode = false;

    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'dayGridMonth',
        selectable: true,
        headerToolbar: {
            left: 'prev,next today',
            center: 'title',
            right: ''
        },
        eventContent: function (arg) {
            const workout = arg.event.extendedProps;
            const icon = workout.linked
                ? `<span class="linked-icon" title="Linked to Strava">&#x1F517;</span>`  // 🏃
                : "";

            return {
                html: `${icon}<span>${arg.event.title}</span>`
            };
        },
        dateClick: function (info) {
            form.reset();
            isEditMode = false;
            document.getElementById("workout_id").value = '';
            workoutDateInput.value = info.dateStr;
            deleteBtn.style.display = "none";
            modal.style.display = 'block';
        },
        eventClick: function (info) {
            const eventId = info.event.id;
            fetch(`${EVENTS_API_URL}/${eventId}`)
                .then(res => res.json())
                .then(data => {
                    isEditMode = true;
                    document.getElementById("workout_id").value = data.id;
                    document.getElementById("workout_date").value = data.workout_date;
                    document.getElementById("workout_name").value = data.workout_name || '';
                    document.getElementById("workout_type").value = data.workout_type || '';
                    document.getElementById("effort").value = data.effort || '';
                    document.getElementById("success").value = data.success || '';
                    document.getElementById("planned_notes").value = data.planned_notes || '';
                    document.getElementById("recap_notes").value = data.recap_notes || '';
                    document.getElementById("linked_activity_id").value = data.linked_activity_id || '';
                    deleteBtn.style.display = "inline-block";
                    modal.style.display = 'block';
                });
        },
        events: EVENTS_API_URL
    });

    calendar.render();

    form.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(new FormData(form).entries());
        const method = isEditMode ? 'PUT' : 'POST';

        try {
            const response = await fetch(POST_API_URL, {
                method: method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            if (!response.ok) throw new Error("Save failed");

            modal.style.display = 'none';
            form.reset();
            calendar.refetchEvents();
        } catch (err) {
            console.error(err);
            alert("Failed to save workout.");
        }
    };

    deleteBtn.onclick = async function () {
        const workoutId = document.getElementById("workout_id").value;
        if (!confirm("Are you sure you want to delete this workout?")) return;

        try {
            const response = await fetch(`${POST_API_URL}/${workoutId}`, { method: 'DELETE' });
            if (!response.ok) throw new Error("Delete failed");

            modal.style.display = 'none';
            form.reset();
            calendar.refetchEvents();
        } catch (err) {
            console.error(err);
            alert("Failed to delete workout.");
        }
    };

    // Close modal when clicking outside
    window.onclick = function (event) {
        if (event.target == modal) {
            modal.style.display = "none";
            form.reset();
        }
    };
});