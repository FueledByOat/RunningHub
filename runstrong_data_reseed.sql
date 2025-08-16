-- ==========================
-- EXERCISES
-- ==========================

-- Lower Body Strength
INSERT INTO exercises (name, description, type) VALUES
('Back Squat','Barbell squat focusing on bilateral leg strength','Strength'),
('Front Squat','Barbell squat emphasizing quadriceps and core stability','Strength'),
('Trap Bar Deadlift','Hex bar deadlift emphasizing posterior chain','Strength'),
('Conventional Deadlift','Standard deadlift for posterior chain strength','Strength'),
('Bulgarian Split Squat','Unilateral squat with rear foot elevated','Strength'),
('Step-Up','Weighted or bodyweight step-ups onto a box or bench','Strength'),
('Single-Leg Romanian Deadlift','Unilateral hip hinge for hamstrings and glutes','Strength'),
('Walking Lunge','Forward lunges for unilateral strength and balance','Strength'),
('Skater Squat','Single-leg squat with rear leg hovering behind','Strength'),
('Cossack Squat','Lateral squat variation for adductors and mobility','Strength'),
('Single-Leg Glute Bridge','Unilateral glute and hamstring bridge','Strength');

-- Calves & Foot
INSERT INTO exercises (name, description, type) VALUES
('Standing Calf Raise','Straight-leg calf raise targeting gastrocnemius','Strength'),
('Seated Calf Raise','Bent-knee calf raise for soleus','Strength'),
('Single-Leg Calf Raise','Unilateral calf strengthening exercise','Strength'),
('Eccentric Heel Drop','Achilles tendon loading exercise','Accessory'),
('Toe Yoga','Foot intrinsic strengthening exercise','Accessory');

-- Plyometrics
INSERT INTO exercises (name, description, type) VALUES
('Box Jump','Explosive bilateral jump onto a box','Plyometric'),
('Broad Jump','Horizontal jump for power','Plyometric'),
('Bounding','Running bounds for plyometric strength','Plyometric'),
('Hurdle Hops','Jumps over hurdles for reactivity','Plyometric'),
('Split Squat Jump','Jumping lunge for power','Plyometric'),
('Lateral Skater Jump','Side-to-side explosive jumps','Plyometric'),
('Depth Jump','Drop from a box into immediate vertical jump','Plyometric');

-- Core / Anti-Rotation
INSERT INTO exercises (name, description, type) VALUES
('Pallof Press','Anti-rotation core exercise with band or cable','Core'),
('Dead Bug','Core stability exercise with arm and leg movement','Core'),
('Bird Dog','Spinal stability exercise with opposite arm and leg lift','Core'),
('Plank','Isometric core hold with variations','Core'),
('Suitcase Carry','Unilateral loaded carry for anti-lateral flexion','Core'),
('Turkish Get-Up','Full-body movement emphasizing stability','Core'),
('Cable Chop','Rotational core exercise with cable machine','Core'),
('Russian Twist','Rotational core exercise seated on floor','Core'),
('Medicine Ball Slam','Explosive rotational or overhead slam','Core');

-- Upper Body
INSERT INTO exercises (name, description, type) VALUES
('Pull-Up','Bodyweight vertical pulling exercise','Upper Body'),
('Inverted Row','Horizontal pulling exercise using bodyweight','Upper Body'),
('Face Pull','Posterior shoulder and scapular stabilizer exercise','Upper Body'),
('Band Pull-Apart','Scapular retraction exercise with band','Upper Body'),
('Push-Up','Bodyweight horizontal pushing exercise','Upper Body'),
('Dumbbell Bench Press','Horizontal pressing variation with dumbbells','Upper Body'),
('Overhead Press','Vertical press with dumbbells or barbell','Upper Body'),
('Arnold Press','Shoulder press variation with rotation','Upper Body'),
('Bicep Curl','Elbow flexion strengthening exercise','Upper Body'),
('Tricep Dip','Triceps strengthening bodyweight or weighted dip','Upper Body');

-- Accessory / Injury Prevention
INSERT INTO exercises (name, description, type) VALUES
('Banded Side Walk','Hip abduction activation exercise','Accessory'),
('Clamshell','Glute medius activation exercise','Accessory'),
('Copenhagen Plank','Adductor isometric strengthening exercise','Accessory'),
('Nordic Hamstring Curl','Hamstring eccentric strengthening exercise','Accessory'),
('Swiss Ball Hamstring Curl','Supine hamstring curl on stability ball','Accessory'),
('Reverse Hyperextension','Glute and hamstring posterior chain exercise','Accessory'),
('Standing Hip Flexion with Band','Hip flexor strengthening exercise','Accessory');

-- ==========================
-- MUSCLE MAPPINGS
-- ==========================

-- 1.0 = primary mover

-- 0.7–0.9 = strong secondary mover

-- 0.3–0.6 = stabilizer or tertiary involvement

-- Squat Family
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Back Squat'),14,1.0),
((SELECT id FROM exercises WHERE name='Back Squat'),15,0.8),
((SELECT id FROM exercises WHERE name='Back Squat'),16,0.6),
((SELECT id FROM exercises WHERE name='Back Squat'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Front Squat'),14,1.0),
((SELECT id FROM exercises WHERE name='Front Squat'),15,0.6),
((SELECT id FROM exercises WHERE name='Front Squat'),21,0.5),
((SELECT id FROM exercises WHERE name='Front Squat'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Bulgarian Split Squat'),14,0.9),
((SELECT id FROM exercises WHERE name='Bulgarian Split Squat'),15,0.8),
((SELECT id FROM exercises WHERE name='Bulgarian Split Squat'),16,0.5),
((SELECT id FROM exercises WHERE name='Bulgarian Split Squat'),18,0.3);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Step-Up'),14,0.9),
((SELECT id FROM exercises WHERE name='Step-Up'),15,0.8),
((SELECT id FROM exercises WHERE name='Step-Up'),16,0.4),
((SELECT id FROM exercises WHERE name='Step-Up'),25,0.3);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Walking Lunge'),14,0.8),
((SELECT id FROM exercises WHERE name='Walking Lunge'),15,0.8),
((SELECT id FROM exercises WHERE name='Walking Lunge'),16,0.6),
((SELECT id FROM exercises WHERE name='Walking Lunge'),25,0.3);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Skater Squat'),14,0.8),
((SELECT id FROM exercises WHERE name='Skater Squat'),15,0.7),
((SELECT id FROM exercises WHERE name='Skater Squat'),16,0.6),
((SELECT id FROM exercises WHERE name='Skater Squat'),18,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Cossack Squat'),14,0.7),
((SELECT id FROM exercises WHERE name='Cossack Squat'),17,0.9),
((SELECT id FROM exercises WHERE name='Cossack Squat'),18,0.6),
((SELECT id FROM exercises WHERE name='Cossack Squat'),25,0.3);

-- Hinge Family
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Trap Bar Deadlift'),15,0.9),
((SELECT id FROM exercises WHERE name='Trap Bar Deadlift'),16,1.0),
((SELECT id FROM exercises WHERE name='Trap Bar Deadlift'),25,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Conventional Deadlift'),15,0.9),
((SELECT id FROM exercises WHERE name='Conventional Deadlift'),16,1.0),
((SELECT id FROM exercises WHERE name='Conventional Deadlift'),25,0.7);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Single-Leg Romanian Deadlift'),16,1.0),
((SELECT id FROM exercises WHERE name='Single-Leg Romanian Deadlift'),15,0.8),
((SELECT id FROM exercises WHERE name='Single-Leg Romanian Deadlift'),18,0.4),
((SELECT id FROM exercises WHERE name='Single-Leg Romanian Deadlift'),25,0.3);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Single-Leg Glute Bridge'),15,1.0),
((SELECT id FROM exercises WHERE name='Single-Leg Glute Bridge'),16,0.8),
((SELECT id FROM exercises WHERE name='Single-Leg Glute Bridge'),25,0.3);

-- Calves / Foot
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Standing Calf Raise'),19,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Seated Calf Raise'),19,0.7),
((SELECT id FROM exercises WHERE name='Seated Calf Raise'),20,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Single-Leg Calf Raise'),19,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Eccentric Heel Drop'),19,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Toe Yoga'),20,0.8),
((SELECT id FROM exercises WHERE name='Toe Yoga'),19,0.4);

-- Plyometrics (similar to squat/hinge + calves)
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Box Jump'),14,0.8),
((SELECT id FROM exercises WHERE name='Box Jump'),15,0.8),
((SELECT id FROM exercises WHERE name='Box Jump'),19,0.9);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Broad Jump'),15,0.9),
((SELECT id FROM exercises WHERE name='Broad Jump'),16,0.7),
((SELECT id FROM exercises WHERE name='Broad Jump'),19,0.8);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Bounding'),15,0.9),
((SELECT id FROM exercises WHERE name='Bounding'),16,0.7),
((SELECT id FROM exercises WHERE name='Bounding'),19,1.0),
((SELECT id FROM exercises WHERE name='Bounding'),18,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Hurdle Hops'),14,0.7),
((SELECT id FROM exercises WHERE name='Hurdle Hops'),15,0.7),
((SELECT id FROM exercises WHERE name='Hurdle Hops'),19,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Split Squat Jump'),14,0.8),
((SELECT id FROM exercises WHERE name='Split Squat Jump'),15,0.8),
((SELECT id FROM exercises WHERE name='Split Squat Jump'),19,0.8);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Lateral Skater Jump'),18,0.7),
((SELECT id FROM exercises WHERE name='Lateral Skater Jump'),17,0.6),
((SELECT id FROM exercises WHERE name='Lateral Skater Jump'),19,0.9);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Depth Jump'),14,0.7),
((SELECT id FROM exercises WHERE name='Depth Jump'),15,0.7),
((SELECT id FROM exercises WHERE name='Depth Jump'),19,1.0);

-- Core
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Pallof Press'),22,0.8),
((SELECT id FROM exercises WHERE name='Pallof Press'),23,0.8),
((SELECT id FROM exercises WHERE name='Pallof Press'),24,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Dead Bug'),21,0.8),
((SELECT id FROM exercises WHERE name='Dead Bug'),24,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Bird Dog'),25,0.7),
((SELECT id FROM exercises WHERE name='Bird Dog'),26,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Plank'),21,0.8),
((SELECT id FROM exercises WHERE name='Plank'),24,0.6),
((SELECT id FROM exercises WHERE name='Plank'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Suitcase Carry'),22,0.7),
((SELECT id FROM exercises WHERE name='Suitcase Carry'),23,0.7),
((SELECT id FROM exercises WHERE name='Suitcase Carry'),24,0.6),
((SELECT id FROM exercises WHERE name='Suitcase Carry'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Turkish Get-Up'),15,0.5),
((SELECT id FROM exercises WHERE name='Turkish Get-Up'),21,0.5),
((SELECT id FROM exercises WHERE name='Turkish Get-Up'),24,0.5),
((SELECT id FROM exercises WHERE name='Turkish Get-Up'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Cable Chop'),22,0.8),
((SELECT id FROM exercises WHERE name='Cable Chop'),23,0.8),
((SELECT id FROM exercises WHERE name='Cable Chop'),21,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Russian Twist'),21,0.6),
((SELECT id FROM exercises WHERE name='Russian Twist'),22,0.8),
((SELECT id FROM exercises WHERE name='Russian Twist'),23,0.8);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Medicine Ball Slam'),21,0.8),
((SELECT id FROM exercises WHERE name='Medicine Ball Slam'),25,0.6);

-- Upper Body Push
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Push-Up'),1,1.0),
((SELECT id FROM exercises WHERE name='Push-Up'),5,0.7),
((SELECT id FROM exercises WHERE name='Push-Up'),2,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Dumbbell Bench Press'),1,1.0),
((SELECT id FROM exercises WHERE name='Dumbbell Bench Press'),5,0.7),
((SELECT id FROM exercises WHERE name='Dumbbell Bench Press'),2,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Overhead Press'),2,0.8),
((SELECT id FROM exercises WHERE name='Overhead Press'),3,0.8),
((SELECT id FROM exercises WHERE name='Overhead Press'),5,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Arnold Press'),2,0.7),
((SELECT id FROM exercises WHERE name='Arnold Press'),3,0.7),
((SELECT id FROM exercises WHERE name='Arnold Press'),5,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Tricep Dip'),5,1.0),
((SELECT id FROM exercises WHERE name='Tricep Dip'),1,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Bicep Curl'),6,1.0);

-- Upper Body Pull
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Pull-Up'),7,1.0),
((SELECT id FROM exercises WHERE name='Pull-Up'),8,0.7),
((SELECT id FROM exercises WHERE name='Pull-Up'),6,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Inverted Row'),7,0.7),
((SELECT id FROM exercises WHERE name='Inverted Row'),8,0.6),
((SELECT id FROM exercises WHERE name='Inverted Row'),6,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Face Pull'),4,0.6),
((SELECT id FROM exercises WHERE name='Face Pull'),9,0.8),
((SELECT id FROM exercises WHERE name='Face Pull'),10,0.6);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Band Pull-Apart'),8,0.8),
((SELECT id FROM exercises WHERE name='Band Pull-Apart'),9,0.7),
((SELECT id FROM exercises WHERE name='Band Pull-Apart'),4,0.6);

-- Accessory / Injury Prevention
INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Banded Side Walk'),18,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Clamshell'),18,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Copenhagen Plank'),17,1.0);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Nordic Hamstring Curl'),16,1.0),
((SELECT id FROM exercises WHERE name='Nordic Hamstring Curl'),15,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Swiss Ball Hamstring Curl'),16,1.0),
((SELECT id FROM exercises WHERE name='Swiss Ball Hamstring Curl'),15,0.5);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Reverse Hyperextension'),15,1.0),
((SELECT id FROM exercises WHERE name='Reverse Hyperextension'),16,0.8),
((SELECT id FROM exercises WHERE name='Reverse Hyperextension'),25,0.4);

INSERT INTO exercise_muscle_map VALUES
((SELECT id FROM exercises WHERE name='Standing Hip Flexion with Band'),20,0.8),
((SELECT id FROM exercises WHERE name='Standing Hip Flexion with Band'),21,0.4);

-- bonus additions
INSERT INTO exercises (name, description, type)
VALUES ('Lat Pulldown', 'Vertical pulling exercise targeting the lats and upper back, performed with a cable machine or pulldown station.', 'Strength');

-- Muscle mapping
INSERT INTO exercise_muscle_map (exercise_id, muscle_group_id, load_factor)
VALUES 
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 7, 1.0),  -- Latissimus Dorsi
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 2, 0.4),  -- Anterior Deltoid
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 3, 0.5),  -- Medial Deltoid
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 4, 0.6),  -- Posterior Deltoid
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 8, 0.7),  -- Rhomboids
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 9, 0.7),  -- Middle Trapezius
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 11, 0.4), -- Upper Trapezius
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 5, 0.4),  -- Triceps (stabilizer)
    ((SELECT id FROM exercises WHERE name='Lat Pulldown'), 6, 0.5);  -- Biceps

INSERT INTO exercises (name, description, type)
VALUES ('Chest Fly', 'Isolation exercise for the chest performed with dumbbells or a cable machine.', 'Strength');

-- Muscle mapping
INSERT INTO exercise_muscle_map (exercise_id, muscle_group_id, load_factor)
VALUES 
    ((SELECT id FROM exercises WHERE name='Chest Fly'), 1, 1.0),  -- Chest
    ((SELECT id FROM exercises WHERE name='Chest Fly'), 2, 0.5),  -- Anterior Deltoid
    ((SELECT id FROM exercises WHERE name='Chest Fly'), 5, 0.3),  -- Triceps (stabilizer)
    ((SELECT id FROM exercises WHERE name='Chest Fly'), 12, 0.2); -- Serratus Anterior

INSERT INTO exercises (name, description, type)
VALUES ('Pistol Squat', 'Single-leg squat requiring balance, strength, and mobility.', 'Strength');

-- Muscle mapping
INSERT INTO exercise_muscle_map (exercise_id, muscle_group_id, load_factor)
VALUES 
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 14, 1.0), -- Quadriceps
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 15, 0.7), -- Glutes
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 16, 0.6), -- Hamstrings
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 17, 0.5), -- Adductors
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 18, 0.5), -- Abductors
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 19, 0.4), -- Calves
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 21, 0.3), -- Rectus Abdominis
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 22, 0.3), -- External Obliques
    ((SELECT id FROM exercises WHERE name='Pistol Squat'), 25, 0.4); -- Erector Spinae

INSERT INTO exercises (name, description, type)
VALUES ('Tricep Pushdown', 'Cable isolation exercise for the triceps performed with a rope, bar, or band.', 'Strength');

-- Muscle mapping
INSERT INTO exercise_muscle_map (exercise_id, muscle_group_id, load_factor)
VALUES 
    ((SELECT id FROM exercises WHERE name='Tricep Pushdown'), 5, 1.0),  -- Triceps
    ((SELECT id FROM exercises WHERE name='Tricep Pushdown'), 2, 0.3),  -- Anterior Deltoid (stabilizer)
    ((SELECT id FROM exercises WHERE name='Tricep Pushdown'), 13, 0.4), -- Forearms (grip)
    ((SELECT id FROM exercises WHERE name='Tricep Pushdown'), 6, 0.2);  -- Biceps (antagonist balance)
