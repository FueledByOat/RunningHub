import sqlite3
import re
from collections import defaultdict

DB_PATH = "strava_data.db"
MUSCLE_GROUP_TABLE = "muscle_groups"
RELATED_TABLES = ["muscle_group_fatigue", "exercise_muscle_groups"]

def normalize_name(name):
    if not name:
        return ""
    name = re.sub(r'[\[\]\"\']+', '', name).strip()
    name = re.sub(r'\s+', ' ', name)
    return name.lower().capitalize()

def deduplicate_muscle_groups():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Step 1: Fetch all muscle group rows
    cursor.execute(f"SELECT id, name FROM {MUSCLE_GROUP_TABLE}")
    rows = cursor.fetchall()

    # Step 2: Build normalized name → list of ids
    name_map = defaultdict(list)
    for row in rows:
        clean_name = normalize_name(row['name'])
        name_map[clean_name].append(row['id'])

    print(f"🧹 Found {len(name_map)} unique cleaned names from {len(rows)} records.")

    duplicates = {name: ids for name, ids in name_map.items() if len(ids) > 1}

    for clean_name, ids in duplicates.items():
        ids = sorted(ids)
        keep_id = ids[0]
        drop_ids = ids[1:]

        print(f"🔁 Merging {drop_ids} into {keep_id} for muscle group: '{clean_name}'")

        # Step 3: Update foreign keys in related tables
        for table in RELATED_TABLES:
            try:
                cursor.execute("PRAGMA table_info(%s)" % table)
                columns = [col["name"] for col in cursor.fetchall()]
                if "muscle_group_id" in columns:
                    for old_id in drop_ids:
                        cursor.execute(
                            f"""UPDATE {table}
                            SET muscle_group_id = ?
                            WHERE muscle_group_id = ?""",
                            (keep_id, old_id)
                        )
            except Exception as e:
                print(f"⚠️ Error updating table '{table}':", e)

        # Step 4: Delete duplicate entries from muscle_groups
        cursor.execute(
            f"DELETE FROM {MUSCLE_GROUP_TABLE} WHERE id IN ({','.join('?' for _ in drop_ids)})",
            drop_ids
        )

        # Step 5: Ensure the kept row has the clean name
        cursor.execute(
            f"UPDATE {MUSCLE_GROUP_TABLE} SET name = ? WHERE id = ?",
            (clean_name, keep_id)
        )

    conn.commit()
    conn.close()
    print("✅ Deduplication complete.")

if __name__ == "__main__":
    deduplicate_muscle_groups()