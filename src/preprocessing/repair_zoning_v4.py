import json
import re
from pathlib import Path

INPUT_PATH = Path("zoningv4data.jsonl")
OUTPUT_PATH = Path("repaired_zoning_v4.jsonl")
BAD_LINES_PATH = Path("unrepaired_zoning_v4.txt")


def normalize_action_text(text: str) -> str:
    text = text.strip()
    if text.startswith("action:"):
        text = text[len("action:"):].strip()
    if text.startswith("Action:"):
        text = text[len("Action:"):].strip()
    return text


def extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        raise ValueError(f"Start marker not found: {start_marker}")
    start += len(start_marker)

    end = text.find(end_marker, start)
    if end == -1:
        raise ValueError(f"End marker not found: {end_marker}")

    return text[start:end]


def find_matching_brace(s: str, start_idx: int) -> int:
    depth = 0
    in_string = False
    escape = False

    for i in range(start_idx, len(s)):
        ch = s[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

    raise ValueError("No matching closing brace found")


def parse_projectiles_array(array_text: str):
    """
    Parse a projectile array that may look like:
    []
    or
    ["{\"damage\":5,...}", "{\"damage\":15,...}"]

    Returns a Python list of strings (same style as original dataset).
    """
    array_text = array_text.strip()
    if array_text == "[]":
        return []

    if not (array_text.startswith("[") and array_text.endswith("]")):
        raise ValueError(f"Invalid projectile array text: {array_text[:120]}")

    inner = array_text[1:-1].strip()
    if not inner:
        return []

    items = []
    cur = []
    in_string = False
    escape = False

    for ch in inner:
        if escape:
            cur.append(ch)
            escape = False
            continue

        if ch == "\\":
            cur.append(ch)
            escape = True
            continue

        if ch == '"':
            cur.append(ch)
            in_string = not in_string
            continue

        if ch == "," and not in_string:
            item = "".join(cur).strip()
            if item:
                items.append(item)
            cur = []
            continue

        cur.append(ch)

    tail = "".join(cur).strip()
    if tail:
        items.append(tail)

    parsed = []
    for item in items:
        item = item.strip()
        if item.startswith('"') and item.endswith('"'):
            parsed.append(json.loads(item))
        else:
            parsed.append(item)

    return parsed


def parse_state_content(raw_content: str) -> dict:
    """
    Repair-parse the inner messages[1]['content'] string.
    Assumes the overall shape is:
    {"self":{...},"opponent":{...}}
    with projectile arrays containing escaped JSON strings.
    """
    # First try the easy path
    try:
        return json.loads(raw_content)
    except Exception:
        pass

    # Manual recovery path
    raw = raw_content.strip()
    if not raw.startswith("{"):
        raise ValueError("Inner content does not start with {")

    def parse_side(side_name: str):
        side_marker = f'"{side_name}":{{'
        side_pos = raw.find(side_marker)
        if side_pos == -1:
            raise ValueError(f"Missing side: {side_name}")

        obj_start = raw.find("{", side_pos)
        obj_end = find_matching_brace(raw, obj_start)
        side_obj_text = raw[obj_start:obj_end + 1]

        # Try direct parse first
        try:
            return json.loads(side_obj_text)
        except Exception:
            pass

        # Fallback manual parse for projectile field
        proj_marker = '"projectiles":'
        proj_pos = side_obj_text.find(proj_marker)
        if proj_pos == -1:
            raise ValueError(f"Missing projectiles in {side_name}")

        proj_array_start = side_obj_text.find("[", proj_pos)
        if proj_array_start == -1:
            raise ValueError(f"Missing projectile array start in {side_name}")

        # Find matching ] for projectiles
        depth = 0
        in_string = False
        escape = False
        proj_array_end = None

        for i in range(proj_array_start, len(side_obj_text)):
            ch = side_obj_text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    proj_array_end = i
                    break

        if proj_array_end is None:
            raise ValueError(f"Could not find projectile array end in {side_name}")

        projectile_array_text = side_obj_text[proj_array_start:proj_array_end + 1]
        projectiles = parse_projectiles_array(projectile_array_text)

        # Replace projectile text with a placeholder valid JSON array
        placeholder = "__PROJECTILES_PLACEHOLDER__"
        repaired_side_text = (
            side_obj_text[:proj_pos]
            + f'"projectiles":"{placeholder}"'
            + side_obj_text[proj_array_end + 1:]
        )

        side_data = json.loads(repaired_side_text)
        side_data["projectiles"] = projectiles
        return side_data

    self_data = parse_side("self")
    opponent_data = parse_side("opponent")
    return {"self": self_data, "opponent": opponent_data}


def repair_line(line: str) -> dict:
    """
    Rebuild a clean row from malformed raw line text.
    """
    # system content
    system_content = extract_between(
        line,
        '{"role": "system", "content": "',
        '"},{"role": "user", "content": "'
    )

    # assistant content
    assistant_content = extract_between(
        line,
        '"},{"role": "assistant", "content": "',
        '"}]}'
    )
    assistant_content = normalize_action_text(assistant_content)

    # user content raw blob
    user_start_marker = '"},{"role": "user", "content": "'
    user_end_marker = '"},{"role": "assistant", "content": "'

    user_start = line.find(user_start_marker)
    if user_start == -1:
        raise ValueError("Could not find user content start")
    user_start += len(user_start_marker)

    user_end = line.find(user_end_marker, user_start)
    if user_end == -1:
        raise ValueError("Could not find user content end")

    raw_user_content = line[user_start:user_end]

    # Unescape outer string layer only once
    raw_user_content = raw_user_content.replace('\\"', '"').replace("\\\\", "\\")

    state = parse_state_content(raw_user_content)

    repaired_row = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    return repaired_row


def main():
    repaired = 0
    passed = 0
    failed = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout, \
         BAD_LINES_PATH.open("w", encoding="utf-8") as fbad:

        for line_num, line in enumerate(fin, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue

            # First try normal parsing
            try:
                row = json.loads(raw)

                # Normalize assistant action if needed
                if "messages" in row and len(row["messages"]) >= 3:
                    row["messages"][2]["content"] = normalize_action_text(
                        row["messages"][2]["content"]
                    )

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                passed += 1
                continue

            except Exception:
                pass

            # Try repair path
            try:
                repaired_row = repair_line(raw)
                fout.write(json.dumps(repaired_row, ensure_ascii=False) + "\n")
                repaired += 1

            except Exception as e:
                failed += 1
                fbad.write(f"LINE {line_num}: {e}\n")
                fbad.write(raw + "\n\n")

    print(f"Passed without repair: {passed}")
    print(f"Repaired successfully: {repaired}")
    print(f"Failed to repair:      {failed}")
    print(f"Saved repaired JSONL to: {OUTPUT_PATH.resolve()}")
    print(f"Saved unrepaired lines to: {BAD_LINES_PATH.resolve()}")


if __name__ == "__main__":
    main()