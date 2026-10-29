import argparse
import csv
from collections import defaultdict
from pathlib import Path


def clean_model_name(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("local:"):
        value = value[len("local:"):]
    return Path(value).name or value or "unknown"


def parse_int(value, default=0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def detect_round_winner(row: dict) -> str:
    winner = (row.get("winner") or "").strip().lower()
    if winner in {"p1", "player1", "player_1", "1"}:
        return "p1"
    if winner in {"p2", "player2", "player_2", "2"}:
        return "p2"
    if winner in {"draw", "tie", "none"}:
        return "draw"

    p1_hp = parse_int(row.get("p1_hp"))
    p2_hp = parse_int(row.get("p2_hp"))
    if p1_hp > p2_hp:
        return "p1"
    if p2_hp > p1_hp:
        return "p2"
    return "draw"


def pct(num: int, den: int) -> str:
    if den <= 0:
        return "0.00%"
    return f"{num / den * 100:.2f}%"


def mean(total: float, count: int) -> str:
    if count <= 0:
        return "0.00"
    return f"{total / count:.2f}"


def print_summary(title: str, rows: list[dict]) -> None:
    if not rows:
        return

    print(f"\n=== {title} ===")
    for row in rows:
        print(f"\nmodel: {row['model']}")
        print(f"rounds: {row['rounds']}")
        print(f"round wins: {row['round_wins']}")
        print(f"round win rate: {row['round_winrate']}")
        print(f"games: {row['games']}")
        print(f"game wins: {row['game_wins']}")
        print(f"game win rate: {row['game_winrate']}")
        print(f"average hp: {row['avg_hp']}")


def summarize(csv_paths: list[Path]) -> tuple[dict, dict]:
    model_stats = defaultdict(lambda: {
        "rounds": 0,
        "round_wins": 0,
        "round_draws": 0,
        "hp_sum": 0,
        "games": 0,
        "game_wins": 0,
        "game_draws": 0,
    })
    file_stats = defaultdict(lambda: defaultdict(lambda: {
        "rounds": 0,
        "round_wins": 0,
        "round_draws": 0,
        "hp_sum": 0,
        "games": 0,
        "game_wins": 0,
        "game_draws": 0,
    }))

    games = defaultdict(list)

    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                p1_model = clean_model_name(row.get("p1_model"))
                p2_model = clean_model_name(row.get("p2_model"))
                p1_hp = parse_int(row.get("p1_hp"))
                p2_hp = parse_int(row.get("p2_hp"))
                winner = detect_round_winner(row)

                for stats in (model_stats, file_stats[str(csv_path)]):
                    stats[p1_model]["rounds"] += 1
                    stats[p1_model]["hp_sum"] += p1_hp
                    stats[p2_model]["rounds"] += 1
                    stats[p2_model]["hp_sum"] += p2_hp

                    if winner == "p1":
                        stats[p1_model]["round_wins"] += 1
                    elif winner == "p2":
                        stats[p2_model]["round_wins"] += 1
                    else:
                        stats[p1_model]["round_draws"] += 1
                        stats[p2_model]["round_draws"] += 1

                game_index = row.get("game_index") or row.get("game") or "0"
                games[(str(csv_path), game_index)].append({
                    "p1_model": p1_model,
                    "p2_model": p2_model,
                    "p1_hp": p1_hp,
                    "p2_hp": p2_hp,
                    "winner": winner,
                })

    for (csv_name, _game_index), rounds in games.items():
        if not rounds:
            continue

        p1_model = rounds[0]["p1_model"]
        p2_model = rounds[0]["p2_model"]
        p1_round_wins = sum(1 for r in rounds if r["winner"] == "p1")
        p2_round_wins = sum(1 for r in rounds if r["winner"] == "p2")
        p1_hp_sum = sum(r["p1_hp"] for r in rounds)
        p2_hp_sum = sum(r["p2_hp"] for r in rounds)

        if p1_round_wins > p2_round_wins:
            game_winner = "p1"
        elif p2_round_wins > p1_round_wins:
            game_winner = "p2"
        elif p1_hp_sum > p2_hp_sum:
            game_winner = "p1"
        elif p2_hp_sum > p1_hp_sum:
            game_winner = "p2"
        else:
            game_winner = "draw"

        for stats in (model_stats, file_stats[csv_name]):
            stats[p1_model]["games"] += 1
            stats[p2_model]["games"] += 1
            if game_winner == "p1":
                stats[p1_model]["game_wins"] += 1
            elif game_winner == "p2":
                stats[p2_model]["game_wins"] += 1
            else:
                stats[p1_model]["game_draws"] += 1
                stats[p2_model]["game_draws"] += 1

    return model_stats, file_stats


def format_stats(stats: dict) -> list[dict]:
    rows = []
    for model, s in sorted(stats.items(), key=lambda item: item[0]):
        rows.append({
            "model": model,
            "rounds": s["rounds"],
            "round_wins": s["round_wins"],
            "round_winrate": pct(s["round_wins"], s["rounds"]),
            "games": s["games"],
            "game_wins": s["game_wins"],
            "game_winrate": pct(s["game_wins"], s["games"]),
            "avg_hp": mean(s["hp_sum"], s["rounds"]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize FightingICE LLM-vs-LLM result CSVs."
    )
    parser.add_argument("csvs", nargs="+", help="One or more result CSV files.")
    parser.add_argument(
        "--no_by_file",
        action="store_true",
        help="Only print the combined summary.",
    )
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csvs]
    missing = [p for p in csv_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing CSV file(s): " + ", ".join(str(p) for p in missing)
        )

    model_stats, file_stats = summarize(csv_paths)

    print_summary("Combined Summary", format_stats(model_stats))

    if not args.no_by_file:
        for csv_name, stats in sorted(file_stats.items()):
            print_summary(f"File Summary: {csv_name}", format_stats(stats))


if __name__ == "__main__":
    main()
