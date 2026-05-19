"""
Entity Resolution Showcase — Interactive Menu
==============================================
Run this file to browse methods by phase and execute them.

    python menu.py

Requires: pip install rich
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.padding import Padding
except ImportError:
    print("Missing dependency: pip install rich")
    sys.exit(1)

console = Console()

# ── Method registry ────────────────────────────────────────────────────────────

PHASES = [
    {
        "id": 0,
        "name": "Preprocessing",
        "color": "cyan",
        "icon": "◈",
        "description": "Detect near-duplicate datasets before running ER",
        "methods": [
            {
                "key": "0",
                "name": "Near-Duplicate Dataset Detection",
                "lib": "CatBoost",
                "file": "00_preprocessing.py",
                "dok": "Dok. 3",
                "description": "Classify dataset pairs as near-dup or different-source using CatBoost on Jaccard/size features",
            },
        ],
    },
    {
        "id": 1,
        "name": "Blocking",
        "color": "yellow",
        "icon": "◈",
        "description": "Reduce O(n×m) pairs to a high-recall candidate set",
        "methods": [
            {
                "key": "1",
                "name": "Token Blocking",
                "lib": "recordlinkage",
                "file": "01_token_blocking.py",
                "dok": "Dok. 2",
                "description": "Explode records by tokens; block on exact token match",
            },
            {
                "key": "2",
                "name": "Sorted Neighborhood",
                "lib": "recordlinkage",
                "file": "02_sorted_neighborhood.py",
                "dok": "Dok. 2",
                "description": "Sort by key; sliding window of size W generates candidates",
            },
        ],
    },
    {
        "id": 2,
        "name": "Block Processing",
        "color": "magenta",
        "icon": "◈",
        "description": "Prune candidates without losing true matches",
        "methods": [
            {
                "key": "3",
                "name": "Meta-Blocking (WNP)",
                "lib": "custom",
                "file": "03_meta_blocking.py",
                "dok": "Dok. 2",
                "description": "Weight pairs by block co-occurrence; prune by threshold (WNP)",
            },
        ],
    },
    {
        "id": 3,
        "name": "Labeling",
        "color": "bright_cyan",
        "icon": "◈",
        "description": "Generate training labels via weak supervision — no hand-labeling needed",
        "methods": [
            {
                "key": "4",
                "name": "Snorkel Labeling",
                "lib": "snorkel",
                "file": "04_snorkel_labeling.py",
                "dok": "Dok. 3",
                "description": "Programmatic LFs combined by LabelModel → saves snorkel_labels.csv for matchers",
            },
        ],
    },
    {
        "id": 4,
        "name": "Matching",
        "color": "green",
        "icon": "◈",
        "description": "Classify candidate pairs as match / non-match (uses snorkel_labels.csv)",
        "methods": [
            {
                "key": "5",
                "name": "Ditto",
                "lib": "HuggingFace Transformers",
                "file": "05_ditto.py",
                "dok": "Dok. 4",
                "description": "Full fine-tuning of BERT on serialised [COL]/[VAL] record pairs",
            },
            {
                "key": "6",
                "name": "AdapterEM",
                "lib": "PEFT / LoRA",
                "file": "06_adapter_em.py",
                "dok": "Dok. 5",
                "description": "Parameter-efficient fine-tuning — only adapter weights (~13%) are trained",
            },
            {
                "key": "7",
                "name": "DeepMatcher",
                "lib": "PyTorch",
                "file": "07_deepmatcher.py",
                "dok": "Dok. 4–5",
                "description": "GRU attribute embeddings + attention aggregation + MLP classifier",
            },
            {
                "key": "8",
                "name": "DeepER",
                "lib": "sentence-transformers",
                "file": "08_deeper.py",
                "dok": "Dok. 4",
                "description": "Bi-encoder: encode each record, match by cosine similarity",
            },
            {
                "key": "9",
                "name": "CoT Distillation",
                "lib": "OpenAI + DistilBERT",
                "file": "09_cot_distillation.py",
                "dok": "Dok. 1",
                "description": "LLM generates chain-of-thought; smaller model distilled on CoT labels",
            },
        ],
    },
    {
        "id": 5,
        "name": "Clustering",
        "color": "red",
        "icon": "◈",
        "description": "Group matched pairs into entity clusters",
        "methods": [
            {
                "key": "10",
                "name": "Connected Components",
                "lib": "networkx",
                "file": "10_connected_components.py",
                "dok": "Dok. 2",
                "description": "Graph of match edges; each connected component = one entity",
            },
            {
                "key": "11",
                "name": "Correlation Clustering",
                "lib": "pyjedai",
                "file": "11_correlation_clustering.py",
                "dok": "Dok. 2",
                "description": "Optimise agreement/disagreement edges for globally consistent clusters",
            },
        ],
    },
]

ALL_METHODS = {m["key"]: (phase, m) for phase in PHASES for m in phase["methods"]}


# ── Rendering helpers ──────────────────────────────────────────────────────────

def render_header():
    console.print()
    console.print(Panel(
        Text.assemble(
            ("Entity Resolution Showcase", "bold white"),
            "\n",
            ("Pedigree Dog Records  ·  Source A (15 rec) vs Source B (15 rec)  ·  10 true matches", "dim"),
        ),
        box=box.DOUBLE,
        border_style="bright_blue",
        padding=(0, 4),
    ))


def render_menu():
    render_header()

    for phase in PHASES:
        color = phase["color"]
        console.print(Rule(
            f"[bold {color}]Phase {phase['id']} — {phase['name']}[/]  [dim]{phase['description']}[/]",
            style=color,
        ))

        table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
        table.add_column("key",  style="bold white", width=5, justify="right")
        table.add_column("name", style=f"bold {color}", width=30)
        table.add_column("lib",  style="dim", width=26)
        table.add_column("dok",  style="dim cyan", width=8)
        table.add_column("desc", style="dim white")

        for m in phase["methods"]:
            table.add_row(
                f"[{m['key']}]",
                m["name"],
                m["lib"],
                m["dok"],
                m["description"],
            )

        console.print(table)
        console.print()

    console.print(Rule("[dim]Actions[/]", style="bright_black"))
    table = Table(box=None, show_header=False, padding=(0, 2), expand=False)
    table.add_column("key",  style="bold white", width=5, justify="right")
    table.add_column("desc", style="white")
    table.add_row("[P]", "[bold green]Run Full Pipeline[/]  (all stages end-to-end)")
    table.add_row("[Q]", "[bold red]Quit[/]")
    console.print(table)
    console.print()


def run_script(filepath: str, label: str):
    console.print()
    console.print(Panel(
        f"[bold white]Running:[/] [yellow]{label}[/]\n[dim]{filepath}[/]",
        border_style="yellow",
        box=box.ROUNDED,
        padding=(0, 2),
    ))
    console.print()

    result = subprocess.run(
        [sys.executable, filepath],
        cwd=HERE,
    )

    console.print()
    if result.returncode == 0:
        console.print(Panel(
            "[bold green]Completed successfully.[/]",
            border_style="green",
            box=box.ROUNDED,
            padding=(0, 2),
        ))
    else:
        console.print(Panel(
            f"[bold red]Exited with code {result.returncode}.[/]",
            border_style="red",
            box=box.ROUNDED,
            padding=(0, 2),
        ))


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    while True:
        console.clear()
        render_menu()

        try:
            choice = console.input("[bold bright_blue]Select [dim](0–11 / P / Q)[/]:[/] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye.[/]")
            break

        if choice == "q":
            console.print("\n[dim]Bye.[/]")
            break

        elif choice == "p":
            run_script(os.path.join(HERE, "pipeline.py"), "Full ER Pipeline")
            console.input("\n[dim]Press Enter to return to menu…[/]")

        elif choice in ALL_METHODS:
            phase, method = ALL_METHODS[choice]
            filepath = os.path.join(HERE, method["file"])
            run_script(filepath, f"Phase {phase['id']} — {method['name']}")
            console.input("\n[dim]Press Enter to return to menu…[/]")

        else:
            console.print(f"\n[red]Unknown choice: '{choice}'. Use 0–10, P, or Q.[/]")
            console.input("[dim]Press Enter to continue…[/]")


if __name__ == "__main__":
    main()
