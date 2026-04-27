from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

from md2_dice import build_dice_3mf, build_final_svgs, generate_previews


console = Console()


@dataclass(frozen=True)
class Stage:
    label: str
    success: str
    action: Callable[[], None]


def print_header(title: str, detail: str) -> None:
    console.print(Panel.fit(f"[bold cyan]{title}[/]\n{detail}", border_style="cyan"))


def run_stage(stage: Stage) -> None:
    print_header(stage.label, "md2-dice is preparing the requested assets.")
    with console.status(f"{stage.label}...", spinner="dots"):
        stage.action()
    console.print(f"[bold green]OK[/] {stage.success}")


def run_pipeline(stages: list[Stage]) -> None:
    print_header("Massive Darkness dice assets", "Running the full SVG, 3MF, and preview pipeline.")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=len(stages))
        for stage in stages:
            progress.update(task, description=stage.label)
            stage.action()
            progress.advance(task)
        progress.update(task, description="Complete")
    console.print("[bold green]OK[/] Dice assets are up to date.")


def command_validate(_args: argparse.Namespace) -> None:
    run_stage(
        Stage(
            "Validating SVG assets",
            "SVG manifests and fit reports were regenerated.",
            build_final_svgs.validate,
        )
    )


def command_svgs(args: argparse.Namespace) -> None:
    run_stage(
        Stage(
            "Regenerating SVG candidates",
            "Source traces and SVG validation reports were regenerated.",
            lambda: build_final_svgs.regenerate(replace_from_source=args.replace_from_source),
        )
    )


def command_build(_args: argparse.Namespace) -> None:
    run_stage(
        Stage(
            "Building 3MF dice",
            "Final 3MF files were regenerated in build/3mf.",
            build_dice_3mf.main,
        )
    )


def command_previews(_args: argparse.Namespace) -> None:
    run_stage(
        Stage(
            "Rendering preview images",
            "Preview PNGs were regenerated in docs/previews.",
            generate_previews.main,
        )
    )


def command_all(args: argparse.Namespace) -> None:
    run_pipeline(
        [
            Stage(
                "Regenerating SVG candidates",
                "Source traces and SVG validation reports were regenerated.",
                lambda: build_final_svgs.regenerate(replace_from_source=args.replace_from_source),
            ),
            Stage(
                "Building 3MF dice",
                "Final 3MF files were regenerated in build/3mf.",
                build_dice_3mf.main,
            ),
            Stage(
                "Rendering preview images",
                "Preview PNGs were regenerated in docs/previews.",
                generate_previews.main,
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="md2-dice",
        description="Generate Massive Darkness 2 Hellscape dice assets.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="Validate SVG fit and regenerate SVG reports.")
    validate.set_defaults(func=command_validate)

    svgs = subcommands.add_parser("svgs", help="Regenerate source SVG candidates and validate assets/svg.")
    svgs.add_argument(
        "--replace-from-source",
        action="store_true",
        help="Overwrite automatically traceable SVGs with deterministic source-image traces.",
    )
    svgs.set_defaults(func=command_svgs, replace_from_source=False)

    build = subcommands.add_parser("build", help="Generate final 3MF dice files in build/3mf.")
    build.set_defaults(func=command_build)

    previews = subcommands.add_parser("previews", help="Render PNG previews from build/3mf.")
    previews.set_defaults(func=command_previews)

    all_command = subcommands.add_parser("all", help="Run SVG regeneration, build, and previews.")
    all_command.add_argument(
        "--replace-from-source",
        action="store_true",
        help="Overwrite automatically traceable SVGs before building.",
    )
    all_command.set_defaults(func=command_all, replace_from_source=False)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as error:
        console.print(f"[bold red]ERROR[/] {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
