from __future__ import annotations

import argparse

from md2_dice import build_dice_3mf, build_final_svgs, generate_previews


def command_validate(_args: argparse.Namespace) -> None:
    build_final_svgs.validate()


def command_svgs(args: argparse.Namespace) -> None:
    build_final_svgs.regenerate(replace_from_source=args.replace_from_source)


def command_build(_args: argparse.Namespace) -> None:
    build_dice_3mf.main()


def command_previews(_args: argparse.Namespace) -> None:
    generate_previews.main()


def command_all(args: argparse.Namespace) -> None:
    command_svgs(args)
    command_build(args)
    command_previews(args)


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
    args.func(args)


if __name__ == "__main__":
    main()
