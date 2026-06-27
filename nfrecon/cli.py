"""Top-level CLI entrypoint for the NFRECON framework.

This module provides a minimal command dispatcher that routes user
commands to the appropriate submodules (e.g., reconstruction or evaluation),
while delegating all argument parsing to the underlying implementations.
"""

import sys

AVAILABLE_COMMANDS = ["reconstruct", "evaluate"]


def main() -> None:
    """Entry point for the NFRECON command-line interface.

    This function dispatches execution to one of the supported commands:
    - ``reconstruct``: MRI reconstruction using Hydra configuration
    - ``evaluate``: model evaluation using argparse

    All arguments after the command are passed through unchanged to the
    corresponding module.

    Raises:
        SystemExit: If no command or an unknown command is provided.
    """
    if len(sys.argv) < 2:
        _print_usage_and_exit()

    command: str = sys.argv[1]
    args: list[str] = sys.argv[2:]

    # Forward arguments to the target command
    sys.argv = [sys.argv[0]] + args

    if command == "reconstruct":
        from nfrecon.reconstruct import main as reconstruct_main

        reconstruct_main()

    elif command == "evaluate":
        from nfrecon.evaluate import main as evaluate_main

        evaluate_main()

    else:
        _print_unknown_command_and_exit(command)


def _print_usage_and_exit() -> None:
    """Print a helpful usage message and exit the program.

    Called when no command is provided.
    """
    print("Usage: nfrecon <command> [options]\n")
    print("Available commands:")
    for cmd in AVAILABLE_COMMANDS:
        print(f"  {cmd}")
    sys.exit(1)


def _print_unknown_command_and_exit(command: str) -> None:
    """Print an error message for an unknown command and exit.

    Parameters
    ----------
    command : str
        unrecognized command name provided by the user.
    """
    print(f"Error: unknown command '{command}'\n")
    print("Available commands:")
    for cmd in AVAILABLE_COMMANDS:
        print(f"  {cmd}")
    sys.exit(1)
