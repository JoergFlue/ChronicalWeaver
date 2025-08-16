#!/usr/bin/env python3
"""
Chronicle Weaver - Main Application Entry Point

This is the main entry point for the Chronicle Weaver application.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import chronicle_weaver


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Chronicle Weaver - AI-driven roleplaying assistant"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"Chronicle Weaver {chronicle_weaver.__version__}",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    if args.debug:
        print(f"Chronicle Weaver {chronicle_weaver.__version__}")
        print(f"Debug mode: {args.debug}")
        print(f"Log level: {args.log_level}")
        print("Project structure verified successfully!")

        # In Phase 1, we'll initialize the UI here
        print("\nPhase 0 setup complete. Ready for Phase 1 development.")
        return 0

    # For now, just show version info
    print(f"Chronicle Weaver {chronicle_weaver.__version__}")
    print("Phase 0: Foundation setup complete")
    print("UI initialization will be implemented in Phase 1")

    return 0


if __name__ == "__main__":
    sys.exit(main())
