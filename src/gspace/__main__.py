"""Main entry point for gspace."""

import sys


def main():
    """Run gspace phases."""
    if len(sys.argv) < 2:
        print("Usage: gspace <phase>")
        print("  phase0  — model selection benchmark")
        print("  phase1  — behavioral + probe confirmation")
        print("  phase2  — manifold discovery")
        sys.exit(1)

    phase = sys.argv[1]
    if phase == "phase0":
        from gspace.phase0_benchmark import main as run
        run()
    else:
        print(f"Unknown phase: {phase}")
        sys.exit(1)


if __name__ == "__main__":
    main()
