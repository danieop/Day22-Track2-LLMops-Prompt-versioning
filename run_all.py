import argparse
import subprocess
import sys


STEPS = {
    "1": "01_langsmith_rag_pipeline.py",
    "2": "02_prompt_hub_ab_routing.py",
    "3": "03_ragas_evaluation.py",
    "4": "04_guardrails_validator.py",
}


def run_step(step: str) -> None:
    script = STEPS[step]
    print("\n" + "=" * 72)
    print(f"Running step {step}: {script}")
    print("=" * 72)
    subprocess.run([sys.executable, script], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=STEPS.keys(), help="Run a single step")
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
        return

    for step in STEPS:
        run_step(step)


if __name__ == "__main__":
    main()
