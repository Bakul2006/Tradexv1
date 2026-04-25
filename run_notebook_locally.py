import json
import traceback
from pathlib import Path

NOTEBOOK_PATH = Path(r"c:\Users\Lenovo\OneDrive\Desktop\TRADEXV1\Tradexv1\GRPO_Training_Colab.ipynb")


def load_notebook(path: Path):
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text)


def main():
    notebook = load_notebook(NOTEBOOK_PATH)
    namespace = {"__name__": "__main__"}

    for index, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        if source and str(source[0]).lstrip().startswith("!"):
            print(f"\n=== Skipping install-only code cell {index} ===")
            continue

        code = "\n".join(source)
        print(f"\n=== Running code cell {index} ===")

        try:
            exec(compile(code, f"{NOTEBOOK_PATH.name}:cell{index}", "exec"), namespace)
        except Exception:
            print(f"\nFAILED AT CODE CELL {index}")
            traceback.print_exc()
            raise

    print("\nNOTEBOOK EXECUTION FINISHED")


if __name__ == "__main__":
    main()
