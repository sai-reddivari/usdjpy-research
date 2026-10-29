"""Compile the two public documents. Never fits models or acquires market data."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    engine = os.environ.get("CODEX_TECTONIC_PATH") or shutil.which("tectonic")
    if not engine:
        raise SystemExit("Install Tectonic or set CODEX_TECTONIC_PATH to its executable.")
    (ROOT / "reports").mkdir(exist_ok=True)
    for source, destination in [("main", "usdjpy_research.pdf"), ("appendix", "usdjpy_technical_appendix.pdf")]:
        with tempfile.TemporaryDirectory(prefix="usdjpy-latex-") as tmp:
            subprocess.run([engine, "--keep-logs", "--outdir", tmp, source+".tex"],
                           cwd=ROOT/"latex", check=True)
            log = (Path(tmp)/(source+".log")).read_text()
            if "Overfull" in log or "undefined references" in log.lower():
                raise RuntimeError("Resolve layout/reference warnings before publishing: "+source)
            shutil.copy2(Path(tmp)/(source+".pdf"), ROOT/"reports"/destination)
            print("Built", destination)


if __name__ == "__main__":
    main()
