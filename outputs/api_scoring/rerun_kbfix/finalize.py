"""Build final rerun reports only after all answers and both judges are complete."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from outputs.api_scoring.rerun_kbfix import build_extra_reports, build_reports, prepare_v4_data

HERE = Path(__file__).resolve().parent


def main() -> None:
    # build_reports checks the live answer file, snapshot hashes, all 1,200 item
    # files, and every metric from both judges before its first write.
    build_reports.main()
    build_extra_reports.main()
    prepare_v4_data.main()
    env = os.environ.copy()
    env["RERUN_SCORING_DIR"] = str(build_reports.OUT)
    for script in ("build_workbooks.mjs", "build_v4.mjs"):
        subprocess.run(["node", str(HERE / script)], check=True, env=env)
    print("Final rerun reports and workbooks are complete.")


if __name__ == "__main__":
    main()
