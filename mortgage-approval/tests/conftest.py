import os
import sys
import tempfile
from pathlib import Path

import pytest

TMP = Path(tempfile.mkdtemp(prefix="mortgage-tests-"))
os.environ["REPORT_DIR"] = str(TMP / "reports")  # set before the package is imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def quick_run():
    from mortgage_approval.config import QUICK
    from mortgage_approval.train import run

    return run(QUICK)
