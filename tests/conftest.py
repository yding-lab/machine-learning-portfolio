import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Point artifacts and reports at a temp dir BEFORE ml_portfolio is imported, so tests never
# overwrite the committed reports or a locally trained model.
TMP = Path(tempfile.mkdtemp(prefix="ml_portfolio_tests_"))
os.environ["ARTIFACT_DIR"] = str(TMP / "artifacts")
os.environ["REPORT_DIR"] = str(TMP / "reports")
os.environ["MODEL_PATH"] = str(TMP / "artifacts" / "best_model.joblib")


@pytest.fixture(scope="session")
def trained():
    """Train once with the quick grid into the temp dir."""
    from ml_portfolio import train

    return {"metadata": train.train_and_select(train.QUICK_CONFIG), "dir": TMP}
