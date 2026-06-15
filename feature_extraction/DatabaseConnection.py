from pymongo import MongoClient
import pandas as pd
from pathlib import Path
from datetime import datetime

MCLIENT = "mongodb://localhost:27017/"
MDATABASE = "apticonduite_db3"


# ============================================================
# MongoDB Connection
# ============================================================

class myDatabase:
    def __init__(self, client=MCLIENT, db=MDATABASE):
        self.client = MongoClient(client)
        self.db = self.client[db]

    def get_collection(self, name, query={}):
        collection = self.db[name]
        cursor = collection.find(query)
        df = pd.DataFrame(list(cursor))
        return df

    def close(self):
        if self.client:
            self.client.close()


# ============================================================
# Latest Data Path Resolver (CORE FUNCTION)
# ============================================================

def _find_latest_local_output(output_root=Path("./output")):
    """
    Find latest timestamp folder under ./output

    Returns
    -------
    tuple[str, Path]
        latest_time, data_path
    """
    output_root = Path(output_root)

    if not output_root.exists():
        raise FileNotFoundError(f"[ERROR] Output root not found: {output_root}")

    candidates = []

    for folder in output_root.iterdir():
        if not folder.is_dir():
            continue

        try:
            t = datetime.strptime(folder.name, "%Y-%m-%d_%H_%M_%S")
            candidates.append((t, folder))
        except:
            continue

    if not candidates:
        raise FileNotFoundError("[ERROR] No valid timestamp folders found in output.")

    candidates.sort(key=lambda x: x[0])
    latest_dir = candidates[-1][1]

    data_path = latest_dir / "data"

    if not data_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing data folder: {data_path}")

    return latest_dir.name, data_path


def get_latest_output_save_path(output_root=Path("./output")):
    """
    Unified data entry for ML pipelines.

    Priority
    --------
    1. MongoDB → evaluation.eval_date
    2. Fallback → local ./output latest folder
    3. If both fail → raise error

    Returns
    -------
    tuple[str, Path]
        latest_time, save_path
    """

    db = None

    # =========================
    # Try MongoDB first
    # =========================
    try:
        db = myDatabase()
        df = db.get_collection("evaluation")

        if df.empty:
            raise ValueError("Empty evaluation collection")

        if "eval_date" not in df.columns:
            raise ValueError("Missing eval_date column")

        df = df.sort_values(by="eval_date")

        latest_time = (
            df["eval_date"]
            .iloc[-1]
            .strftime("%Y-%m-%d_%H_%M_%S")
            .split(".")[0]
        )

        save_path = Path(output_root) / latest_time 

        if not save_path.exists():
            raise FileNotFoundError(f"MongoDB path not found locally: {save_path}")

        print(f"[INFO] Using MongoDB latest data: {save_path}")
        return latest_time, save_path

    except Exception as e:
        print(f"[WARNING] MongoDB failed → fallback local: {e}")

    finally:
        if db is not None:
            db.close()

    # =========================
    # Fallback to local
    # =========================
    latest_time, save_path = _find_latest_local_output(output_root)

    print(f"[INFO] Using local latest data: {save_path}")
    return latest_time, save_path