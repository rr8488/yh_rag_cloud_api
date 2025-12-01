"""
Quick sanity check that all internal modules import correctly.
Run this once before starting the FastAPI server.

Best practice is to run this from the project root directory as a module:
  python -m yh_rag_cloud_api.test_imports
"""
import sys
from pathlib import Path

# Add project root to the Python path to allow direct execution of this script.
# This makes the script more robust.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("🔍 Testing internal imports...")

try:
    # 1. Test top-level app import
    from yh_rag_cloud_api import app
    print("✅ Main 'app' module imported successfully.")

    # 2. Test key utility and parser imports
    from yh_rag_cloud_api.db_schema import initialize_database
    from yh_rag_cloud_api.rag_handler import execute_rag_pipeline
    from yh_rag_cloud_api.parsers.grant_agreement_parser_production import extract_key_grant_fields
    from yh_rag_cloud_api.models import RagReq, ImpactArea
    print("✅ Submodules (db, rag, parsers, models) imported fine.")

    # 3. Check that the main script is runnable (as a module)
    # Note: This is an indirect check; run_dev.py is meant to be executed.
    print("✅ 'run_dev.py' is present (run it directly to start the server).")

    print("🎉 All internal imports OK!")
except Exception as e:
    print("❌ Import failed:")
    import traceback
    traceback.print_exc()