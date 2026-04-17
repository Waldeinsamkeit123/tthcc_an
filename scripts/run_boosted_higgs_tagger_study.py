#!/usr/bin/env python3

import importlib.machinery
import importlib.util
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _load_main():
    try:
        from tthcc_an.boosted_higgs_tagger_study import main as imported_main

        return imported_main
    except ModuleNotFoundError as exc:
        if exc.name != "tthcc_an.boosted_higgs_tagger_study":
            raise

        pycache_dir = SRC_ROOT / "tthcc_an" / "__pycache__"
        version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        candidates = sorted(pycache_dir.glob("boosted_higgs_tagger_study.cpython-*.pyc"))
        preferred = [path for path in candidates if version_tag in path.name]
        chosen = preferred[0] if preferred else (candidates[0] if candidates else None)
        if chosen is None:
            raise

        module_name = "tthcc_an.boosted_higgs_tagger_study"
        loader = importlib.machinery.SourcelessFileLoader(module_name, str(chosen))
        spec = importlib.util.spec_from_file_location(module_name, chosen, loader=loader)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.main


main = _load_main()


if __name__ == "__main__":
    main()
