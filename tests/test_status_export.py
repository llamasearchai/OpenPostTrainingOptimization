import json
from pathlib import Path

from openposttraining.cli import build_parser


def test_status_export_tmp(tmp_path):
    # Run the CLI parser and handler directly to avoid spawning subprocess
    parser = build_parser()
    args = parser.parse_args(["status", "--device", "cpu", "-e", str(tmp_path / "status.json")])
    rc = args.func(args)
    assert rc == 0
    data = json.loads(Path(tmp_path / "status.json").read_text())
    assert "backend" in data
