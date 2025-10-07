from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetteIntegration:
    """Integration with Datasette and sqlite-utils for data management."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize Datasette integration.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or Path.cwd() / "data" / "openposttraining.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema."""
        try:
            import sqlite_utils

            db = sqlite_utils.Database(self.db_path)

            # Prompts table
            if "prompts" not in db.table_names():
                db["prompts"].create(
                    {
                        "id": str,
                        "prompt": str,
                        "model": str,
                        "response": str,
                        "timestamp": str,
                        "metadata": str,
                    },
                    pk="id",
                )

            # Runs table
            if "runs" not in db.table_names():
                db["runs"].create(
                    {
                        "id": str,
                        "command": str,
                        "status": str,
                        "start_time": str,
                        "end_time": str,
                        "duration_seconds": float,
                        "output": str,
                        "error": str,
                    },
                    pk="id",
                )

            # Traces table
            if "traces" not in db.table_names():
                db["traces"].create(
                    {
                        "id": str,
                        "run_id": str,
                        "operation": str,
                        "timestamp": str,
                        "duration_ms": float,
                        "metadata": str,
                    },
                    pk="id",
                )

            # Datasets table
            if "datasets" not in db.table_names():
                db["datasets"].create(
                    {
                        "id": str,
                        "name": str,
                        "description": str,
                        "size_bytes": int,
                        "num_samples": int,
                        "created_at": str,
                        "metadata": str,
                    },
                    pk="id",
                )

            # Metrics table
            if "metrics" not in db.table_names():
                db["metrics"].create(
                    {
                        "id": str,
                        "run_id": str,
                        "metric_name": str,
                        "metric_value": float,
                        "timestamp": str,
                        "metadata": str,
                    },
                    pk="id",
                )

            # Models table
            if "models" not in db.table_names():
                db["models"].create(
                    {
                        "id": str,
                        "name": str,
                        "backend": str,
                        "size_gb": float,
                        "quantized": int,
                        "quantization_method": str,
                        "created_at": str,
                        "metadata": str,
                    },
                    pk="id",
                )

        except ImportError:
            pass

    def insert_prompt(
        self,
        prompt: str,
        model: str,
        response: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a prompt record.

        Args:
            prompt: Prompt text
            model: Model name
            response: Model response
            metadata: Additional metadata

        Returns:
            Record ID
        """
        try:
            import sqlite_utils
            import uuid

            db = sqlite_utils.Database(self.db_path)
            record_id = str(uuid.uuid4())
            db["prompts"].insert(
                {
                    "id": record_id,
                    "prompt": prompt,
                    "model": model,
                    "response": response,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "metadata": json.dumps(metadata or {}),
                }
            )
            return record_id
        except ImportError:
            return "error-sqlite-utils-not-installed"

    def insert_run(
        self,
        command: str,
        status: str = "running",
        output: str = "",
        error: str = "",
    ) -> str:
        """Insert a run record.

        Args:
            command: Command executed
            status: Run status
            output: Command output
            error: Error message if any

        Returns:
            Record ID
        """
        try:
            import sqlite_utils
            import uuid

            db = sqlite_utils.Database(self.db_path)
            record_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()
            db["runs"].insert(
                {
                    "id": record_id,
                    "command": command,
                    "status": status,
                    "start_time": now,
                    "end_time": now,
                    "duration_seconds": 0.0,
                    "output": output,
                    "error": error,
                }
            )
            return record_id
        except ImportError:
            return "error-sqlite-utils-not-installed"

    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        duration: Optional[float] = None,
    ) -> None:
        """Update a run record.

        Args:
            run_id: Run ID to update
            status: New status
            output: Command output
            error: Error message
            duration: Duration in seconds
        """
        try:
            import sqlite_utils

            db = sqlite_utils.Database(self.db_path)
            updates = {"end_time": datetime.now(UTC).isoformat()}
            if status:
                updates["status"] = status
            if output:
                updates["output"] = output
            if error:
                updates["error"] = error
            if duration is not None:
                updates["duration_seconds"] = duration
            db["runs"].update(run_id, updates)
        except ImportError:
            pass

    def insert_metric(
        self,
        run_id: str,
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Insert a metric record.

        Args:
            run_id: Associated run ID
            metric_name: Name of the metric
            metric_value: Metric value
            metadata: Additional metadata

        Returns:
            Record ID
        """
        try:
            import sqlite_utils
            import uuid

            db = sqlite_utils.Database(self.db_path)
            record_id = str(uuid.uuid4())
            db["metrics"].insert(
                {
                    "id": record_id,
                    "run_id": run_id,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "metadata": json.dumps(metadata or {}),
                }
            )
            return record_id
        except ImportError:
            return "error-sqlite-utils-not-installed"

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a SQL query.

        Args:
            sql: SQL query string

        Returns:
            Query results as list of dictionaries
        """
        try:
            import sqlite_utils

            db = sqlite_utils.Database(self.db_path)
            return list(db.query(sql))
        except ImportError:
            return [{"error": "sqlite-utils not installed"}]

    def list_tables(self) -> List[str]:
        """List all tables in the database.

        Returns:
            List of table names
        """
        try:
            import sqlite_utils

            db = sqlite_utils.Database(self.db_path)
            return db.table_names()
        except ImportError:
            return []

    def serve(self, port: int = 9000, host: str = "localhost") -> str:
        """Start Datasette server.

        Args:
            port: Port to serve on
            host: Host to bind to

        Returns:
            Status message
        """
        try:
            cmd = [
                "datasette",
                "serve",
                str(self.db_path),
                "-p",
                str(port),
                "-h",
                host,
                "--cors",
            ]
            subprocess.Popen(cmd)
            return f"Datasette server started at http://{host}:{port}"
        except FileNotFoundError:
            return "Error: datasette not found. Install with: pip install datasette"

    def export_table(self, table_name: str, format: str = "json") -> Any:
        """Export a table to specified format.

        Args:
            table_name: Name of table to export
            format: Export format (json, csv, etc.)

        Returns:
            Exported data
        """
        try:
            import sqlite_utils

            db = sqlite_utils.Database(self.db_path)
            if table_name not in db.table_names():
                return {"error": f"Table {table_name} not found"}

            rows = list(db[table_name].rows)
            if format == "json":
                return rows
            elif format == "csv":
                import csv
                import io

                output = io.StringIO()
                if rows:
                    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                return output.getvalue()
            else:
                return {"error": f"Unsupported format: {format}"}
        except ImportError:
            return {"error": "sqlite-utils not installed"}

