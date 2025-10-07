import pytest

from openposttraining.integrations import DatasetteIntegration


def test_datasette_integration_init(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    assert datasette.db_path == db_path
    assert db_path.parent.exists()


def test_datasette_list_tables(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    tables = datasette.list_tables()
    assert isinstance(tables, list)


def test_datasette_insert_prompt(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    record_id = datasette.insert_prompt(
        prompt="Test prompt",
        model="gpt2",
        response="Test response",
        metadata={"key": "value"},
    )
    assert isinstance(record_id, str)


def test_datasette_insert_run(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    record_id = datasette.insert_run(
        command="opt status",
        status="running",
    )
    assert isinstance(record_id, str)


def test_datasette_update_run(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    run_id = datasette.insert_run("test command")
    datasette.update_run(run_id, status="completed", duration=1.5)


def test_datasette_insert_metric(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    run_id = datasette.insert_run("test command")
    metric_id = datasette.insert_metric(
        run_id=run_id,
        metric_name="latency",
        metric_value=10.5,
        metadata={"unit": "ms"},
    )
    assert isinstance(metric_id, str)


def test_datasette_query(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    datasette.insert_prompt("test", "gpt2", "response")
    results = datasette.query("SELECT * FROM prompts LIMIT 1")
    assert isinstance(results, list)


def test_datasette_export_table_json(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    datasette.insert_prompt("test", "gpt2", "response")
    data = datasette.export_table("prompts", format="json")
    assert isinstance(data, list)


def test_datasette_export_table_csv(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    datasette.insert_prompt("test", "gpt2", "response")
    data = datasette.export_table("prompts", format="csv")
    assert isinstance(data, str)


def test_datasette_export_nonexistent_table(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    result = datasette.export_table("nonexistent")
    assert isinstance(result, (list, dict))


def test_datasette_serve(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    result = datasette.serve(port=9001)
    assert isinstance(result, str)


def test_datasette_query_empty_db(tmp_path):
    db_path = tmp_path / "test.db"
    datasette = DatasetteIntegration(db_path=db_path)
    results = datasette.query("SELECT 1 as test")
    assert isinstance(results, list)

