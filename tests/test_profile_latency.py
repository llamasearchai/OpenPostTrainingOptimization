from openposttraining.profiling.performance_analyzer import profile_entry


def test_profile_latency(tmp_path):
    out = tmp_path / "profile.json"
    res = profile_entry("sshleifer/tiny-gpt2", metrics=["latency"], device="cpu", output_path=str(out), iters=1)
    assert "latency" in res
    assert "mean" in res["latency"]
