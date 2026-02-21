# Testing

For development, you should install all dependencies so that tests have access to them.

```bash
uv sync --all-extras --no-extra dds
```

## Types of tests

There are different types of tests based on what their goal is:

| Type | Description | Mocking | Speed |
|------|-------------|---------|-------|
| Unit | Test a small individual piece of code | All dependencies | Very fast |
| Integration | Test the integration between multiple units of code | Most dependencies | Some fast, some slow |
| Functional | Test a particular desired functionality | Some dependencies | Some fast, some slow |
| End-to-end | Test the entire system as a whole from the perspective of the user | None | Very slow |

The distinction between unit, integration, and functional tests is often debated and rarely productive.

Rather than waste time on classifying tests, it's better to separate tests by how they are used:

* **fast tests**: tests which you can run after each code change (people often run them with filesystem watchers: whenever a file is saved, automatically run the tests)
* **slow tests**: tests which you run every once in a while to make sure you haven't broken anything (maybe every commit, but definitely before publishing a PR)

The purpose of running tests in a loop is to get immediate feedback. The faster the loop, the easier it is to identify a problem since the source is the tiny bit of code you changed.

## Usage

### Fast tests

Run the fast tests:

```bash
./bin/pytest-fast
```

This is the same as:

```bash
pytest dimos
```

The default `addopts` in `pyproject.toml` includes a `-m` filter that excludes slow markers (like `integration`, `heavy`, `e2e`, etc.), so plain `pytest dimos` only runs fast tests.

### Slow tests

Run the slow tests:

```bash
./bin/pytest-slow
```

This overrides the default `-m` filter to include most markers. When writing or debugging a specific slow test, override `-m` yourself:

```bash
pytest -m integration dimos/path/to/test_something.py
```

Note: passing `-m` on the command line overrides the default from `addopts`, so you get exactly the marker set you asked for.

## Writing tests

Test files live next to the code they test. If you have `dimos/core/pubsub.py`, its tests go in `dimos/core/test_pubsub.py`.

When writing tests you probably want to limit the run to whatever tests you're writing:

```bash
pytest -sv dimos/core/test_my_code.py
```

### Fixtures

Pytest fixtures are very useful for making sure test failures don't affect other tests.

Whenever you have something that needs to be cleaned up when the test is over (disconnect, close, delete temp files, etc.), you should use a fixture.

Simple example code:

```python
@pytest.fixture
def arm():
    arm = RobotArm(device="/dev/ttyUSB0")
    arm.connect()
    yield arm
    arm.disconnect()

def test_arm_moves_to_position(arm):
    arm.move_to(x=0.5, y=0.3, z=0.1)
    assert arm.position == (0.5, 0.3, 0.1)
```

The `yield` is key: everything before it is setup, everything after is teardown. The teardown runs even if the test fails, so you never leak resources between tests.

### Mocking

It's easier to use the `mocker` fixture instead of `unittest.mock`. It automatically undoes all patches when the test ends, so you don't need `with` blocks.

Patching a method:

```python
def test_uses_cached_position(mocker):
    mocker.patch("dimos.hardware.RobotArm.get_position", return_value=(0.0, 0.0, 0.0))
    arm = RobotArm()
    assert arm.get_position() == (0.0, 0.0, 0.0)
```

There are other useful things in `mocker`, like `mocker.MagicMock()` for creating fake objects.

## Useful pytest options

| Option | Description |
|--------|-------------|
| `-s` | Show stdout/stderr output |
| `-v` | More verbose test names |
| `-x` | Stop on first failure |
| `-k foo` | Only run tests matching `foo` |
| `--lf` | Rerun only the tests that failed last time |
| `--pdb` | Drop into the debugger when a test fails |
| `--tb=short` | Shorter tracebacks |
| `--durations=0` | Measure the speed of each test |
