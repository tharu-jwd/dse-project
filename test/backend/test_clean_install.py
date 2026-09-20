"""Clean-install and compose validation (Task Gap README Task 9). Two
things were previously only ever exercised by hand: migrations run
against a genuinely empty database (not the shared dev database, which
has never been empty since it was created), and the compose files
themselves. Both `pytest test/backend` locally and CI's Postgres
service container start from real migrations, but neither ever adds
the demo seed, so "does a fresh checkout + seed produce a working
login" had never actually been checked end to end.

Everything here skips (not fails) without Docker - see ground rule 3 in
TEST_GAPS_README.md.
"""

import shutil
import subprocess
import time

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker is not available in this environment"
)

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]
COMPOSE_PROJECT = "dse-clean-install-test"
DB_CONTAINER = f"{COMPOSE_PROJECT}-database-1"
DB_PORT = 5434


def _docker_available() -> bool:
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="module")
def _clean_database():
    if not _docker_available():
        pytest.skip("Docker daemon is not reachable")

    subprocess.run(
        [
            "docker", "run", "-d",
            "--name", DB_CONTAINER,
            "-e", "POSTGRES_DB=clean_install",
            "-e", "POSTGRES_USER=clean_install",
            "-e", "POSTGRES_PASSWORD=clean_install",
            "-p", f"127.0.0.1:{DB_PORT}:5432",
            "postgres:17-alpine",
        ],
        check=True,
        capture_output=True,
    )

    try:
        for _ in range(30):
            ready = subprocess.run(
                ["docker", "exec", DB_CONTAINER, "pg_isready", "-U", "clean_install", "-d", "clean_install"],
                capture_output=True,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            pytest.fail("clean database never became ready")

        yield {
            "POSTGRES_HOST": "127.0.0.1",
            "POSTGRES_PORT": str(DB_PORT),
            "POSTGRES_DB": "clean_install",
            "POSTGRES_USER": "clean_install",
            "POSTGRES_PASSWORD": "clean_install",
        }
    finally:
        subprocess.run(["docker", "rm", "-f", DB_CONTAINER], capture_output=True)


def _run_in_backend_image(env: dict, *command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "docker", "run", "--rm", "--network", "host",
            *[arg for key, value in env.items() for arg in ("-e", f"{key}={value}")],
            "-e", "JWT_SECRET_KEY=clean-install-test-not-for-prod",
            "-v", f"{REPO_ROOT}:/repo",
            "-w", "/repo/backend",
            "dse-project-backend:latest",
            *command,
        ],
        capture_output=True,
        text=True,
    )


def test_migrations_then_seed_then_demo_login_works_on_a_brand_new_database(_clean_database):
    if subprocess.run(["docker", "image", "inspect", "dse-project-backend:latest"], capture_output=True).returncode != 0:
        pytest.skip("dse-project-backend:latest image is not built locally")

    migrate = _run_in_backend_image(_clean_database, "alembic", "upgrade", "head")
    assert migrate.returncode == 0, migrate.stdout + migrate.stderr

    seed = _run_in_backend_image(_clean_database, "python3", "-m", "scripts.seed_users")
    assert seed.returncode == 0, seed.stdout + seed.stderr

    login_check = _run_in_backend_image(
        _clean_database,
        "python3", "-c",
        "from starlette.testclient import TestClient; from app.main import app;"
        "c = TestClient(app);"
        "r = c.post('/auth/login', json={'email': 'student@sinhaspeech.lk', 'password': 'demo123'});"
        "assert r.status_code == 200, (r.status_code, r.text);"
        "assert 'token' in r.json();"
        "print('demo login OK')",
    )
    assert login_check.returncode == 0, login_check.stdout + login_check.stderr
    assert "demo login OK" in login_check.stdout


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.scratch.yml"])
def test_compose_file_is_syntactically_valid(compose_file):
    if not _docker_available():
        pytest.skip("Docker daemon is not reachable")

    path = REPO_ROOT / compose_file
    if not path.exists():
        pytest.skip(f"{compose_file} does not exist")

    # `config` only parses and validates; it starts nothing. A dummy .env
    # (via env vars) satisfies files that require POSTGRES_* / JWT_SECRET_KEY
    # to interpolate, since compose config fails on undefined required vars.
    result = subprocess.run(
        ["docker", "compose", "-f", str(path), "config", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={
            "PATH": __import__("os").environ.get("PATH", ""),
            "POSTGRES_DB": "x", "POSTGRES_USER": "x", "POSTGRES_PASSWORD": "x", "POSTGRES_PORT": "5432",
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_env_var_referenced_by_a_compose_file_exists_in_env_example():
    import re

    env_example = (REPO_ROOT / ".env.example").read_text()
    declared = set(re.findall(r"^([A-Z_][A-Z0-9_]*)=", env_example, re.MULTILINE))

    referenced: set[str] = set()
    for compose_file in ["docker-compose.yml", "docker-compose.prod.yml", "docker-compose.scratch.yml"]:
        path = REPO_ROOT / compose_file
        if not path.exists():
            continue
        text = path.read_text()
        referenced |= set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", text))

    missing = referenced - declared
    assert not missing, (
        f"compose files reference {missing} but .env.example never declares them - "
        "a fresh checkout following .env.example would fail to start"
    )
