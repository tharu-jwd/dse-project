#!/usr/bin/env bash
# Runs the full backend suite against a disposable scratch database
# (docker-compose.scratch.yml) instead of the shared dev database, so
# it can run cleanly with no .env and an empty local Postgres. See
# test/conftest.py for how SINHASPEECH_TEST_DB_URL takes effect, and
# docs/EC2_TEST_RUNBOOK.md for the same pattern run against a live host.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="docker compose -f docker-compose.scratch.yml"
DB_URL="postgresql://scratch:scratch@127.0.0.1:5433/scratch"

cleanup() {
  echo "[test-isolated] tearing down scratch stack"
  $COMPOSE down -v >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[test-isolated] starting scratch database"
$COMPOSE up -d database

echo "[test-isolated] waiting for it to become healthy"
until docker exec dse-scratch-database-1 pg_isready -U scratch -d scratch >/dev/null 2>&1; do
  sleep 1
done

echo "[test-isolated] running migrations"
# alembic runs outside pytest, so it never sees test/conftest.py's
# SINHASPEECH_TEST_DB_URL translation - point it at the scratch DB directly.
docker run --rm --network host \
  -e POSTGRES_HOST=127.0.0.1 -e POSTGRES_PORT=5433 -e POSTGRES_DB=scratch \
  -e POSTGRES_USER=scratch -e POSTGRES_PASSWORD=scratch \
  -e JWT_SECRET_KEY=test-isolated-not-for-prod \
  -v "$PWD:/repo" -w /repo/backend \
  dse-project-backend:latest alembic upgrade head

echo "[test-isolated] running pytest test/backend"
docker run --rm --network host \
  -e SINHASPEECH_TEST_DB_URL="$DB_URL" \
  -e JWT_SECRET_KEY=test-isolated-not-for-prod \
  -v "$PWD:/repo" -w /repo \
  dse-project-backend:latest pytest test/backend -v
