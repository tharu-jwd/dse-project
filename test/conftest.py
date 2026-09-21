"""Top-level conftest, loaded by pytest before test/backend/conftest.py or
any test module. Its only job: if SINHASPEECH_TEST_DB_URL is set, translate
it into the POSTGRES_* environment variables app.core.config.Settings reads,
before anything imports app.db.session (which builds its engine at import
time from those settings). This lets the whole backend suite run against an
isolated scratch database (docker-compose.scratch.yml) instead of the
shared development database - see scripts/test-isolated.sh.

Must not import anything from `app` here: that would build Settings() from
the *unmodified* environment, defeating the point.
"""

import os
from urllib.parse import urlparse

_url = os.environ.get("SINHASPEECH_TEST_DB_URL")

if _url:
    parsed = urlparse(_url)
    os.environ["POSTGRES_HOST"] = parsed.hostname or "127.0.0.1"
    os.environ["POSTGRES_PORT"] = str(parsed.port or 5432)
    os.environ["POSTGRES_USER"] = parsed.username or ""
    os.environ["POSTGRES_PASSWORD"] = parsed.password or ""
    os.environ["POSTGRES_DB"] = parsed.path.lstrip("/")
