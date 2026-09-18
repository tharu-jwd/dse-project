# Deployment Guide — Backend + Model on AWS, Frontend on Vercel

A step-by-step guide written for someone deploying this project for the first time.
Follow it top to bottom.

---

## Part 1 — Understanding what we're building

### The plan

```
┌───────────────────────┐            ┌─────────────────────────────────────┐
│  VERCEL  (free)       │            │  AWS EC2  (one Linux machine)       │
│                       │            │                                     │
│  The React app        │  internet  │  backend  — API + live streaming    │
│  (just HTML/CSS/JS    │ ─────────► │  worker   — transcribes uploads     │
│   files, no server)   │            │  postgres — the database            │
│                       │            │  Caddy    — handles HTTPS           │
│                       │            │                                     │
│                       │            │  models/  — the 1.2 GB Whisper files│
│                       │            │  storage/ — uploaded audio          │
└───────────────────────┘            └─────────────────────────────────────┘
```

**Vercel** hosts the frontend. The frontend is just static files — there's no server-side
code in it. Vercel gives you free hosting, automatic HTTPS, and redeploys whenever you push
to GitHub.

**AWS EC2** is one rented Linux computer. Everything else runs there, using the
`docker-compose.yml` you already have.

### Why split it this way?

The frontend is tiny and static — putting it on Vercel is free and takes 5 minutes. The
backend needs 1.2 GB of model files, ~3 GB of RAM, and must stay running — that needs a real
machine.

### What the four backend pieces do

| Piece | What it does | Why it's separate |
|---|---|---|
| **backend** | Handles API requests and live microphone streaming over WebSockets | Loads the small fast model (238 MB, CTranslate2) at startup |
| **worker** | Transcribes uploaded audio files in the background | A lecture recording takes minutes to transcribe — too long for a web request, so it runs as a separate background process that picks jobs off a queue |
| **postgres** | Stores users, transcripts, quizzes, and the job queue | Standard database |
| **Caddy** | Sits in front and provides HTTPS | Browsers block an https website from calling an http API, so this is required, not optional |

### About the models

You have two model files, and they're used by different processes:

| File | Size | Used by | When it loads |
|---|---|---|---|
| `models/whisper-sinhala1-ct2` | 238 MB | backend | At startup (~70 seconds) |
| `models/whisper-sinhala1` | 925 MB | worker | The first time someone uploads a file |

They're the **same fine-tuned model** in two formats. The CT2 one is compressed and fast,
which is what makes live transcription possible. The other is the original, used for batch
files where speed matters less.

This is why the machine needs ~3 GB of RAM — two separate processes each hold their own
model in memory.

---

## Part 2 — Before you start

You need:

- [ ] An AWS account
- [ ] A GitHub account with this repo pushed to it
- [ ] A Vercel account (sign up with GitHub — it's free)
- [ ] A domain name, **or** a free subdomain from [DuckDNS](https://www.duckdns.org/)
- [ ] The two model folders on your computer (`models/whisper-sinhala1`, `models/whisper-sinhala1-ct2`)

### Important: do NOT put the models in Git

They're 1.2 GB. GitHub rejects files over 100 MB, and it would make every clone painfully
slow. We'll copy them to the server separately in Step 5.

### Roughly what this costs

| Item | Cost |
|---|---|
| EC2 t3.medium | ~$30/month |
| 30 GB disk | ~$2.50/month |
| Public IP address | ~$3.60/month |
| Vercel | Free |
| **Total** | **~$36/month** |

Your AWS free-tier credits should cover the first few months. You can stop the instance when
you're not demoing it and only pay for storage — see "Saving money" at the end.

---

## Part 3 — Launch the server

1. Sign in to the AWS Console. In the top-right, **pick a region close to you** and remember
   it — everything must be created in the same region.

2. Search for **EC2** → **Instances** → **Launch instances**.

3. Fill in:
   - **Name**: `sinhaspeech`
   - **OS**: Ubuntu Server 24.04 LTS
   - **Instance type**: `t3.medium`

   > ⚠️ **Pick `t3.medium`, not `t4g.medium`.** The `t4g` types use ARM chips, and some of
   > this project's Python packages are difficult to build for ARM. `t3` is the normal Intel
   > type and will just work.
   >
   > ⚠️ **Don't pick `t3.micro`** even though it says "Free tier eligible". It has 1 GB of
   > RAM and the backend alone needs 892 MB — it will crash on startup.

4. **Key pair**: click *Create new key pair*, name it `sinhaspeech-key`, type RSA, format
   `.pem`. It downloads automatically. **Keep this file safe — it's the only way to log in,
   and AWS won't give you another copy.**

5. **Network settings** → Edit. Add these rules:

   | Type | Port | Source | Why |
   |---|---|---|---|
   | SSH | 22 | My IP | So you can log in |
   | HTTP | 80 | Anywhere | Caddy needs it to get the HTTPS certificate |
   | HTTPS | 443 | Anywhere | The actual website traffic |

   Leave "Auto-assign public IP" **enabled** — the server needs internet access.

6. **Storage**: change 8 GB to **30 GB**, type gp3.

7. Click **Launch instance**.

### Give it a permanent address

By default the server's IP changes every time it restarts, which would break your domain.

1. EC2 sidebar → **Elastic IPs** → **Allocate Elastic IP address** → Allocate.
2. Select it → **Actions** → **Associate** → choose your `sinhaspeech` instance → Associate.

**Write this IP down.** We'll call it `YOUR_SERVER_IP` from here on.

---

## Part 4 — Point your domain at the server

In your domain provider's DNS settings, add an **A record**:

| Type | Name | Value |
|---|---|---|
| A | `api` | `YOUR_SERVER_IP` |

That makes `api.yourdomain.com` point to your server.

Using DuckDNS instead? Create a subdomain there and set its IP to `YOUR_SERVER_IP`. Your
address will be something like `sinhaspeech.duckdns.org`.

DNS can take a few minutes to update. Check it's working:

```bash
ping api.yourdomain.com
```

It should show `YOUR_SERVER_IP`. Wait until it does before continuing — Caddy can't get an
HTTPS certificate until DNS is correct.

---

## Part 5 — Connect and set up the server

### Log in

On your own computer, in a terminal, go to where your `.pem` file downloaded:

```bash
chmod 400 sinhaspeech-key.pem
ssh -i sinhaspeech-key.pem ubuntu@YOUR_SERVER_IP
```

Type `yes` when it asks about authenticity. You're now controlling the server — everything
from here runs **on the server** unless it says otherwise.

### Install Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu
```

Log out and back in so the group change takes effect:

```bash
exit
ssh -i sinhaspeech-key.pem ubuntu@YOUR_SERVER_IP
```

Check it works:

```bash
docker run hello-world
```

### Add swap space

With only 4 GB of RAM, building the Docker image can run out of memory. Swap is emergency
overflow space on disk that prevents this:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Get the code

```bash
git clone https://github.com/YOUR_USERNAME/dse-project.git
cd dse-project
```

### Copy the models up

**Open a second terminal on your own computer** (leave the server one open). From inside your
project folder:

```bash
scp -i sinhaspeech-key.pem -r models/ ubuntu@YOUR_SERVER_IP:~/dse-project/
```

This uploads 1.2 GB — expect 5–20 minutes depending on your connection.

Back **on the server**, confirm they arrived:

```bash
du -sh ~/dse-project/models/*
```

You should see roughly `925M` and `238M`.

---

## Part 6 — Configure the app

### Create the settings file

On the server, in `~/dse-project`:

```bash
nano .env
```

Paste this, replacing the marked values:

```bash
# Database
POSTGRES_DB=sinhaspeech
POSTGRES_USER=sinhaspeech_app
POSTGRES_PASSWORD=PUT_A_LONG_RANDOM_PASSWORD_HERE
POSTGRES_HOST=database
POSTGRES_PORT=5432

# Security
JWT_SECRET_KEY=PUT_A_LONG_RANDOM_SECRET_HERE
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Your Vercel address goes here - update it after Part 9
CORS_ORIGINS=https://your-app.vercel.app

# Storage
MEDIA_STORAGE_DIR=storage/uploads
MAX_UPLOAD_SIZE_BYTES=104857600

# Transcription model
TRANSCRIBER_BACKEND=whisper
WHISPER_MODEL=models/whisper-sinhala1
WHISPER_LANGUAGE=si
WHISPER_BASE_MODEL=openai/whisper-small
WHISPER_ADAPTER_MODEL=SPEAK-ASR/whisper-si-exp-10

# Live streaming + voice commands
STREAMING_ENABLED=true
VOICE_COMMAND_EMBEDDING_MATCHING_ENABLED=true
```

Save with `Ctrl+O`, `Enter`, then `Ctrl+X`.

Generate the two secrets by running this twice, and paste the results in:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(48))"
```

> ⚠️ Never commit this file to Git. It contains your database password and signing key.

### The production compose file

Good news — `docker-compose.prod.yml` is **already in the repo**, so there's nothing to type.
It's your normal `docker-compose.yml` with two changes: the frontend service is removed
(Vercel handles that), and Caddy is added for HTTPS.

For reference, this is what it contains:

```yaml
services:
  database:
    image: postgres:17-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER -d $$POSTGRES_DB"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  backend:
    build: ./backend
    env_file: .env
    environment:
      POSTGRES_HOST: database
      MEDIA_STORAGE_DIR: /app/storage/uploads
      VOICE_SAMPLES_DIR: /app/storage/voice_samples
      STREAMING_SOURCE_MODEL: /app/models/whisper-sinhala1
      STREAMING_MODEL_PATH: /app/models/whisper-sinhala1-ct2
    volumes:
      - ./models:/app/models:ro
      - ./storage:/app/storage
    depends_on:
      database:
        condition: service_healthy
    restart: unless-stopped

  worker:
    build: ./backend
    command: python -m scripts.run_transcription_worker
    env_file: .env
    environment:
      POSTGRES_HOST: database
      MEDIA_STORAGE_DIR: /app/storage/uploads
      VOICE_SAMPLES_DIR: /app/storage/voice_samples
      STREAMING_SOURCE_MODEL: /app/models/whisper-sinhala1
      STREAMING_MODEL_PATH: /app/models/whisper-sinhala1-ct2
    volumes:
      - ./models:/app/models:ro
      - ./storage:/app/storage
    depends_on:
      database:
        condition: service_healthy
    restart: unless-stopped

  caddy:
    image: caddy:2-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile:ro
      - caddy_data:/data
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  postgres_data:
  caddy_data:
```

Note the backend no longer publishes port 8000 to the internet — only Caddy is exposed, and
it forwards traffic internally. That's deliberate and safer.

### Create the Caddy config

A template is already in the repo. Copy it and put your own domain in:

```bash
cp Caddyfile.example Caddyfile
nano Caddyfile
```

The whole file is just:

```
api.yourdomain.com {
    reverse_proxy backend:8000
}
```

Replace with your real domain. Caddy gets and renews the HTTPS certificate automatically, and
forwards WebSockets without extra configuration.

(`Caddyfile` itself is gitignored, since the domain differs per server — `Caddyfile.example`
is the version kept in the repo.)

---

## Part 7 — Start everything

```bash
cd ~/dse-project
docker compose -f docker-compose.prod.yml up -d --build
```

The first build takes **10–20 minutes** (it downloads PyTorch, which is large). Watch it with:

```bash
docker compose -f docker-compose.prod.yml logs -f
```

Wait for `Streaming models loaded.` followed by `Application startup complete.` — the model
load takes about 70 seconds, which is normal. Press `Ctrl+C` to stop watching (this does not
stop the server).

### Set up the database tables

```bash
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

Optionally, create the demo accounts:

```bash
docker compose -f docker-compose.prod.yml exec backend python -m scripts.seed_users
```

### Check it's alive

```bash
curl https://api.yourdomain.com/health
```

Expected: `{"status":"healthy"}`

Now open `https://api.yourdomain.com/docs` in a browser. You should see the API documentation
page with a padlock icon. **If you see this, the backend is fully deployed.**

---

## Part 8 — Upload your voice command recordings

If you want the voice commands working, copy your recordings up. From **your own computer**:

```bash
scp -i sinhaspeech-key.pem -r storage/voice_samples/ ubuntu@YOUR_SERVER_IP:~/dse-project/storage/
```

Then on the server, import them for your user:

```bash
docker compose -f docker-compose.prod.yml exec backend \
  python -m scripts.reimport_voice_enrollment student@sinhaspeech.lk \
  --commands next,previous,delete,submit,save,stop --language si
```

---

## Part 9 — Deploy the frontend to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub.
2. **Add New → Project** → import your repository.
3. Configure:
   - **Root Directory**: click Edit, select `frontend`
   - **Framework Preset**: Vite (usually auto-detected)
4. Expand **Environment Variables** and add:

   | Name | Value |
   |---|---|
   | `VITE_API_BASE_URL` | `https://api.yourdomain.com` |
   | `VITE_USE_MOCK_API` | `false` |

5. Click **Deploy**. You'll get a URL like `https://your-app.vercel.app`.

### Connect the two together

Back **on the server**, put your real Vercel URL into the settings:

```bash
cd ~/dse-project
nano .env
# change CORS_ORIGINS to your actual Vercel URL, e.g.
# CORS_ORIGINS=https://your-app.vercel.app
```

Restart so it takes effect:

```bash
docker compose -f docker-compose.prod.yml up -d
```

> ⚠️ **`VITE_API_BASE_URL` is baked in when Vercel builds the site.** If you change it later,
> changing the variable is not enough — you must redeploy in Vercel for it to take effect.

---

## Part 10 — Check everything works

Open your Vercel URL and confirm:

- [ ] The login page loads
- [ ] You can log in
- [ ] The dashboard loads with data (this proves the API connection works)
- [ ] Recording a note works (this proves WebSockets work)
- [ ] Voice commands respond (this proves the models loaded)
- [ ] Uploading a file eventually produces a transcript (this proves the worker works)

---

## Troubleshooting

### The website loads but nothing works / login fails

Open the browser console (F12). Look at the errors:

**"blocked by CORS policy"** — `CORS_ORIGINS` on the server doesn't exactly match your Vercel
URL. It must match exactly: `https://` included, no trailing slash. Fix `.env` and restart.

**Requests going to `localhost:8000`** — the frontend was built with the wrong API URL. Fix
`VITE_API_BASE_URL` in Vercel and **redeploy** there.

**"Mixed Content" blocked** — your `VITE_API_BASE_URL` starts with `http://` instead of
`https://`.

### `curl https://api.yourdomain.com/health` fails

Check Caddy got its certificate:

```bash
docker compose -f docker-compose.prod.yml logs caddy
```

Common causes:
- DNS isn't pointing at the server yet — check with `ping api.yourdomain.com`
- Port 80 isn't open in the AWS security group (Caddy needs it to verify the certificate)

### The backend keeps restarting

```bash
docker compose -f docker-compose.prod.yml logs backend
```

- **`Killed` / exit code 137** — out of memory. Check with `free -m`. Make sure you added
  swap, and that you're on `t3.medium` not `t3.micro`.
- **Can't connect to database** — check `POSTGRES_HOST=database` in `.env` (the service name,
  not an IP).
- **Model file not found** — check `du -sh ~/dse-project/models/*` shows both folders.

### Uploads never finish transcribing

The worker is the piece that does this:

```bash
docker compose -f docker-compose.prod.yml logs worker
```

The worker loads its 925 MB model on the *first* job, so the first upload is always slow. If
it's getting OOM-killed, see "Saving memory" below.

### Useful commands

```bash
# See what's running
docker compose -f docker-compose.prod.yml ps

# Watch logs live
docker compose -f docker-compose.prod.yml logs -f backend

# Restart everything
docker compose -f docker-compose.prod.yml restart

# Check memory
free -m
```

---

## Updating after a code change

```bash
cd ~/dse-project
git pull
docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec backend alembic upgrade head
```

The frontend updates by itself whenever you push to GitHub — that's Vercel's doing.

---

## Saving memory

If the 4 GB machine is struggling, in order of preference:

1. **Turn off the worker.** It's ~2 GB, and only needed for uploaded-file transcription. Live
   recording and voice commands keep working without it:
   ```bash
   docker compose -f docker-compose.prod.yml stop worker
   ```
2. **Move the database to RDS.** Frees ~250 MB, and `db.t4g.micro` is free for 12 months.
3. **Upgrade to `t3.large`** (8 GB, ~$60/month).

## Saving money

Stop the instance when you're not using it — EC2 → select instance → Instance state → Stop.
You stop paying for compute, and keep paying only for storage (a few dollars a month). Start
it again before a demo; your Elastic IP stays the same.

> ⚠️ Use **Stop**, never **Terminate**. Terminate permanently deletes the machine and
> everything on it.

---

## Security notes for a real deployment

The setup above is appropriate for a project demo. Before real users:

- Restrict SSH (port 22) to your IP only — not `0.0.0.0/0`
- Keep `.env` out of Git (check your `.gitignore`)
- Back up the database regularly: `docker compose -f docker-compose.prod.yml exec database pg_dump -U sinhaspeech_app sinhaspeech > backup.sql`
- Apply system updates: `sudo apt update && sudo apt upgrade -y`
