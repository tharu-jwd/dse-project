# Running the remaining tests on the live EC2 instance

The automated suites cover everything that can be verified without the real server.
Three things cannot, and this is how to do them by hand:

1. **Hardware facts** — what the instance actually is, and whether the inference thread
   count needs overriding (§A.2 finding 3 in `TESTING_REPORT.md`).
2. **Load testing** — the concurrency ceiling on real hardware. The figure in the report
   (1–2 concurrent streaming users) came from a CPU-capped container on a laptop and is a
   floor, not a prediction.
3. **Failover** — that the containers return after a real instance reboot. `restart:
   unless-stopped` is configured and has never been tested against an actual reboot.

Each part is independent. Part 1 is read-only. Parts 2 and 3 write data and cause downtime
respectively, so both start with a backup.

Throughout: `$SERVER` is the instance, `$APP` is the public API base URL.

```bash
export SERVER=ubuntu@54.86.239.120          # or: sinhaspeech (your ssh config host)
export APP=https://sinhaspeech.duckdns.org
```

---

## Part 0 — Back up first

Do this before Part 2 or Part 3. Part 2 writes real rows; Part 3 takes the service down.

```bash
ssh $SERVER
cd ~/dse-project
source .env    # POSTGRES_USER / POSTGRES_DB

docker compose -f docker-compose.prod.yml exec -T database \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > ~/backup-$(date +%F-%H%M).sql

ls -lh ~/backup-*.sql        # sanity: should be more than a few KB
```

Copy it off the instance, because a backup that only exists on the machine you are about to
reboot is not a backup:

```bash
# from your laptop
scp $SERVER:~/backup-*.sql ./
```

> The restore path itself is already proven — `test/failover/test_failover.py` restores a
> `pg_dump` into a scratch database and compares every table. What has never been proven is a
> restore of *this* data on *this* host.

---

## Part 1 — Hardware facts (read-only, safe any time)

```bash
ssh $SERVER '
  echo "cores:  $(nproc)"
  echo "arch:   $(uname -m)"
  free -g | head -2
  TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
  echo -n "type:   "; curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/instance-type; echo
  echo "--- containers ---"
  docker ps --format "{{.Names}}\t{{.Status}}"
'
```

Then the one that decides whether a config change is needed:

```bash
ssh $SERVER 'cd ~/dse-project && docker compose -f docker-compose.prod.yml exec -T backend \
  python -c "
import multiprocessing
print(\"cores the container sees:\", multiprocessing.cpu_count())
print(\"cpu quota:\", open(\"/sys/fs/cgroup/cpu.max\").read().strip())
"'
```

**How to read it:**

| Quota line | Meaning | Action |
|---|---|---|
| `max 100000` | No CPU limit | Nothing to do. CTranslate2 uses one thread per core, which is correct. |
| e.g. `200000 100000` | Limited to 2 cores | Cores-seen will exceed that. Set `STREAMING_CPU_THREADS` to the quota (here `2`) in `.env` and restart the backend — measured 2.8x faster inference. |

`docker-compose.prod.yml` sets no `cpus:` limit, so the first row is expected. Confirm rather
than assume.

**Also worth recording:** if `uname -m` says `aarch64` you are on ARM (Graviton). That
contradicts `DEPLOYMENT.md`, which steers away from ARM over Python packaging. If it works,
that guidance is stale and the doc should be corrected.

---

## Part 2 — Load testing

### Where to run it

**From your laptop, not the server.** A load generator on a 2-core box competes with the
server for the exact resource being measured. Note that your network latency to `us-east-1`
(~250 ms from Sri Lanka) is included in the figures — honest for real users, but it is not
server time.

### The wake word problem

Production has `voice_command_wake_required = True`, and the only clips in
`storage/voice_samples/` are bare commands (`delete`, `next`, `previous`, `save`, `stop`,
`submit`). The server will correctly ignore every one of them, and the `ws speech->verdict`
metric — the one that matters — will simply time out.

Pick one:

**(a) Measure the real production path.** Record a few clips of yourself saying
"zimi <command>" at 16 kHz mono, put them in a folder, and point `LOAD_WAV_DIR` at it. Most
faithful, and needs no server change.

```bash
# convert anything to the required format
ffmpeg -i raw.m4a -ar 16000 -ac 1 -sample_fmt s16 zimi_next_1.wav
```

**(b) Turn the gate off temporarily.** Faster, but it is a production config change and must
be put back.

```bash
ssh $SERVER
cd ~/dse-project
cp .env .env.bak                                   # so restoring is not from memory
echo 'VOICE_COMMAND_WAKE_REQUIRED=false' >> .env
docker compose -f docker-compose.prod.yml up -d backend
# ~70s for the model to load
until curl -sf $APP/health >/dev/null; do sleep 5; done
```

**Restoring it is not optional** — leaving the gate off means any stray speech can trigger a
command:

```bash
ssh $SERVER 'cd ~/dse-project && mv .env.bak .env && \
  docker compose -f docker-compose.prod.yml up -d backend'
# then verify:
ssh $SERVER 'cd ~/dse-project && docker compose -f docker-compose.prod.yml exec -T backend \
  python -c "from app.core.config import settings; print(settings.voice_command_wake_required)"'
# must print True
```

### The sweep

Watch the server in a second terminal while it runs:

```bash
ssh $SERVER 'docker stats'
```

Then, from your laptop, one user at a time, stepping up:

```bash
cd ~/Documents/GIt_Repos/dse-project

for u in 1 2 3 5 8; do
  echo "===== $u concurrent streaming users ====="
  LOAD_EMAIL=student@sinhaspeech.lk LOAD_PASSWORD=demo123 \
  locust -f test/load/locustfile.py --host $APP \
         --headless -u $u -r 1 -t 3m --only-summary StreamingUser
done
```

For the REST path (cheap, no audio), swap `StreamingUser` for `ApiUser`.

### What to look for

- **`ws speech->verdict`** — time from the end of speech to the command verdict. This is the
  number that decides whether voice control feels usable. Above ~5 s it feels broken.
- **The user count where the failure rate leaves 0%.** That is your ceiling.
- **`docker stats`** — if the backend sits pinned at `N00%` CPU, you are inference-bound,
  which is expected. If memory climbs steadily and never returns, that is the untested leak
  in §3.1.4 and worth capturing.
- **`restarts`** — must stay constant. Any increase means the backend crashed:
  `docker inspect dse-project-backend-1 --format '{{.RestartCount}}'`

### Clean up the rows it created

Streaming sessions create real transcripts. Remove them afterwards:

```bash
ssh $SERVER 'cd ~/dse-project && source .env && \
  docker compose -f docker-compose.prod.yml exec -T database \
  psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c \
  "select count(*) from transcripts where created_at > now() - interval '"'"'2 hours'"'"';"'
```

Inspect before deleting, then remove by title or timestamp as appropriate. Do not blanket-delete
by time window without looking — a real user may have been working.

---

## Part 3 — Reboot and recovery

Do Part 0 first. The service will be down for a minute or two.

```bash
# Baseline
ssh $SERVER 'docker ps --format "{{.Names}}\t{{.Status}}"'
curl -s $APP/health

# RebootTask	Tool/Framework	Videos/References																							
- Manual Testing - Web, Desktop & Mobile	-																								
																									
Unit Testing 	JUnit (Java)	Java Unit Testing with JUnit - Tutorial - How to Create And Use Unit Tests - YouTube																							
	TestNG (Java)	Tutorial 1: TestNG with intelliJ IDE | Introduction Advantages of TestNG | Setup Environment																							
	Pytest (Flask/FastApi/ Django)	Python Unit Testing | FastAPI with Pytest Tutorial (fast & easy) - YouTube																							
		PyTest • REST API Integration Testing with Python - YouTube																							
	Jest (Express. js)	Express JS #20 - Unit Testing with Jest - YouTube																							
	React Testing Libaray / Jest (React. js)	React Testing Tutorial - YouTube																							
		React Testing for Beginners: Start Here! - YouTube																							
	sonarQube (Code Quality)	How To Sonarqube Setup From Scratch And Code Analysis (2024) - YouTube																							
	Jasmine (Pure Java Script)	How To Sonarqube Setup From Scratch And Code Analysis (2024) - YouTube																							
	Firebase	Intro to Firebase Cloud Function - Setup, Emulator, and Unit Testing - Typescript																							
API Testing	Postman	Postman Beginner's Course - API Testing																							
		Postman Api Testing Tutorial for beginners - YouTube																							
Test Automation	Copilot	Writing API Tests Faster With AI and Github Copilot (Tutorial) | Serenity Dojo TV - YouTube																							
	Selenium	Selenium Course for Beginners - Web Scraping Bots, Browser Automation, Testing (Tutorial) - YouTube																							
		Selenium Automation Testing Tutorial | Selenium Tutorial For Beginners | Selenium| Simplilearn - YouTube																							
Browser automation.functional tests	Selenium	Selenium Course for Beginners - Web Scraping Bots, Browser Automation, Testing (Tutorial) - YouTube																							
Integration testing 	selenium (UI) /rest assured(API)																								
Basic Performance Testing	Jmeter	"Ultimate Guide to JMeter Performance Testing ,

 JMeter Full Course Masterclass | Step by Step for Beginners | Raghav Pal | - YouTube"																							
	HP load runner	"HP LoadRunner Guide,

Learn HP Loadrunner for Beginners - Full Course"																							
	Gatling	"Beginner's Guide to Gatling

Gatling Load Testing - Ultimate Crash Course Tutorial For Beginners - YouTube"																							
	BlazeMeter	"BlazeMeter Load Testing

BlazeMeter Performance Testing"																							
Basic Accessibility Testing	JAWS	"JAWS to Evaluate Web Accessibility

How to Quickly Test Your Website for JAWS Screen Reader Compatibility"																							
	color-contrast-checker	"A Guide on Contrast Checker

How to test for Color Contrast | TPGI & WebAIM | WCAG - YouTube"																							
	Axe DevTools	"Axe Dev tools Guide 

Getting Started with the axe DevTools Browser Extension - YouTube"																							
	Google Lighthouse	"A Beginner's Guide to Lighthouse

Improve Your Website's Performance with Google Lighthouse - YouTube"																							
Atlassian tools	Jira	"Jira - Comprehensive Beginner's Guide

Jira Training | Jira Tutorial for Beginners | Jira Course | Intellipaat"																							
	Confluence	"The Ultimate Confluence Guide for Beginners

Confluence Tutorial for Beginners: 1+ Hour Confluence Training Course - YouTube"																							
	ClickUp	"Clickup Beginner's Guide

How to use ClickUp for Beginners - YouTube"																							
Database	MongoDB	Testing a REST API with Mocha & Chai - 05 - Test DB routes	Using with MongoDB · Jest																						
	FireBase	Test in the lab, not on your users	Getting started with Firebase Test Lab on Android (Firecasts)	Firebase Test Lab: using Robo Tests (Firecasts)	Getting started with Firebase Test Lab on Android (Firecasts)																				
	SQL	SQL Queries and Database Testing - Learn Basics in 10 minutes!	Database Testing																						
Deployment 	Jenkins	Learn Jenkins! Complete Jenkins Course - Zero to Hero	Build+Deploy+Test with Jenkins 2.0																						
Analyzing Error Logs	 KubeCTL	What on earth is kubectl?	Understanding KUBECTL - Learning Kubernetes																						
	AWScloudWatch	AWS Cloudwatch Guides - Learn AWS Monitoring Techniques																							
Version Control - GitLab (Gitbash, Sourcetree)	Git	What is version control | Atlassian Git Tutorial	13 Advanced (but useful) Git Techniques and Shortcuts	Git and GitHub for Beginners - Crash Course																					
	Sourcetree	Sourcetree																							
	GitLab	GitLab Tutorial For Beginners | What Is GitLab And How To Use It? | GitLab Tutorial | Simplilearn	GitLab CI CD Tutorial for Beginners [Crash Course]																						
- Master Automation Control Dashboard																									
- Test Management -Application Life cycle Management																									
Monitoring Tools 	NewRelic	Advance New Relic Tutorial from Basic																							
	PagerDuty	Introduction to PagerDuty																							
GUI/ front end testing 	Cypress	Cypress Beginner Tutorials																							
Load testing 	HP Load Runner	Load Runner Tutorials																							
	Jmeter	JMeter Load Testing | Load Testing Using JMmeter | JMeter Tutorial For Beginners | Simplilearn																							
Security testing	checkMarx	CheckMarx																							
	OWASP	OWASP API Security Top 10 Course – Secure Your Web Apps																							
																									
																									
																									
																									
																									
																									
																									
																									
																									
																													
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																									
																														
ssh $SERVER 'sudo reboot'

# Wait for it to come back (ssh drops immediately; this is expected)
until ssh -o ConnectTimeout=5 -o BatchMode=yes $SERVER 'true' 2>/dev/null; do sleep 10; done
echo "ssh is back"

# The backend needs ~70s to load the model, so poll rather than checking once
until curl -sf $APP/health >/dev/null 2>&1; do sleep 10; done
echo "API is back"
```

**Pass criteria — all four must hold:**

```bash
ssh $SERVER 'docker ps --format "{{.Names}}\t{{.Status}}"'   # database, backend, worker, caddy
curl -s $APP/health                                          # {"status":"healthy"}
curl -s -o /dev/null -w "%{http_code}\n" $APP/health/database # 200
```

Then confirm the data survived and the app genuinely works, not just that it answers:

```bash
SMOKE_API_URL=$APP SMOKE_APP_URL=https://sinhaspeech.vercel.app \
pytest test/deployment/ -v
```

**If a container does not return,** that is the finding — `restart: unless-stopped` is not
sufficient on its own. The usual cause is Docker itself not being enabled at boot:

```bash
ssh $SERVER 'systemctl is-enabled docker'     # should say "enabled"
ssh $SERVER 'sudo systemctl enable docker'    # fix if it does not
```

---

## Part 4 — Optional: performance figures

**Voice-command latency**, using the script already in the repo:

```bash
ssh $SERVER 'cd ~/dse-project && docker compose -f docker-compose.prod.yml exec -T backend \
  python -m scripts.benchmark_command_latency'
```

**Memory growth over a long session** — the untested item in §3.1.4. Start this, use the app
normally for an hour, then look at the trend:

```bash
ssh $SERVER 'while true; do \
  echo "$(date +%H:%M:%S) $(docker stats --no-stream --format "{{.Name}} {{.MemUsage}}" \
    | tr "\n" " ")"; sleep 60; done' | tee ~/mem-trend.log
```

Flat or sawtooth is fine. A steady climb that never comes down is a leak.

---

## When you are done

- [ ] `.env` restored and `voice_command_wake_required` confirmed `True`
- [ ] Test transcripts removed
- [ ] Backup copied off the instance
- [ ] Any temporary SSH security-group rule removed
- [ ] Numbers recorded — paste the Locust summaries back and the report's §3.1.5 and
      Appendix A.3 can be updated with real figures instead of the scratch-stack estimates

---

## What this cannot tell you

- **Anything about other browsers or devices.** That is the e2e matrix, which runs locally.
- **Transcription quality under load.** The harness measures whether responses arrive and
  when, not whether the text is right.
- **Multi-worker behaviour.** `_active_sessions` is per-process, so the session cap and the
  slot-leak guarantees hold only while streaming runs on a single worker.
