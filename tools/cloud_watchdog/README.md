# Cloud Watchdog

A tiny job runner that survives SSH drops and laptop power-offs by
running every job inside a detached `tmux` session on the cloud GPU.

## Components

| script | role |
|---|---|
| `watchdog.sh` | polls `queue/`, launches each new job in a fresh `tmux` session, reaps finished jobs into `done/` or `failed/` |
| `enqueue.sh`  | drops a job spec into `queue/` with the right JSON shape |
| `notify.sh`   | pushes a single ntfy.sh notification (used by the daemon) |
| `heartbeat.sh`| periodically pushes a status summary to your phone |

## One-time setup on the cloud GPU

```bash
ssh ubuntu@192.222.50.225
cd ~/memory_residuals
chmod +x tools/cloud_watchdog/*.sh
sudo apt-get install -y jq tmux curl                                # if missing
mkdir -p tools/cloud_watchdog/{queue,running,done,failed,logs}

# Pick a hard-to-guess topic for ntfy notifications and subscribe to it
# in the ntfy app on your phone.
TOPIC="memres-$(openssl rand -hex 4)"
echo "subscribe to https://ntfy.sh/$TOPIC on your phone"
echo "$TOPIC" > tools/cloud_watchdog/.ntfy_topic

# Start the daemon (detached). Logs go to logs/watchdog.log.
nohup tools/cloud_watchdog/watchdog.sh \
    > tools/cloud_watchdog/logs/watchdog.log 2>&1 &
echo "watchdog PID=$!"

# Optional: push a status heartbeat every 30 min.
nohup tools/cloud_watchdog/heartbeat.sh "$TOPIC" 1800 \
    > tools/cloud_watchdog/logs/heartbeat.log 2>&1 &
```

The daemon is now polling `queue/` every 30 s. Power-cycle the laptop;
nothing on the cloud notices.

## Queueing a job

From your laptop:

```bash
ssh ubuntu@192.222.50.225 \
  '~/memory_residuals/tools/cloud_watchdog/enqueue.sh \
     niah_v3_softparity \
     "python tools/niah_eval.py \
       --model output/chain_v3_softparity_full/best \
       --depths 1,5,10,20,30 \
       --output results/eval/niah_v3_softparity.json" \
     0 \
     "$(cat ~/memory_residuals/tools/cloud_watchdog/.ntfy_topic)"'
```

Inside the cloud GPU you can also do the equivalent shorter form:

```bash
TOPIC=$(cat tools/cloud_watchdog/.ntfy_topic)
tools/cloud_watchdog/enqueue.sh niah_v3_softparity \
  "python tools/niah_eval.py --model output/chain_v3_softparity_full/best ..." \
  0 "$TOPIC"
```

Within ~30 s the daemon picks the spec up, moves it from `queue/` to
`running/`, starts a tmux session named `cwd-niah_v3_softparity`, pushes
a `[start]` ntfy notification, and streams output to
`logs/niah_v3_softparity.log`. When the job ends, the daemon writes
the exit code, moves the spec to `done/` or `failed/`, and pushes a
`[done]` or `[failed]` notification with the last 12 lines of the log.

## Inspecting state

```bash
# How many jobs in each bucket?
for d in queue running done failed; do
  printf '%-8s %s\n' "$d" "$(ls -1 tools/cloud_watchdog/$d 2>/dev/null | wc -l)"
done

# Live tail of a running job
tail -f tools/cloud_watchdog/logs/<job_name>.log

# Attach to a tmux session to see what it's doing
tmux ls
tmux attach -t cwd-<job_name>          # Ctrl-b d to detach again

# Kill a job (terminates the tmux session; daemon notices and moves spec)
tmux kill-session -t cwd-<job_name>
```

## Job spec JSON schema

Anything in `queue/*.json` matching this schema gets picked up:

```json
{
  "name": "niah_v3_softparity",      // required
  "cmd":  "python ...",              // required, full shell command
  "cwd":  "/home/ubuntu/memory_residuals",   // optional (default: $REPO)
  "venv": "/home/ubuntu/venv",       // optional (default: $VENV)
  "gpu":  "0",                       // optional (default: "0")
  "ntfy_topic": "memres-..."         // optional, enables phone notifications
}
```

Convention: queue filenames are `<unix_timestamp>_<name>.json` so the
daemon picks them up in oldest-first order.

## Stopping everything

```bash
ssh ubuntu@192.222.50.225 \
  'pkill -f cloud_watchdog/watchdog.sh; pkill -f cloud_watchdog/heartbeat.sh; tmux kill-server'
```

Anything in `running/` will then be moved to `failed/` on next daemon
start because their tmux sessions no longer exist; that's correct
behaviour.
