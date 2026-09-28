#!/usr/bin/env bash
# Runs image/discover.sh against mentatd status taken from the four GLM boxes
# (fixtures/mentat, addresses moved to the documentation ranges), with
# mentat_query answering from the fixtures and sleep stubbed out. Needs only
# bash and python3. Exit status is the number of failures.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
disc="$here/../../image/discover.sh"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fails=0

check() {
  if [[ "$2" == "$3" ]]; then echo "  ok   $1"
  else echo "  FAIL $1: got '$2', want '$3'"; fails=$(( fails + 1 )); fi
}

# One box's boot as far as the entrypoint takes discovery, in a clean
# environment. $1 is the fixture directory, $2 the box (the last octet of its
# address), the rest VAR=value settings from .env. Both waits are bounded, so
# a broken parse ends in a failure while sleep is stubbed. Prints
# VLLM_HOST_IP|FABRIC_SUBNETS|CLUSTER_SUBNET|MENTAT_NODE_IP|HEAD_HOST|ROLE|queries
# on stdout, and the entrypoint's own messages on stderr.
boot() {
  local fx=$1 node=$2; shift 2
  : > "$tmp/queries"
  env -i PATH="$PATH" DISC="$disc" FX="$fx" NODE="$node" QLOG="$tmp/queries" \
      TP=4 MENTAT_GROUP=glm53 RAY_ADDRESS=127.0.0.1:6379 \
      FABRIC_WAIT_S=600 WORKER_WAIT_S=600 "$@" bash -c '
    set -euo pipefail
    . "$DISC"
    mentat_query() {
      echo "$1" >> "$QLOG"
      case "$1" in
        local) cat "$FX/local-$NODE.json" 2>/dev/null ;;
        group) cat "$FX/group-glm53.json" 2>/dev/null ;;
      esac
    }
    sleep() { :; }
    # As entrypoint.sh does, in order.
    _elect=""
    [[ -z "${ROLE:-}" && -z "${HEAD_HOST:-}" ]] && _elect=1
    discover_network >&2
    CLUSTER_SUBNET="${CLUSTER_SUBNET:-${FABRIC_SUBNETS%% *}}"
    if [[ -n "$_elect" ]]; then elect_head >&2; else fixed_role; fi
    echo "${VLLM_HOST_IP:-}|${FABRIC_SUBNETS:-}|${CLUSTER_SUBNET:-}|${MENTAT_NODE_IP:-}|${HEAD_HOST:-}|${ROLE:-}|$(sort -u "$QLOG" | tr "\n" " ")"
  '
}

# A copy of the fixtures with $2 (python, `local` and `group` as dicts keyed
# by box) applied, in directory $1.
variant() {
  mkdir -p "$1"
  python3 - "$here/fixtures/mentat" "$1" "$2" <<'PY'
import json, os, sys
src, dst, edit = sys.argv[1:]
local = {n: json.load(open(f"{src}/local-{n}.json")) for n in ("36", "70", "77", "93")}
group = json.load(open(f"{src}/group-glm53.json"))
agents = group["groups"]["glm53"]["agents"]
exec(edit)
for n, d in local.items():
    json.dump(d, open(f"{dst}/local-{n}.json", "w"))
json.dump(group, open(f"{dst}/group-glm53.json", "w"))
PY
}

echo "== all four boxes, as the fleet tags them today (one rdma root)"
for n in 36 70 77 93; do
  role=worker; [[ $n == 36 ]] && role=head
  check "box .$n" "$(boot "$here/fixtures/mentat" $n 2>/dev/null)" \
    "192.0.2.$n|198.18.0.|198.18.0.|192.0.2.$n|192.0.2.36|$role|group local "
done
boot "$here/fixtures/mentat" 70 2>&1 >/dev/null | grep '^election:' | sed 's/^/       /'

echo "== both PCIe roots tagged rdma, listed out of order"
variant "$tmp/roots" '
for n, d in local.items():
    a = f"198.19.0.{n}"
    d["addrs"].insert(1, a)
    d["addr_tags"][a] = ["connectx", "rdma"]
    d["addr_ifaces"][a] = "enP2p1s0f1np1"'
for n in 36 93; do
  check "box .$n prefixes" "$(boot "$tmp/roots" $n 2>/dev/null | cut -d'|' -f2,3)" \
    "198.18.0. 198.19.0.|198.18.0."
done

echo "== nothing tagged lan: VLLM_HOST_IP falls back to the daemon's node_ip"
variant "$tmp/nolan" 'local["70"]["addr_tags"]["192.0.2.70"] = []'
check "box .70" "$(boot "$tmp/nolan" 70 2>/dev/null | cut -d'|' -f1)" "192.0.2.70"

echo "== a gone agent and a register-only agent below the lowest box"
variant "$tmp/stale" '
agents["glm53@glm53@192.0.2.5"] = dict(agents["glm53@glm53@192.0.2.36"],
    node_ip="192.0.2.5", alive=False, gone_since_ms=1790624000000)
agents["glm53@glm53-api@192.0.2.4"] = dict(agents["glm53@glm53@192.0.2.36"],
    node_ip="192.0.2.4", machine={"cpus": 20, "gpus": [], "memory": 0})'
for n in 36 77; do
  role=worker; [[ $n == 36 ]] && role=head
  check "box .$n" "$(boot "$tmp/stale" $n 2>/dev/null | cut -d'|' -f5,6)" "192.0.2.36|$role"
done

echo "== fewer than TP live agents: keeps waiting, then WORKER_WAIT_S ends it"
variant "$tmp/three" 'agents["glm53@glm53@192.0.2.77"]["alive"] = False'
set +e
out=$(boot "$tmp/three" 70 WORKER_WAIT_S=65 2>&1); rc=$?
set -e
check "exit status" "$rc" 1
check "waited" "$(grep -c '^electing: waiting for 4 agents in group glm53, have 3' <<< "$out")" 2
check "reason" "$(grep -o 'FATAL: only 3 of 4 agents.*' <<< "$out")" \
  "FATAL: only 3 of 4 agents in group glm53 after 65s (192.0.2.36 192.0.2.70 192.0.2.93)"

echo "== mentatd not answering, or no rdma address: FABRIC_WAIT_S ends the wait"
set +e
out=$(boot "$tmp/none" 70 FABRIC_WAIT_S=30 2>&1); rc=$?
set -e
check "no daemon" "$rc $(grep -c '^FATAL: no answer from mentatd' <<< "$out")" "1 1"
variant "$tmp/nordma" 'local["70"]["addr_tags"]["198.18.0.1"] = ["connectx"]'
set +e
out=$(boot "$tmp/nordma" 70 FABRIC_WAIT_S=30 2>&1); rc=$?
set -e
check "no rdma" "$rc $(grep -c '^FATAL: mentatd lists no address tagged rdma' <<< "$out")" "1 1"

echo "== explicit settings skip discovery"
check "all four set" \
  "$(boot /nonexistent 70 ROLE=worker HEAD_HOST=192.0.2.93 VLLM_HOST_IP=192.0.2.70 CLUSTER_SUBNET=10.0.0. 2>/dev/null)" \
  "192.0.2.70||10.0.0.||192.0.2.93|worker|"
check "HEAD_HOST alone: a box that is not HEAD_HOST works" \
  "$(boot /nonexistent 70 HEAD_HOST=192.0.2.93 VLLM_HOST_IP=192.0.2.70 CLUSTER_SUBNET=10.0.0. 2>/dev/null | cut -d'|' -f5-)" \
  "192.0.2.93|worker|"
check "HEAD_HOST alone: the box at HEAD_HOST leads" \
  "$(boot /nonexistent 93 HEAD_HOST=192.0.2.93 VLLM_HOST_IP=192.0.2.93 CLUSTER_SUBNET=10.0.0. 2>/dev/null | cut -d'|' -f5-)" \
  "192.0.2.93|head|"
check "roles set, networking from mentat" \
  "$(boot "$here/fixtures/mentat" 70 ROLE=worker HEAD_HOST=192.0.2.93 2>/dev/null)" \
  "192.0.2.70|198.18.0.|198.18.0.||192.0.2.93|worker|local "
check "networking set, roles elected" \
  "$(boot "$here/fixtures/mentat" 93 VLLM_HOST_IP=192.0.2.93 CLUSTER_SUBNET=10.0.0. 2>/dev/null)" \
  "192.0.2.93||10.0.0.|192.0.2.93|192.0.2.36|worker|group local "
check "FABRIC_SUBNETS alone names CLUSTER_SUBNET" \
  "$(boot /nonexistent 70 ROLE=worker HEAD_HOST=192.0.2.93 VLLM_HOST_IP=192.0.2.70 "FABRIC_SUBNETS=10.0.0. 10.0.1." 2>/dev/null)" \
  "192.0.2.70|10.0.0. 10.0.1.|10.0.0.||192.0.2.93|worker|"

echo "$fails failure(s)"
exit "$fails"
