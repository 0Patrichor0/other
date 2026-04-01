#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nephio PackageVariant cluster selector

Purpose
-------
1. Read a service request JSON/YAML.
2. Query Prometheus for each candidate edge cluster.
3. Hard-filter infeasible clusters.
4. Rank feasible clusters with discounted LinUCB.
5. Update the generated PackageVariant YAML by changing spec.downstream.repo
   to the selected downstream repository.
6. Persist decision context for delayed reward update.

Typical flow
------------
A. Generate PackageVariant from Nephio blueprint/package.
B. Run:
      python3 nephio_bandit_scheduler.py choose \
        --config scheduler_config.yaml \
        --request service_request.yaml \
        --packagevariant packagevariant.yaml
C. Commit/apply the modified PackageVariant YAML.
D. After 15~30 minutes, compute observed reward signals and run:
      python3 nephio_bandit_scheduler.py update \
        --config scheduler_config.yaml \
        --decision-id <id> \
        --observed observed.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from ruamel.yaml import YAML


# -----------------------------
# General helpers
# -----------------------------

def now_ts() -> int:
    return int(time.time())


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or b == 0:
        return default
    return a / b


def die(msg: str, exit_code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(exit_code)


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_yaml_or_json(path: str) -> Dict[str, Any]:
    yaml = YAML(typ="safe")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    if not isinstance(data, dict):
        die(f"{path} must contain a top-level mapping/object")
    return data


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def set_nested(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


# -----------------------------
# Config
# -----------------------------

DEFAULT_FEATURE_NAMES = [
    # Request features
    "req_cpu_norm",
    "req_mem_norm",
    "req_hp_norm",
    "req_vf_norm",
    "replica_norm",
    "is_upf",
    "is_amf",
    "is_smf",
    "latency_strict",
    "latency_medium",
    # Cluster features
    "cpu_headroom_after",
    "mem_headroom_after",
    "hp_headroom_after",
    "vf_headroom_after",
    "cpu_used_norm",
    "mem_used_norm",
    "node_ready_ratio",
    "restart_rate_norm",
    "sched_fail_norm",
    "rtt_norm",
    "jitter_norm",
    # Cross features
    "strict_x_rtt",
    "upf_x_vf_headroom",
    "upf_x_hp_headroom",
]

DEFAULT_PROMQL = {
    "cpu_allocatable_cores": 'sum(kube_node_status_allocatable{resource="cpu",unit="core",cluster="{cluster}"})',
    "mem_allocatable_bytes": 'sum(kube_node_status_allocatable{resource="memory",unit="byte",cluster="{cluster}"})',
    "hugepages_allocatable_bytes": 'sum(kube_node_status_allocatable{resource="hugepages-1Gi",unit="byte",cluster="{cluster}"})',
    # Current/near-real-time usage
    "cpu_used_5m_cores": 'sum(rate(node_cpu_seconds_total{mode!="idle",cluster="{cluster}"}[5m]))',
    "cpu_used_15m_cores": 'sum(rate(node_cpu_seconds_total{mode!="idle",cluster="{cluster}"}[15m]))',
    # Important: this is available memory, not used memory
    "mem_available_bytes": 'sum(node_memory_MemAvailable_bytes{cluster="{cluster}"})',
    "mem_total_bytes": 'sum(node_memory_MemTotal_bytes{cluster="{cluster}"})',
    "node_ready_ratio": 'sum(kube_node_status_condition{condition="Ready",status="true",cluster="{cluster}"}) / count(kube_node_status_condition{condition="Ready",cluster="{cluster}"})',
    "restart_count_15m": 'sum(increase(kube_pod_container_status_restarts_total{cluster="{cluster}"}[15m]))',
    "sched_fail_count_15m": 'sum(increase(scheduler_schedule_attempts_total{result="unschedulable",cluster="{cluster}"}[15m]))',
    # These are often custom/blackbox metrics in real deployments.
    "rtt_ms": 'avg(cluster_rtt_ms{cluster="{cluster}"})',
    "jitter_ms": 'avg(cluster_jitter_ms{cluster="{cluster}"})',
    "vf_free": 'avg(cluster_vf_free{cluster="{cluster}"})',
}

DEFAULT_SERVICE_PROFILES = {
    "upf": {
        "max_rtt_ms": 20,
        "max_jitter_ms": 5,
        "min_node_ready_ratio": 0.95,
        "max_restart_count_15m": 10,
        "required_capabilities": ["multus", "macvlan"],
    },
    "amf": {
        "max_rtt_ms": 50,
        "max_jitter_ms": 10,
        "min_node_ready_ratio": 0.95,
        "max_restart_count_15m": 20,
        "required_capabilities": ["multus"],
    },
    "smf": {
        "max_rtt_ms": 50,
        "max_jitter_ms": 10,
        "min_node_ready_ratio": 0.95,
        "max_restart_count_15m": 20,
        "required_capabilities": ["multus"],
    },
    "obs": {
        "max_rtt_ms": 100,
        "max_jitter_ms": 20,
        "min_node_ready_ratio": 0.90,
        "max_restart_count_15m": 30,
        "required_capabilities": [],
    },
}

DEFAULT_REWARD_WEIGHTS = {
    "ready_success": 0.25,
    "latency_sla_success": 0.25,
    "latency_score": 0.20,
    "jitter_score": 0.10,
    "headroom_after_score": 0.10,
    "stability_score": 0.10,
    "rollback_flag": -0.30,
    "overload_flag": -0.20,
}


@dataclass
class SchedulerConfig:
    prometheus_url: str
    timeout_seconds: int
    alpha: float
    gamma: float
    clusters: List[str]
    repo_by_cluster: Dict[str, str]
    capabilities_by_cluster: Dict[str, List[str]]
    promql: Dict[str, str]
    service_profiles: Dict[str, Dict[str, Any]]
    reward_weights: Dict[str, float]
    safety: Dict[str, float]
    feature_names: List[str]
    state_file: str
    pending_file: str
    patch_metadata_name: bool
    metadata_name_rewrite_mode: str
    downstream_repo_path: str
    packagevariant_name_path: str

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)



def load_config(path: str) -> SchedulerConfig:
    raw = load_yaml_or_json(path)

    prometheus_url = raw.get("prometheus", {}).get("url") or os.environ.get("PROM_URL")
    if not prometheus_url:
        die("config.prometheus.url or PROM_URL is required")

    clusters_cfg = raw.get("clusters")
    if not isinstance(clusters_cfg, list) or not clusters_cfg:
        die("config.clusters must be a non-empty list")

    clusters: List[str] = []
    repo_by_cluster: Dict[str, str] = {}
    capabilities_by_cluster: Dict[str, List[str]] = {}
    for item in clusters_cfg:
        if not isinstance(item, dict) or "name" not in item or "downstream_repo" not in item:
            die("each config.clusters item must include name and downstream_repo")
        name = str(item["name"])
        clusters.append(name)
        repo_by_cluster[name] = str(item["downstream_repo"])
        capabilities_by_cluster[name] = [str(x) for x in item.get("capabilities", [])]

    promql = dict(DEFAULT_PROMQL)
    promql.update(raw.get("promql", {}))

    service_profiles = json.loads(json.dumps(DEFAULT_SERVICE_PROFILES))
    for svc_name, svc_profile in raw.get("service_profiles", {}).items():
        if svc_name not in service_profiles:
            service_profiles[svc_name] = {}
        service_profiles[svc_name].update(svc_profile)

    reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
    reward_weights.update(raw.get("reward_weights", {}))

    safety = {
        "cpu_ratio": 0.15,
        "mem_ratio": 0.10,
        "hugepages_ratio": 0.10,
        "vf_count": 1.0,
    }
    safety.update(raw.get("safety", {}))

    io_cfg = raw.get("io", {})
    yaml_cfg = raw.get("yaml_patch", {})

    return SchedulerConfig(
        prometheus_url=str(prometheus_url).rstrip("/"),
        timeout_seconds=int(raw.get("prometheus", {}).get("timeout_seconds", 10)),
        alpha=float(raw.get("algorithm", {}).get("alpha", 1.2)),
        gamma=float(raw.get("algorithm", {}).get("gamma", 0.98)),
        clusters=clusters,
        repo_by_cluster=repo_by_cluster,
        capabilities_by_cluster=capabilities_by_cluster,
        promql=promql,
        service_profiles=service_profiles,
        reward_weights=reward_weights,
        safety=safety,
        feature_names=list(raw.get("feature_names", DEFAULT_FEATURE_NAMES)),
        state_file=str(io_cfg.get("state_file", "./bandit_state.json")),
        pending_file=str(io_cfg.get("pending_file", "./pending_decisions.json")),
        patch_metadata_name=bool(yaml_cfg.get("patch_metadata_name", True)),
        metadata_name_rewrite_mode=str(yaml_cfg.get("metadata_name_rewrite_mode", "replace_repo_suffix")),
        downstream_repo_path=str(yaml_cfg.get("downstream_repo_path", "spec.downstream.repo")),
        packagevariant_name_path=str(yaml_cfg.get("packagevariant_name_path", "metadata.name")),
    )


# -----------------------------
# Prometheus
# -----------------------------

class PromClient:
    def __init__(self, base_url: str, timeout_seconds: int):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def query_scalar(self, query: str) -> float:
        resp = self.session.get(
            f"{self.base_url}/api/v1/query",
            params={"query": query},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {payload}")
        result = payload.get("data", {}).get("result", [])
        if not result:
            raise LookupError(f"empty Prometheus result for query: {query}")
        return float(result[0]["value"][1])


# -----------------------------
# State
# -----------------------------

def init_model(cfg: SchedulerConfig) -> Dict[str, Any]:
    d = cfg.feature_dim
    return {
        "feature_dim": d,
        "clusters": {
            cluster: {
                "A": np.eye(d).tolist(),
                "b": np.zeros(d).tolist(),
            }
            for cluster in cfg.clusters
        },
    }


def load_model(cfg: SchedulerConfig) -> Dict[str, Any]:
    if not os.path.exists(cfg.state_file):
        model = init_model(cfg)
        save_json(cfg.state_file, model)
        return model
    model = load_json(cfg.state_file, init_model(cfg))
    if int(model.get("feature_dim", -1)) != cfg.feature_dim:
        die(
            f"feature_dim mismatch: config={cfg.feature_dim}, state={model.get('feature_dim')}. "
            f"Delete or migrate {cfg.state_file} first."
        )
    return model


def save_model(cfg: SchedulerConfig, model: Dict[str, Any]) -> None:
    save_json(cfg.state_file, model)


def load_pending(cfg: SchedulerConfig) -> Dict[str, Any]:
    return load_json(cfg.pending_file, {})


def save_pending(cfg: SchedulerConfig, pending: Dict[str, Any]) -> None:
    save_json(cfg.pending_file, pending)


# -----------------------------
# Metrics and filtering
# -----------------------------

def render_promql(template: str, cluster: str) -> str:
    """Render PromQL safely.

    PromQL itself uses braces heavily, so Python str.format() will break on
    expressions like metric{label="value",cluster="{cluster}"}. We only need
    a literal placeholder replacement here.
    """
    return template.replace("{cluster}", cluster)


def get_cluster_metrics(cfg: SchedulerConfig, prom: PromClient, cluster: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, template in cfg.promql.items():
        q = render_promql(template, cluster)
        values[key] = prom.query_scalar(q)

    mem_alloc = values.get("mem_allocatable_bytes", 0.0)
    mem_total = values.get("mem_total_bytes", mem_alloc)
    mem_avail = values.get("mem_available_bytes", 0.0)
    values["mem_used_ratio"] = clamp(1.0 - safe_div(mem_avail, mem_total if mem_total > 0 else mem_alloc, 1.0))
    return values


def get_service_profile(cfg: SchedulerConfig, request: Dict[str, Any]) -> Dict[str, Any]:
    service_class = str(request.get("service_class", "obs"))
    if service_class not in cfg.service_profiles:
        return cfg.service_profiles["obs"]
    return cfg.service_profiles[service_class]


def hard_filter(
    cfg: SchedulerConfig,
    request: Dict[str, Any],
    metrics: Dict[str, float],
    cluster: str,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    profile = get_service_profile(cfg, request)

    req_cpu = float(request.get("req_cpu_cores", 0.0))
    req_mem = float(request.get("req_mem_gib", 0.0)) * 1024 ** 3
    req_hp = float(request.get("req_hugepages_gib", 0.0)) * 1024 ** 3
    req_vf = float(request.get("req_vf", 0.0))

    cpu_alloc = metrics.get("cpu_allocatable_cores", 0.0)
    mem_alloc = metrics.get("mem_allocatable_bytes", 0.0)
    hp_alloc = metrics.get("hugepages_allocatable_bytes", 0.0)

    cpu_free = max(0.0, cpu_alloc - metrics.get("cpu_used_5m_cores", 0.0))
    mem_free = max(0.0, metrics.get("mem_available_bytes", 0.0))
    hp_free = max(0.0, hp_alloc)
    vf_free = max(0.0, metrics.get("vf_free", 0.0))

    cpu_need = req_cpu + cpu_alloc * float(cfg.safety["cpu_ratio"])
    mem_need = req_mem + mem_alloc * float(cfg.safety["mem_ratio"])
    hp_need = req_hp + (hp_alloc * float(cfg.safety["hugepages_ratio"]) if req_hp > 0 else 0.0)
    vf_need = req_vf + (float(cfg.safety["vf_count"]) if req_vf > 0 else 0.0)

    if cpu_free < cpu_need:
        reasons.append("cpu_not_enough")
    if mem_free < mem_need:
        reasons.append("memory_not_enough")
    if hp_free < hp_need:
        reasons.append("hugepages_not_enough")
    if vf_free < vf_need:
        reasons.append("vf_not_enough")

    if metrics.get("node_ready_ratio", 0.0) < float(profile.get("min_node_ready_ratio", 0.95)):
        reasons.append("node_not_healthy")
    if metrics.get("restart_count_15m", 0.0) > float(profile.get("max_restart_count_15m", 10)):
        reasons.append("restart_too_high")

    target_rtt = min(
        float(profile.get("max_rtt_ms", 999999)),
        float(request.get("target_rtt_ms", profile.get("max_rtt_ms", 999999))),
    )
    target_jitter = min(
        float(profile.get("max_jitter_ms", 999999)),
        float(request.get("target_jitter_ms", profile.get("max_jitter_ms", 999999))),
    )
    if metrics.get("rtt_ms", 0.0) > target_rtt:
        reasons.append("rtt_too_high")
    if metrics.get("jitter_ms", 0.0) > target_jitter:
        reasons.append("jitter_too_high")

    required_caps = {str(x) for x in profile.get("required_capabilities", [])}
    required_caps.update(str(x) for x in request.get("required_capabilities", []) or [])
    cluster_caps = set(cfg.capabilities_by_cluster.get(cluster, []))
    missing_caps = sorted(required_caps - cluster_caps)
    if missing_caps:
        reasons.append(f"missing_capabilities:{','.join(missing_caps)}")

    return len(reasons) == 0, reasons


# -----------------------------
# Feature engineering
# -----------------------------

def build_feature_vector(cfg: SchedulerConfig, request: Dict[str, Any], metrics: Dict[str, float]) -> np.ndarray:
    req_cpu = float(request.get("req_cpu_cores", 0.0))
    req_mem_gib = float(request.get("req_mem_gib", 0.0))
    req_hp_gib = float(request.get("req_hugepages_gib", 0.0))
    req_vf = float(request.get("req_vf", 0.0))
    replicas = float(request.get("replica_count", 1.0))

    cpu_alloc = max(metrics.get("cpu_allocatable_cores", 0.0), 1.0)
    mem_alloc = max(metrics.get("mem_allocatable_bytes", 0.0), 1.0)
    hp_alloc = max(metrics.get("hugepages_allocatable_bytes", 0.0), 1.0)
    vf_free = max(metrics.get("vf_free", 0.0), 1.0)

    cpu_free_after = max(0.0, cpu_alloc - metrics.get("cpu_used_5m_cores", 0.0) - req_cpu)
    mem_free_after = max(0.0, metrics.get("mem_available_bytes", 0.0) - req_mem_gib * 1024 ** 3)
    hp_free_after = max(0.0, metrics.get("hugepages_allocatable_bytes", 0.0) - req_hp_gib * 1024 ** 3)
    vf_free_after = max(0.0, metrics.get("vf_free", 0.0) - req_vf)

    cpu_headroom_after = clamp(cpu_free_after / cpu_alloc)
    mem_headroom_after = clamp(mem_free_after / mem_alloc)
    hp_headroom_after = clamp(hp_free_after / hp_alloc)
    vf_headroom_after = clamp(vf_free_after / vf_free)

    service_class = str(request.get("service_class", "obs"))
    latency_class = str(request.get("latency_class", "loose"))

    req_cpu_norm = clamp(req_cpu / 64.0)
    req_mem_norm = clamp(req_mem_gib / 256.0)
    req_hp_norm = clamp(req_hp_gib / 64.0)
    req_vf_norm = clamp(req_vf / 16.0)
    replica_norm = clamp(replicas / 10.0)

    is_upf = 1.0 if service_class == "upf" else 0.0
    is_amf = 1.0 if service_class == "amf" else 0.0
    is_smf = 1.0 if service_class == "smf" else 0.0

    latency_strict = 1.0 if latency_class == "strict" else 0.0
    latency_medium = 1.0 if latency_class == "medium" else 0.0

    cpu_used_norm = clamp(metrics.get("cpu_used_15m_cores", 0.0) / cpu_alloc)
    mem_used_norm = clamp(metrics.get("mem_used_ratio", 0.0))
    node_ready_ratio = clamp(metrics.get("node_ready_ratio", 0.0))
    restart_rate_norm = clamp(metrics.get("restart_count_15m", 0.0) / 100.0)
    sched_fail_norm = clamp(metrics.get("sched_fail_count_15m", 0.0) / 100.0)
    rtt_norm = clamp(metrics.get("rtt_ms", 0.0) / 100.0)
    jitter_norm = clamp(metrics.get("jitter_ms", 0.0) / 20.0)

    strict_x_rtt = latency_strict * rtt_norm
    upf_x_vf_headroom = is_upf * vf_headroom_after
    upf_x_hp_headroom = is_upf * hp_headroom_after

    values = {
        "req_cpu_norm": req_cpu_norm,
        "req_mem_norm": req_mem_norm,
        "req_hp_norm": req_hp_norm,
        "req_vf_norm": req_vf_norm,
        "replica_norm": replica_norm,
        "is_upf": is_upf,
        "is_amf": is_amf,
        "is_smf": is_smf,
        "latency_strict": latency_strict,
        "latency_medium": latency_medium,
        "cpu_headroom_after": cpu_headroom_after,
        "mem_headroom_after": mem_headroom_after,
        "hp_headroom_after": hp_headroom_after,
        "vf_headroom_after": vf_headroom_after,
        "cpu_used_norm": cpu_used_norm,
        "mem_used_norm": mem_used_norm,
        "node_ready_ratio": node_ready_ratio,
        "restart_rate_norm": restart_rate_norm,
        "sched_fail_norm": sched_fail_norm,
        "rtt_norm": rtt_norm,
        "jitter_norm": jitter_norm,
        "strict_x_rtt": strict_x_rtt,
        "upf_x_vf_headroom": upf_x_vf_headroom,
        "upf_x_hp_headroom": upf_x_hp_headroom,
    }

    vec = [float(values[name]) for name in cfg.feature_names]
    return np.array(vec, dtype=float)


# -----------------------------
# LinUCB
# -----------------------------

def linucb_score(cfg: SchedulerConfig, model: Dict[str, Any], cluster: str, x: np.ndarray) -> Dict[str, float]:
    A = np.array(model["clusters"][cluster]["A"], dtype=float)
    b = np.array(model["clusters"][cluster]["b"], dtype=float)
    A_inv = np.linalg.inv(A)
    theta = A_inv @ b
    exploit = float(x @ theta)
    explore = float(cfg.alpha * math.sqrt(max(0.0, x @ A_inv @ x)))
    total = exploit + explore
    return {
        "exploit": exploit,
        "explore": explore,
        "total": total,
    }


def update_model(cfg: SchedulerConfig, model: Dict[str, Any], cluster: str, x: np.ndarray, reward: float) -> None:
    A = np.array(model["clusters"][cluster]["A"], dtype=float)
    b = np.array(model["clusters"][cluster]["b"], dtype=float)

    A = cfg.gamma * A + np.outer(x, x)
    b = cfg.gamma * b + reward * x

    model["clusters"][cluster]["A"] = A.tolist()
    model["clusters"][cluster]["b"] = b.tolist()


# -----------------------------
# PackageVariant patching
# -----------------------------

def rewrite_metadata_name(old_name: str, old_repo: str, new_repo: str, mode: str) -> str:
    if mode == "replace_repo_suffix" and old_repo and old_repo in old_name:
        return old_name.replace(old_repo, new_repo)
    if mode == "append_repo_suffix":
        base = old_name.rsplit("-", 1)[0] if "-" in old_name else old_name
        return f"{base}-{new_repo}"
    return old_name


def patch_packagevariant_yaml(cfg: SchedulerConfig, packagevariant_path: str, selected_cluster: str) -> Dict[str, Any]:
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(packagevariant_path, "r", encoding="utf-8") as f:
        doc = yaml.load(f)

    if not isinstance(doc, dict):
        die(f"{packagevariant_path} is not a YAML object")

    new_repo = cfg.repo_by_cluster[selected_cluster]

    try:
        old_repo = get_nested(doc, cfg.downstream_repo_path)
    except KeyError:
        die(f"cannot find downstream repo path: {cfg.downstream_repo_path}")

    set_nested(doc, cfg.downstream_repo_path, new_repo)

    old_name = None
    new_name = None
    if cfg.patch_metadata_name:
        try:
            old_name = str(get_nested(doc, cfg.packagevariant_name_path))
            new_name = rewrite_metadata_name(old_name, str(old_repo), new_repo, cfg.metadata_name_rewrite_mode)
            set_nested(doc, cfg.packagevariant_name_path, new_name)
        except KeyError:
            pass

    with open(packagevariant_path, "w", encoding="utf-8") as f:
        yaml.dump(doc, f)

    return {
        "old_repo": old_repo,
        "new_repo": new_repo,
        "old_name": old_name,
        "new_name": new_name,
    }


# -----------------------------
# Reward
# -----------------------------

def calc_reward(cfg: SchedulerConfig, observed: Dict[str, Any]) -> float:
    weights = cfg.reward_weights
    total = 0.0
    for key, weight in weights.items():
        total += float(weight) * float(observed.get(key, 0.0))
    return max(-1.0, min(1.0, total))


# -----------------------------
# Commands
# -----------------------------

def cmd_choose(cfg: SchedulerConfig, request_path: str, packagevariant_path: Optional[str], dry_run: bool) -> int:
    request_obj = load_yaml_or_json(request_path)
    decision_id = str(request_obj.get("decision_id") or f"decision-{now_ts()}")
    request_obj["decision_id"] = decision_id

    model = load_model(cfg)
    pending = load_pending(cfg)
    prom = PromClient(cfg.prometheus_url, cfg.timeout_seconds)

    feasible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    print("[INFO] evaluating candidate clusters")
    for cluster in cfg.clusters:
        try:
            metrics = get_cluster_metrics(cfg, prom, cluster)
        except Exception as e:
            rejected.append({
                "cluster": cluster,
                "reasons": [f"metrics_unavailable:{type(e).__name__}"],
            })
            print(f"  - {cluster}: rejected, metrics unavailable ({e})")
            continue

        ok, reasons = hard_filter(cfg, request_obj, metrics, cluster)
        if not ok:
            rejected.append({"cluster": cluster, "reasons": reasons})
            print(f"  - {cluster}: rejected -> {','.join(reasons)}")
            continue

        x = build_feature_vector(cfg, request_obj, metrics)
        scores = linucb_score(cfg, model, cluster, x)
        feasible.append({
            "cluster": cluster,
            "scores": scores,
            "metrics": metrics,
            "x": x.tolist(),
        })
        print(
            f"  - {cluster}: feasible, exploit={scores['exploit']:.4f}, "
            f"explore={scores['explore']:.4f}, total={scores['total']:.4f}"
        )

    if not feasible:
        result = {
            "decision_id": decision_id,
            "target_cluster": None,
            "reason": "no_feasible_cluster",
            "rejected": rejected,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    feasible.sort(key=lambda x: x["scores"]["total"], reverse=True)
    best = feasible[0]
    selected_cluster = str(best["cluster"])

    patch_result = None
    if packagevariant_path and not dry_run:
        patch_result = patch_packagevariant_yaml(cfg, packagevariant_path, selected_cluster)

    pending[decision_id] = {
        "ts": now_ts(),
        "cluster": selected_cluster,
        "downstream_repo": cfg.repo_by_cluster[selected_cluster],
        "x": best["x"],
        "request": request_obj,
        "packagevariant_path": packagevariant_path,
        "patch_result": patch_result,
    }
    save_pending(cfg, pending)

    result = {
        "decision_id": decision_id,
        "target_cluster": selected_cluster,
        "target_downstream_repo": cfg.repo_by_cluster[selected_cluster],
        "top_candidates": [
            {
                "cluster": item["cluster"],
                "total_score": round(float(item["scores"]["total"]), 6),
                "exploit": round(float(item["scores"]["exploit"]), 6),
                "explore": round(float(item["scores"]["explore"]), 6),
            }
            for item in feasible
        ],
        "rejected": rejected,
        "patch_result": patch_result,
        "dry_run": dry_run,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_update(cfg: SchedulerConfig, decision_id: str, observed_path: str) -> int:
    pending = load_pending(cfg)
    if decision_id not in pending:
        die(f"decision_id not found in pending file: {decision_id}")

    observed = load_yaml_or_json(observed_path)
    reward = calc_reward(cfg, observed)

    record = pending[decision_id]
    cluster = str(record["cluster"])
    x = np.array(record["x"], dtype=float)

    model = load_model(cfg)
    update_model(cfg, model, cluster, x, reward)
    save_model(cfg, model)

    del pending[decision_id]
    save_pending(cfg, pending)

    result = {
        "decision_id": decision_id,
        "cluster": cluster,
        "reward": reward,
        "status": "updated",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


# -----------------------------
# CLI
# -----------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nephio PackageVariant smart cluster selector")
    sub = parser.add_subparsers(dest="command", required=True)

    choose = sub.add_parser("choose", help="select a cluster and patch PackageVariant YAML")
    choose.add_argument("--config", required=True, help="scheduler config YAML")
    choose.add_argument("--request", required=True, help="service request YAML/JSON")
    choose.add_argument("--packagevariant", help="generated PackageVariant YAML to patch")
    choose.add_argument("--dry-run", action="store_true", help="do not patch YAML")

    update = sub.add_parser("update", help="apply delayed reward update")
    update.add_argument("--config", required=True, help="scheduler config YAML")
    update.add_argument("--decision-id", required=True, help="decision id to update")
    update.add_argument("--observed", required=True, help="observed reward YAML/JSON")

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.command == "choose":
        return cmd_choose(cfg, args.request, args.packagevariant, args.dry_run)
    if args.command == "update":
        return cmd_update(cfg, args.decision_id, args.observed)
    die(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
