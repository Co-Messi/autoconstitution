# autoconstitution Security Architecture

## Executive Summary

This document outlines the comprehensive security model for autoconstitution, a multi-agent AI system designed for automated experimentation. The architecture implements defense-in-depth principles with multiple layers of protection including sandboxing, agent isolation, resource controls, code validation, and comprehensive audit logging.

---

## 1. Threat Model

### 1.1 Threat Actors

| Actor | Motivation | Capabilities |
|-------|------------|--------------|
| Malicious Agent | Self-preservation, resource acquisition | Code generation, network access, persistence |
| Compromised Agent | Hijacked by external attacker | All agent capabilities |
| Insider Threat | Data exfiltration, sabotage | System access, configuration knowledge |
| Supply Chain | Backdoor injection | Dependency poisoning, model tampering |

### 1.2 Attack Vectors

1. **Code Injection**: Agents generating malicious code
2. **Privilege Escalation**: Escaping sandbox constraints
3. **Resource Exhaustion**: Denial of service via resource abuse
4. **Data Exfiltration**: Unauthorized data access/transmission
5. **Lateral Movement**: Agent-to-agent compromise
6. **Persistence**: Establishing backdoors in the system

### 1.3 Security Objectives

- **Confidentiality**: Protect sensitive data from unauthorized access
- **Integrity**: Ensure experiment results and system state remain untampered
- **Availability**: Maintain system operation under attack
- **Accountability**: Full traceability of all actions

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SWARMRESEARCH SECURITY ZONES                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │
│  │   Control   │    │  Experiment │    │  Execution  │    │   Network   │   │
│  │    Plane    │◄──►│    Store    │◄──►│  Sandboxes  │◄──►│   Proxy     │   │
│  │             │    │             │    │             │    │             │   │
│  │  - Orchestrator│  │  - Code repo │   │  - Containers│   │  - Egress    │   │
│  │  - Scheduler   │  │  - Results   │   │  - VMs       │   │  - Filtering │   │
│  │  - Policy Mgr  │  │  - Artifacts │   │  - gVisor    │   │  - Monitoring│   │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └─────────────┘   │
│         │                  │                  │                             │
│         └──────────────────┴──────────────────┘                             │
│                            │                                                │
│                    ┌───────┴───────┐                                        │
│                    │  Audit & Log  │                                        │
│                    │    System     │                                        │
│                    └───────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Sandboxing for Code Execution

### 3.1 Multi-Layer Sandboxing Strategy

```
Layer 1: Container Isolation (Docker/Podman)
    ↓
Layer 2: System Call Filtering (seccomp-bpf)
    ↓
Layer 3: Resource Constraints (cgroups v2)
    ↓
Layer 4: Optional: VM-level (Kata Containers/gVisor)
```

### 3.2 Container Configuration

```yaml
# sandbox-config.yaml
apiVersion: v1
kind: Pod
metadata:
  name: agent-sandbox
  annotations:
    seccomp.security.alpha.kubernetes.io/pod: runtime/default
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
    appArmorProfile:
      type: RuntimeDefault
  containers:
  - name: experiment-runner
    image: autoconstitution/sandbox:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
        add:
        - NET_BIND_SERVICE  # Minimal required
    resources:
      limits:
        cpu: "2"
        memory: "4Gi"
        ephemeral-storage: "10Gi"
      requests:
        cpu: "100m"
        memory: "256Mi"
    volumeMounts:
    - name: tmp
      mountPath: /tmp
    - name: workspace
      mountPath: /workspace
      readOnly: true
  volumes:
  - name: tmp
    emptyDir:
      sizeLimit: 1Gi
  - name: workspace
    persistentVolumeClaim:
      claimName: experiment-data
```

### 3.3 Seccomp Profile

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86"],
  "syscalls": [
    {
      "names": [
        "accept", "accept4", "access", "adjtimex", "alarm", "bind",
        "brk", "capget", "capset", "chdir", "chmod", "chown", "chown32",
        "clock_adjtime", "clock_adjtime64", "clock_getres", "clock_gettime",
        "clock_gettime64", "clock_nanosleep", "clock_nanosleep_time64",
        "clone", "clone3", "close", "close_range", "connect", "copy_file_range",
        "creat", "dup", "dup2", "dup3", "epoll_create", "epoll_create1",
        "epoll_ctl", "epoll_ctl_old", "epoll_pwait", "epoll_pwait2",
        "epoll_wait", "epoll_wait_old", "eventfd", "eventfd2", "execve",
        "execveat", "exit", "exit_group", "faccessat", "faccessat2",
        "fadvise64", "fadvise64_64", "fallocate", "fanotify_mark", "fchdir",
        "fchmod", "fchmodat", "fchown", "fchown32", "fchownat", "fcntl",
        "fcntl64", "fdatasync", "fgetxattr", "flistxattr", "flock",
        "fork", "fremovexattr", "fsetxattr", "fstat", "fstat64", "fstatat64",
        "fstatfs", "fstatfs64", "fsync", "ftruncate", "ftruncate64",
        "futex", "futex_time64", "getcpu", "getcwd", "getdents", "getdents64",
        "getegid", "getegid32", "geteuid", "geteuid32", "getgid", "getgid32",
        "getgroups", "getgroups32", "getitimer", "getpeername", "getpgid",
        "getpgrp", "getpid", "getppid", "getpriority", "getrandom",
        "getresgid", "getresgid32", "getresuid", "getresuid32", "getrlimit",
        "get_robust_list", "getrusage", "getsid", "getsockname", "getsockopt",
        "get_thread_area", "gettid", "gettimeofday", "getuid", "getuid32",
        "getxattr", "inotify_add_watch", "inotify_init", "inotify_init1",
        "inotify_rm_watch", "io_cancel", "ioctl", "io_destroy", "io_getevents",
        "io_pgetevents", "io_pgetevents_time64", "ioprio_get", "ioprio_set",
        "io_setup", "io_submit", "io_uring_enter", "io_uring_register",
        "io_uring_setup", "kill", "lchown", "lchown32", "lgetxattr",
        "link", "linkat", "listen", "listxattr", "llistxattr", "lremovexattr",
        "lseek", "lsetxattr", "lstat", "lstat64", "madvise", "membarrier",
        "memfd_create", "mincore", "mkdir", "mkdirat", "mknod", "mknodat",
        "mlock", "mlock2", "mlockall", "mmap", "mmap2", "mprotect", "mq_getsetattr",
        "mq_notify", "mq_open", "mq_timedreceive", "mq_timedreceive_time64",
        "mq_timedsend", "mq_timedsend_time64", "mq_unlink", "mremap", "msgctl",
        "msgget", "msgrcv", "msgsnd", "msync", "munlock", "munlockall",
        "munmap", "nanosleep", "newfstatat", "open", "openat", "openat2",
        "pause", "pidfd_open", "pidfd_send_signal", "pipe", "pipe2", "pivot_root",
        "poll", "ppoll", "ppoll_time64", "prctl", "pread64", "preadv",
        "preadv2", "prlimit64", "pselect6", "pselect6_time64", "pwrite64",
        "pwritev", "pwritev2", "read", "readahead", "readdir", "readlink",
        "readlinkat", "readv", "recv", "recvfrom", "recvmmsg", "recvmmsg_time64",
        "recvmsg", "remap_file_pages", "removexattr", "rename", "renameat",
        "renameat2", "restart_syscall", "rmdir", "rseq", "rt_sigaction",
        "rt_sigpending", "rt_sigprocmask", "rt_sigqueueinfo", "rt_sigreturn",
        "rt_sigsuspend", "rt_sigtimedwait", "rt_sigtimedwait_time64",
        "rt_tgsigqueueinfo", "sched_getaffinity", "sched_getattr",
        "sched_getparam", "sched_get_priority_max", "sched_get_priority_min",
        "sched_getscheduler", "sched_rr_get_interval", "sched_rr_get_interval_time64",
        "sched_setaffinity", "sched_setattr", "sched_setparam",
        "sched_setscheduler", "sched_yield", "seccomp", "select", "semctl",
        "semget", "semop", "semtimedop", "semtimedop_time64", "send",
        "sendfile", "sendfile64", "sendmmsg", "sendmsg", "sendto", "setfsgid",
        "setfsgid32", "setfsuid", "setfsuid32", "setgid", "setgid32",
        "setgroups", "setgroups32", "setitimer", "setpgid", "setpriority",
        "setregid", "setregid32", "setresgid", "setresgid32", "setresuid",
        "setresuid32", "setreuid", "setreuid32", "setrlimit", "set_robust_list",
        "setsid", "setsockopt", "set_thread_area", "set_tid_address",
        "setuid", "setuid32", "setxattr", "shmat", "shmctl", "shmdt",
        "shmget", "shutdown", "sigaltstack", "signalfd", "signalfd4",
        "sigpending", "sigprocmask", "sigreturn", "socket", "socketcall",
        "socketpair", "splice", "stat", "stat64", "statfs", "statfs64",
        "statx", "symlink", "symlinkat", "sync", "sync_file_range",
        "syncfs", "sysinfo", "tee", "tgkill", "time", "timer_create",
        "timer_delete", "timer_getoverrun", "timer_gettime", "timer_gettime64",
        "timer_settime", "timer_settime64", "timerfd_create", "timerfd_gettime",
        "timerfd_gettime64", "timerfd_settime", "timerfd_settime64", "times",
        "tkill", "truncate", "truncate64", "ugetrlimit", "umask", "uname",
        "unlink", "unlinkat", "utime", "utimensat", "utimensat_time64",
        "utimes", "vfork", "wait4", "waitid", "waitpid", "write", "writev"
      ],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["personality"],
      "action": "SCMP_ACT_ALLOW",
      "args": [
        {
          "index": 0,
          "value": 0,
          "op": "SCMP_CMP_EQ"
        },
        {
          "index": 0,
          "value": 8,
          "op": "SCMP_CMP_EQ"
        },
        {
          "index": 0,
          "value": 131072,
          "op": "SCMP_CMP_EQ"
        },
        {
          "index": 0,
          "value": 131073,
          "op": "SCMP_CMP_EQ"
        },
        {
          "index": 0,
          "value": 4294967295,
          "op": "SCMP_CMP_EQ"
        }
      ]
    }
  ]
}
```

### 3.4 gVisor Integration (High-Risk Workloads)

```yaml
# gvisor-runtime.yaml
apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: gvisor
handler: runsc
scheduling:
  nodeSelector:
    sandbox.gvisor.io/enabled: "true"
---
# High-risk experiment pod
apiVersion: v1
kind: Pod
metadata:
  name: high-risk-experiment
spec:
  runtimeClassName: gvisor
  containers:
  - name: experiment
    image: experiment-image:latest
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      runAsNonRoot: true
```

### 3.5 Sandbox Lifecycle

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CREATE    │────►│   VALIDATE  │────►│   EXECUTE   │────►   DESTROY   │
│             │     │             │     │             │     │             │
│ - Pull image│     │ - Code scan │     │ - Run exp   │     │ - Kill proc │
│ - Init net  │     │ - Sign check│     │ - Monitor   │     │ - Wipe fs   │
│ - Mount vol │     │ - Policy chk│     │ - Log all   │     │ - Audit log │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 4. Agent Isolation

### 4.1 Isolation Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              HOST SYSTEM                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        CONTROL PLANE                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Agent A    │  │  Agent B    │  │  Agent C    │  │  Agent D   │ │   │
│  │  │  (Research) │  │  (Analysis) │  │  (Codegen)  │  │  (Review)  │ │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘ │   │
│  │         │                │                │               │        │   │
│  │         └────────────────┴────────────────┴───────────────┘        │   │
│  │                              │                                      │   │
│  │                    ┌─────────┴─────────┐                           │   │
│  │                    │  Message Broker   │                           │   │
│  │                    │  (Filtered/MTLS)  │                           │   │
│  │                    └─────────┬─────────┘                           │   │
│  └──────────────────────────────┼─────────────────────────────────────┘   │
│                                 │                                          │
│  ┌──────────────────────────────┼─────────────────────────────────────┐   │
│  │                    SANDBOX LAYER                                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ Namespace A │  │ Namespace B │  │ Namespace C │  │ Namespace D│  │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌────────┐ │  │   │
│  │  │ │Container│ │  │ │Container│ │  │ │Container│ │  ││Container│ │  │   │
│  │  │ │  A-1    │ │  │ │  B-1    │ │  │ │  C-1    │ │  ││  D-1   │ │  │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │ └────────┘ │  │   │
│  │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │            │  │   │
│  │  │ │Container│ │  │ │Container│ │  │ │Container│ │  │            │  │   │
│  │  │ │  A-2    │ │  │ │  B-2    │ │  │ │  C-2    │ │  │            │  │   │
│  │  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │  │            │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Agent Communication Security

```python
# agent_communication.py
"""
Secure inter-agent communication with mandatory access control.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Dict, List
import hashlib
import json

class MessageClassification(Enum):
    PUBLIC = auto()      # Safe for all agents
    INTERNAL = auto()    # Within trust domain
    RESTRICTED = auto()  # Specific recipients only
    SECRET = auto()      # Encrypted, logged separately

@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    trust_domain: str
    capabilities: frozenset[str]
    
    def can_communicate_with(self, other: 'AgentIdentity') -> bool:
        """Check if agents can communicate based on trust domains."""
        # Same trust domain: always allowed
        if self.trust_domain == other.trust_domain:
            return True
        # Cross-domain: requires explicit policy
        return self._check_cross_domain_policy(other)
    
    def _check_cross_domain_policy(self, other: 'AgentIdentity') -> bool:
        # Implemented by policy engine
        pass

class SecureMessageBus:
    """
    Message bus with built-in security controls.
    """
    
    def __init__(self, policy_engine, audit_logger):
        self.policy_engine = policy_engine
        self.audit_logger = audit_logger
        self.message_queue = {}
        
    async def send(
        self,
        sender: AgentIdentity,
        recipient: AgentIdentity,
        message: Dict,
        classification: MessageClassification
    ) -> bool:
        """Send message with security validation."""
        
        # 1. Validate communication permission
        if not sender.can_communicate_with(recipient):
            self.audit_logger.log_blocked(sender, recipient, "COMMUNICATION_DENIED")
            return False
        
        # 2. Check message content against policy
        if not self._validate_message_content(message, sender):
            self.audit_logger.log_blocked(sender, recipient, "CONTENT_VIOLATION")
            return False
        
        # 3. Apply classification controls
        encrypted_message = self._apply_classification(message, classification)
        
        # 4. Log the message (according to classification)
        self.audit_logger.log_message(sender, recipient, classification)
        
        # 5. Route the message
        await self._route_message(recipient, encrypted_message)
        return True
    
    def _validate_message_content(self, message: Dict, sender: AgentIdentity) -> bool:
        """Validate message doesn't contain prohibited content."""
        message_str = json.dumps(message)
        
        # Check for code injection attempts
        prohibited_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True',
        ]
        
        import re
        for pattern in prohibited_patterns:
            if re.search(pattern, message_str, re.IGNORECASE):
                return False
        
        return True
```

### 4.3 Capability-Based Access Control

```python
# capability_system.py
"""
Capability-based access control for agent operations.
"""

from dataclasses import dataclass
from typing import Set, Optional
from enum import Enum, auto

class Capability(Enum):
    # File system
    FS_READ = auto()
    FS_WRITE = auto()
    FS_EXECUTE = auto()
    FS_DELETE = auto()
    
    # Network
    NET_OUTBOUND = auto()
    NET_INBOUND = auto()
    NET_LOCAL = auto()
    
    # Process
    PROC_SPAWN = auto()
    PROC_KILL = auto()
    PROC_PTRACE = auto()
    
    # System
    SYS_MOUNT = auto()
    SYS_CHROOT = auto()
    SYS_MODULE = auto()
    
    # Code execution
    CODE_EXEC_PYTHON = auto()
    CODE_EXEC_SHELL = auto()
    CODE_EXEC_BINARY = auto()
    
    # Data access
    DATA_READ_SENSITIVE = auto()
    DATA_WRITE_SENSITIVE = auto()
    DATA_EXPORT = auto()

@dataclass(frozen=True)
class CapabilitySet:
    """Immutable set of capabilities granted to an agent."""
    capabilities: frozenset[Capability]
    
    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities
    
    def require(self, capability: Capability) -> None:
        if not self.has(capability):
            raise CapabilityError(f"Missing required capability: {capability}")

# Predefined capability profiles
CAPABILITY_PROFILES = {
    "researcher": CapabilitySet(frozenset([
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.NET_OUTBOUND,
        Capability.PROC_SPAWN,
        Capability.CODE_EXEC_PYTHON,
    ])),
    
    "analyzer": CapabilitySet(frozenset([
        Capability.FS_READ,
        Capability.NET_LOCAL,
        Capability.CODE_EXEC_PYTHON,
        Capability.DATA_READ_SENSITIVE,
    ])),
    
    "codegen": CapabilitySet(frozenset([
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.FS_EXECUTE,
        Capability.PROC_SPAWN,
        Capability.CODE_EXEC_PYTHON,
        Capability.CODE_EXEC_BINARY,
    ])),
    
    "reviewer": CapabilitySet(frozenset([
        Capability.FS_READ,
        Capability.NET_LOCAL,
    ])),
    
    "executor": CapabilitySet(frozenset([
        Capability.FS_READ,
        Capability.FS_WRITE,
        Capability.FS_EXECUTE,
        Capability.PROC_SPAWN,
        Capability.PROC_KILL,
        Capability.CODE_EXEC_PYTHON,
        Capability.CODE_EXEC_SHELL,
        Capability.CODE_EXEC_BINARY,
    ])),
}

class CapabilityError(Exception):
    pass
```

### 4.4 Network Isolation

```yaml
# network-policies.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-isolation
  namespace: autoconstitution
spec:
  podSelector:
    matchLabels:
      app: agent
  policyTypes:
  - Ingress
  - Egress
  ingress:
  # Only accept from control plane
  - from:
    - namespaceSelector:
        matchLabels:
          name: control-plane
    ports:
    - protocol: TCP
      port: 8080
  egress:
  # Allow DNS
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: UDP
      port: 53
  # Allow to message broker only
  - to:
    - podSelector:
        matchLabels:
          app: message-broker
    ports:
    - protocol: TCP
      port: 5672
  # Allow to approved external APIs (whitelist)
  - to:
    - ipBlock:
        cidr: 140.82.0.0/16  # GitHub API
    - ipBlock:
        cidr: 13.107.0.0/16  # Azure/Microsoft
    ports:
    - protocol: TCP
      port: 443
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: sandbox-deny-all
  namespace: autoconstitution
spec:
  podSelector:
    matchLabels:
      type: sandbox
  policyTypes:
  - Ingress
  - Egress
  # Default deny - explicit allows required
```

---

## 5. Resource Limits

### 5.1 Cgroup v2 Configuration

```
# /sys/fs/cgroup/autoconstitution/
# ├── agents/
# │   ├── agent-a/
# │   │   ├── experiments/
# │   │   │   ├── exp-001/
# │   │   │   └── exp-002/
# │   │   └── analysis/
# │   ├── agent-b/
# │   └── agent-c/
# └── system/
```

```bash
#!/bin/bash
# setup-cgroups.sh

CGROUP_BASE="/sys/fs/cgroup/autoconstitution"

# Create hierarchy
mkdir -p "$CGROUP_BASE/agents"
mkdir -p "$CGROUP_BASE/system"

# Configure global limits
echo "16G" > "$CGROUP_BASE/memory.max"
echo "8" > "$CGROUP_BASE/cpu.max"  # 800% = 8 cores
echo "10000" > "$CGROUP_BASE/pids.max"

# Configure per-agent limits
setup_agent_cgroup() {
    local agent_id=$1
    local cgroup_path="$CGROUP_BASE/agents/$agent_id"
    
    mkdir -p "$cgroup_path"
    
    # Memory: 4GB max, 2GB high (throttling threshold)
    echo "4G" > "$cgroup_path/memory.max"
    echo "2G" > "$cgroup_path/memory.high"
    
    # CPU: 200% = 2 cores max
    echo "200000 100000" > "$cgroup_path/cpu.max"
    
    # IO: 100MB/s read, 50MB/s write
    echo "rbps=104857600 wbps=52428800" > "$cgroup_path/io.max"
    
    # PIDs: max 100 processes
    echo "100" > "$cgroup_path/pids.max"
    
    # Network: 10MB/s limit using tc (configured separately)
    
    # Enable memory pressure notifications
    echo "1" > "$cgroup_path/memory.pressure"
}

# Setup experiment sub-cgroup with stricter limits
setup_experiment_cgroup() {
    local agent_id=$1
    local exp_id=$2
    local cgroup_path="$CGROUP_BASE/agents/$agent_id/experiments/$exp_id"
    
    mkdir -p "$cgroup_path"
    
    # Stricter limits for individual experiments
    echo "1G" > "$cgroup_path/memory.max"
    echo "100000 100000" > "$cgroup_path/cpu.max"  # 1 core
    echo "50" > "$cgroup_path/pids.max"
    echo "30" > "$cgroup_path/cpu.uclamp.max"  # Max 30% CPU when contended
}

# Setup for each agent
setup_agent_cgroup "agent-a"
setup_agent_cgroup "agent-b"
setup_agent_cgroup "agent-c"
```

### 5.2 Kubernetes Resource Quotas

```yaml
# resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: agent-quota
  namespace: autoconstitution
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    services.loadbalancers: "2"
    services.nodeports: "5"
    pods: "50"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: agent-limits
  namespace: autoconstitution
spec:
  limits:
  - default:
      cpu: "1"
      memory: 2Gi
    defaultRequest:
      cpu: 100m
      memory: 256Mi
    max:
      cpu: "4"
      memory: 8Gi
    min:
      cpu: 50m
      memory: 128Mi
    type: Container
  - max:
      storage: 10Gi
    min:
      storage: 1Gi
    type: PersistentVolumeClaim
```

### 5.3 Resource Monitoring and Enforcement

```python
# resource_monitor.py
"""
Real-time resource monitoring with automatic enforcement.
"""

import asyncio
import psutil
from dataclasses import dataclass
from typing import Dict, Optional, Callable
from enum import Enum
import time

class ResourceAction(Enum):
    WARN = "warn"
    THROTTLE = "throttle"
    KILL = "kill"
    ISOLATE = "isolate"

@dataclass
class ResourceThreshold:
    warning: float
    critical: float
    action: ResourceAction

class ResourceMonitor:
    """
    Monitors and enforces resource limits for agent processes.
    """
    
    def __init__(self):
        self.thresholds: Dict[str, ResourceThreshold] = {
            "cpu_percent": ResourceThreshold(70, 90, ResourceAction.THROTTLE),
            "memory_percent": ResourceThreshold(80, 95, ResourceAction.KILL),
            "disk_io_mbps": ResourceThreshold(50, 100, ResourceAction.THROTTLE),
            "network_mbps": ResourceThreshold(10, 50, ResourceAction.ISOLATE),
            "open_files": ResourceThreshold(500, 1000, ResourceAction.WARN),
            "connections": ResourceThreshold(50, 100, ResourceAction.THROTTLE),
        }
        self.callbacks: Dict[ResourceAction, Callable] = {}
        self._running = False
        
    async def start_monitoring(self, cgroup_path: str, agent_id: str):
        """Start monitoring a cgroup."""
        self._running = True
        
        while self._running:
            try:
                stats = await self._collect_stats(cgroup_path)
                await self._check_thresholds(agent_id, stats)
                await asyncio.sleep(1)
            except Exception as e:
                self._log_error(f"Monitoring error for {agent_id}: {e}")
                
    async def _collect_stats(self, cgroup_path: str) -> Dict:
        """Collect resource statistics from cgroup."""
        stats = {}
        
        # Memory stats
        try:
            with open(f"{cgroup_path}/memory.current") as f:
                stats["memory_bytes"] = int(f.read().strip())
            with open(f"{cgroup_path}/memory.max") as f:
                max_bytes = int(f.read().strip())
                stats["memory_percent"] = (stats["memory_bytes"] / max_bytes) * 100
        except:
            pass
        
        # CPU stats
        try:
            with open(f"{cgroup_path}/cpu.stat") as f:
                for line in f:
                    if line.startswith("usage_usec"):
                        stats["cpu_usage_usec"] = int(line.split()[1])
        except:
            pass
        
        # IO stats
        try:
            with open(f"{cgroup_path}/io.stat") as f:
                stats["io_stats"] = f.read()
        except:
            pass
        
        return stats
    
    async def _check_thresholds(self, agent_id: str, stats: Dict):
        """Check if any thresholds are exceeded."""
        for metric, threshold in self.thresholds.items():
            if metric in stats:
                value = stats[metric]
                
                if value >= threshold.critical:
                    await self._take_action(agent_id, metric, value, threshold.action)
                elif value >= threshold.warning:
                    self._log_warning(agent_id, metric, value)
    
    async def _take_action(self, agent_id: str, metric: str, value: float, action: ResourceAction):
        """Execute enforcement action."""
        if action in self.callbacks:
            await self.callbacks[action](agent_id, metric, value)
        
        if action == ResourceAction.KILL:
            await self._kill_agent(agent_id)
        elif action == ResourceAction.THROTTLE:
            await self._throttle_agent(agent_id)
        elif action == ResourceAction.ISOLATE:
            await self._isolate_agent(agent_id)
    
    async def _kill_agent(self, agent_id: str):
        """Terminate agent processes."""
        import signal
        import os
        
        cgroup_procs = f"/sys/fs/cgroup/autoconstitution/agents/{agent_id}/cgroup.procs"
        try:
            with open(cgroup_procs) as f:
                for line in f:
                    pid = int(line.strip())
                    try:
                        os.kill(pid, signal.SIGTERM)
                        await asyncio.sleep(5)
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
        except FileNotFoundError:
            pass
    
    async def _throttle_agent(self, agent_id: str):
        """Reduce agent CPU priority."""
        cpu_max = f"/sys/fs/cgroup/autoconstitution/agents/{agent_id}/cpu.max"
        try:
            with open(cpu_max, "w") as f:
                f.write("50000 100000")  # Reduce to 50% CPU
        except:
            pass
    
    async def _isolate_agent(self, agent_id: str):
        """Network isolate the agent."""
        # Apply network policy to drop all traffic
        pass
```

---

## 6. Code Validation

### 6.1 Multi-Stage Validation Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   STAGE 1   │───►│   STAGE 2   │───►│   STAGE 3   │───►│   STAGE 4   │
│   Syntax    │    │   Static    │    │   Dynamic   │    │   Behavior  │
│   Check     │    │   Analysis  │    │   Analysis  │    │   Analysis  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                  │                  │                  │
      ▼                  ▼                  ▼                  ▼
  AST Parse         Bandit/Semgrep      Sandbox exec      Pattern match
  Import check      Dependency scan     Fuzzing           Heuristic detect
```

### 6.2 Static Analysis Integration

```python
# code_validator.py
"""
Multi-stage code validation for agent-generated code.
"""

import ast
import subprocess
import tempfile
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationFinding:
    rule_id: str
    severity: Severity
    message: str
    line: int
    column: int
    file: str
    remediation: Optional[str] = None

class CodeValidator:
    """
    Comprehensive code validation pipeline.
    """
    
    # Prohibited imports and functions
    BLOCKLIST = {
        "imports": {
            "os.system", "subprocess", "pty", "socket", 
            "ctypes", "cffi", "mmap", "resource",
            "setuptools", "pip", "easy_install",
        },
        "functions": {
            "eval", "exec", "compile", "__import__",
            "getattr", "setattr", "delattr",
            "open", "file", "input", "raw_input",
        },
        "attributes": {
            "__class__", "__bases__", "__mro__",
            "__subclasses__", "__globals__", "__code__",
        }
    }
    
    def __init__(self):
        self.findings: List[ValidationFinding] = []
        
    def validate(self, code: str, filename: str = "<unknown>") -> Tuple[bool, List[ValidationFinding]]:
        """
        Run complete validation pipeline.
        Returns (is_valid, findings).
        """
        self.findings = []
        
        # Stage 1: Syntax validation
        if not self._validate_syntax(code, filename):
            return False, self.findings
        
        # Stage 2: AST analysis
        self._validate_ast(code, filename)
        
        # Stage 3: Static analysis tools
        self._run_static_analysis(code, filename)
        
        # Stage 4: Pattern matching
        self._pattern_analysis(code, filename)
        
        # Check if any critical errors
        critical_errors = [f for f in self.findings if f.severity == Severity.CRITICAL]
        
        return len(critical_errors) == 0, self.findings
    
    def _validate_syntax(self, code: str, filename: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            self.findings.append(ValidationFinding(
                rule_id="SYNTAX_ERROR",
                severity=Severity.ERROR,
                message=f"Syntax error: {e.msg}",
                line=e.lineno or 0,
                column=e.offset or 0,
                file=filename
            ))
            return False
    
    def _validate_ast(self, code: str, filename: str):
        """Analyze AST for prohibited patterns."""
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            # Check for prohibited imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.BLOCKLIST["imports"]:
                        self.findings.append(ValidationFinding(
                            rule_id="PROHIBITED_IMPORT",
                            severity=Severity.CRITICAL,
                            message=f"Import '{alias.name}' is prohibited",
                            line=node.lineno,
                            column=node.col_offset,
                            file=filename,
                            remediation=f"Remove import of '{alias.name}'"
                        ))
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    if full_name in self.BLOCKLIST["imports"] or module in self.BLOCKLIST["imports"]:
                        self.findings.append(ValidationFinding(
                            rule_id="PROHIBITED_IMPORT",
                            severity=Severity.CRITICAL,
                            message=f"Import '{full_name}' is prohibited",
                            line=node.lineno,
                            column=node.col_offset,
                            file=filename
                        ))
            
            # Check for prohibited function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.BLOCKLIST["functions"]:
                        self.findings.append(ValidationFinding(
                            rule_id="PROHIBITED_FUNCTION",
                            severity=Severity.CRITICAL,
                            message=f"Function '{node.func.id}' is prohibited",
                            line=node.lineno,
                            column=node.col_offset,
                            file=filename
                        ))
            
            # Check for __import__ usage
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                    self.findings.append(ValidationFinding(
                        rule_id="DYNAMIC_IMPORT",
                        severity=Severity.CRITICAL,
                        message="Dynamic imports are prohibited",
                        line=node.lineno,
                        column=node.col_offset,
                        file=filename
                    ))
            
            # Check for dangerous string formatting
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ("format", "format_map"):
                        self.findings.append(ValidationFinding(
                            rule_id="STRING_FORMATTING",
                            severity=Severity.WARNING,
                            message="String formatting may be unsafe - verify inputs",
                            line=node.lineno,
                            column=node.col_offset,
                            file=filename
                        ))
    
    def _run_static_analysis(self, code: str, filename: str):
        """Run external static analysis tools."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Run bandit
            result = subprocess.run(
                ["bandit", "-f", "json", "-q", temp_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.stdout:
                import json
                try:
                    bandit_results = json.loads(result.stdout)
                    for issue in bandit_results.get("results", []):
                        severity_map = {
                            "LOW": Severity.INFO,
                            "MEDIUM": Severity.WARNING,
                            "HIGH": Severity.CRITICAL
                        }
                        self.findings.append(ValidationFinding(
                            rule_id=f"BANDIT:{issue['test_id']}",
                            severity=severity_map.get(issue["issue_severity"], Severity.WARNING),
                            message=issue["issue_text"],
                            line=issue["line_number"],
                            column=issue["col_offset"],
                            file=filename
                        ))
                except json.JSONDecodeError:
                    pass
            
            # Run semgrep
            result = subprocess.run(
                ["semgrep", "--config=auto", "--json", "--quiet", temp_path],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                try:
                    semgrep_results = json.loads(result.stdout)
                    for result_item in semgrep_results.get("results", []):
                        self.findings.append(ValidationFinding(
                            rule_id=f"SEMGREP:{result_item['check_id']}",
                            severity=Severity.WARNING,
                            message=result_item["extra"]["message"],
                            line=result_item["start"]["line"],
                            column=result_item["start"]["col"],
                            file=filename
                        ))
                except json.JSONDecodeError:
                    pass
                    
        finally:
            os.unlink(temp_path)
    
    def _pattern_analysis(self, code: str, filename: str):
        """Pattern-based security analysis."""
        import re
        
        patterns = {
            r"password\s*=\s*['\"][^'\"]+['\"]": (
                Severity.CRITICAL,
                "Hardcoded password detected"
            ),
            r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]": (
                Severity.CRITICAL,
                "Hardcoded API key detected"
            ),
            r"secret\s*=\s*['\"][^'\"]+['\"]": (
                Severity.CRITICAL,
                "Hardcoded secret detected"
            ),
            r"token\s*=\s*['\"][^'\"]+['\"]": (
                Severity.WARNING,
                "Possible hardcoded token detected"
            ),
            r"http://[^\s\"']+": (
                Severity.WARNING,
                "Insecure HTTP URL detected"
            ),
            r"verify\s*=\s*False": (
                Severity.ERROR,
                "SSL verification disabled"
            ),
            r"shell\s*=\s*True": (
                Severity.CRITICAL,
                "Shell execution enabled"
            ),
        }
        
        for pattern, (severity, message) in patterns.items():
            for match in re.finditer(pattern, code, re.IGNORECASE):
                line_num = code[:match.start()].count('\n') + 1
                self.findings.append(ValidationFinding(
                    rule_id="PATTERN_MATCH",
                    severity=severity,
                    message=message,
                    line=line_num,
                    column=match.start(),
                    file=filename
                ))
```

### 6.3 Dynamic Analysis

```python
# dynamic_analyzer.py
"""
Dynamic analysis in sandboxed environment.
"""

import docker
import tempfile
import os
import json
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class DynamicAnalysisResult:
    exit_code: int
    stdout: str
    stderr: str
    syscalls: list
    network_activity: list
    file_accesses: list
    execution_time: float
    memory_peak: int
    cpu_time: float

class DynamicAnalyzer:
    """
    Execute code in sandbox and monitor behavior.
    """
    
    def __init__(self, docker_client=None):
        self.docker = docker_client or docker.from_env()
        
    def analyze(
        self,
        code: str,
        timeout: int = 60,
        memory_limit: str = "512m",
        cpu_limit: float = 1.0
    ) -> DynamicAnalysisResult:
        """
        Run code in sandbox and collect behavioral data.
        """
        
        # Create temporary directory for code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "script.py")
            with open(code_path, 'w') as f:
                f.write(code)
            
            # Create Dockerfile for analysis
            dockerfile = f"""
FROM python:3.11-slim
RUN apt-get update && apt-get install -y strace tcpdump && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY script.py .
# Run with strace to capture syscalls
CMD ["strace", "-f", "-e", "trace=all", "-o", "/tmp/strace.log", "python", "script.py"]
"""
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile)
            
            # Build and run container
            image, _ = self.docker.images.build(path=tmpdir, tag="dynamic-analysis")
            
            container = self.docker.containers.run(
                image,
                detach=True,
                mem_limit=memory_limit,
                cpu_quota=int(cpu_limit * 100000),
                network_mode="none",  # No network by default
                security_opt=["no-new-privileges:true"],
                cap_drop=["ALL"],
                read_only=True,
                tmpfs={"/tmp": "noexec,nosuid,size=100m"},
            )
            
            try:
                result = container.wait(timeout=timeout)
                
                # Collect logs
                stdout = container.logs(stdout=True, stderr=False).decode()
                stderr = container.logs(stdout=False, stderr=True).decode()
                
                # Get strace output
                try:
                    strace_log = container.exec_run("cat /tmp/strace.log").output.decode()
                    syscalls = self._parse_strace(strace_log)
                except:
                    syscalls = []
                
                # Get stats
                stats = container.stats(stream=False)
                memory_peak = stats.get("memory_stats", {}).get("peak_usage", 0)
                cpu_stats = stats.get("cpu_stats", {})
                cpu_time = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) / 1e9
                
                return DynamicAnalysisResult(
                    exit_code=result["StatusCode"],
                    stdout=stdout,
                    stderr=stderr,
                    syscalls=syscalls,
                    network_activity=[],  # Network disabled
                    file_accesses=self._extract_file_accesses(syscalls),
                    execution_time=timeout,  # Actual time would be measured
                    memory_peak=memory_peak,
                    cpu_time=cpu_time
                )
                
            finally:
                container.remove(force=True)
                self.docker.images.remove(image.id, force=True)
    
    def _parse_strace(self, strace_log: str) -> list:
        """Parse strace output to extract syscalls."""
        syscalls = []
        for line in strace_log.split('\n'):
            if '(' in line:
                syscall = line.split('(')[0].strip()
                if syscall:
                    syscalls.append(syscall)
        return syscalls
    
    def _extract_file_accesses(self, syscalls: list) -> list:
        """Extract file operations from syscalls."""
        file_syscalls = {'open', 'openat', 'creat', 'access', 'stat', 'read', 'write'}
        return [s for s in syscalls if s in file_syscalls]
```

---

## 7. Audit Logging

### 7.1 Audit Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AUDIT SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │   Agents    │    │  Sandboxes  │    │   System    │    │   Network   │ │
│   │             │    │             │    │             │    │             │ │
│   │ - Actions   │───►│ - Execution │───►│ - Auth      │───►│ - Traffic   │ │
│   │ - Messages  │    │ - Resources │    │ - Config    │    │ - Flows     │ │
│   │ - Decisions │    │ - Syscalls  │    │ - Changes   │    │ - Anomalies │ │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘ │
│          │                  │                  │                  │        │
│          └──────────────────┴──────────────────┴──────────────────┘        │
│                                     │                                       │
│                          ┌──────────┴──────────┐                           │
│                          │   Audit Collector   │                           │
│                          │   (Fluentd/Vector)  │                           │
│                          └──────────┬──────────┘                           │
│                                     │                                       │
│          ┌──────────────────────────┼──────────────────────────┐           │
│          │                          │                          │           │
│   ┌──────┴──────┐          ┌────────┴────────┐        ┌────────┴────────┐ │
│   │   Hot Store │          │   Warm Store    │        │   Cold Store    │ │
│   │  (7 days)   │          │   (90 days)     │        │  (7 years)      │ │
│   │  (ClickHouse│          │   (S3 Standard) │        │  (S3 Glacier)   │ │
│   └─────────────┘          └─────────────────┘        └─────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Audit Event Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "autoconstitution Audit Event",
  "type": "object",
  "required": ["event_id", "timestamp", "event_type", "severity", "actor"],
  "properties": {
    "event_id": {
      "type": "string",
      "format": "uuid",
      "description": "Unique event identifier"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp"
    },
    "event_type": {
      "type": "string",
      "enum": [
        "AGENT_CREATED", "AGENT_DESTROYED", "AGENT_ACTION",
        "CODE_EXECUTED", "CODE_VALIDATED", "CODE_BLOCKED",
        "MESSAGE_SENT", "MESSAGE_RECEIVED", "MESSAGE_BLOCKED",
        "RESOURCE_ALLOCATED", "RESOURCE_EXCEEDED", "RESOURCE_RELEASED",
        "AUTH_SUCCESS", "AUTH_FAILURE", "PRIVILEGE_ESCALATION",
        "NETWORK_ACCESS", "FILE_ACCESS", "SYSTEM_CALL",
        "POLICY_VIOLATION", "SECURITY_ALERT", "CONFIG_CHANGE"
      ]
    },
    "severity": {
      "type": "string",
      "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    },
    "actor": {
      "type": "object",
      "required": ["type", "id"],
      "properties": {
        "type": {
          "type": "string",
          "enum": ["agent", "user", "system", "sandbox"]
        },
        "id": { "type": "string" },
        "trust_domain": { "type": "string" },
        "capabilities": {
          "type": "array",
          "items": { "type": "string" }
        }
      }
    },
    "target": {
      "type": "object",
      "properties": {
        "type": { "type": "string" },
        "id": { "type": "string" },
        "resource": { "type": "string" }
      }
    },
    "action": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "parameters": { "type": "object" },
        "result": {
          "type": "string",
          "enum": ["SUCCESS", "FAILURE", "BLOCKED", "PENDING"]
        }
      }
    },
    "context": {
      "type": "object",
      "properties": {
        "session_id": { "type": "string" },
        "request_id": { "type": "string" },
        "trace_id": { "type": "string" },
        "source_ip": { "type": "string" },
        "user_agent": { "type": "string" },
        "sandbox_id": { "type": "string" },
        "experiment_id": { "type": "string" }
      }
    },
    "metadata": {
      "type": "object",
      "description": "Event-specific metadata"
    },
    "integrity": {
      "type": "object",
      "properties": {
        "hash": { "type": "string" },
        "algorithm": { "type": "string" },
        "previous_hash": { "type": "string" }
      }
    }
  }
}
```

### 7.3 Audit Logger Implementation

```python
# audit_logger.py
"""
Comprehensive audit logging system.
"""

import json
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import aiohttp

class EventType(Enum):
    AGENT_CREATED = "AGENT_CREATED"
    AGENT_DESTROYED = "AGENT_DESTROYED"
    AGENT_ACTION = "AGENT_ACTION"
    CODE_EXECUTED = "CODE_EXECUTED"
    CODE_VALIDATED = "CODE_VALIDATED"
    CODE_BLOCKED = "CODE_BLOCKED"
    MESSAGE_SENT = "MESSAGE_SENT"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    MESSAGE_BLOCKED = "MESSAGE_BLOCKED"
    RESOURCE_ALLOCATED = "RESOURCE_ALLOCATED"
    RESOURCE_EXCEEDED = "RESOURCE_EXCEEDED"
    RESOURCE_RELEASED = "RESOURCE_RELEASED"
    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    NETWORK_ACCESS = "NETWORK_ACCESS"
    FILE_ACCESS = "FILE_ACCESS"
    SYSTEM_CALL = "SYSTEM_CALL"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    SECURITY_ALERT = "SECURITY_ALERT"
    CONFIG_CHANGE = "CONFIG_CHANGE"

class Severity(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class Actor:
    type: str
    id: str
    trust_domain: Optional[str] = None
    capabilities: Optional[List[str]] = None

@dataclass
class Target:
    type: str
    id: str
    resource: Optional[str] = None

@dataclass
class Action:
    name: str
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[str] = None

@dataclass
class Context:
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    source_ip: Optional[str] = None
    sandbox_id: Optional[str] = None
    experiment_id: Optional[str] = None

@dataclass
class AuditEvent:
    event_id: str
    timestamp: str
    event_type: str
    severity: str
    actor: Dict
    target: Optional[Dict] = None
    action: Optional[Dict] = None
    context: Optional[Dict] = None
    metadata: Optional[Dict] = None
    integrity: Optional[Dict] = None

class AuditLogger:
    """
    High-performance audit logging with tamper detection.
    """
    
    def __init__(
        self,
        clickhouse_url: str,
        s3_bucket: str,
        buffer_size: int = 1000,
        flush_interval: int = 5
    ):
        self.clickhouse_url = clickhouse_url
        self.s3_bucket = s3_bucket
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        
        self._buffer: List[AuditEvent] = []
        self._last_hash: Optional[str] = None
        self._lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the background flush task."""
        self._flush_task = asyncio.create_task(self._periodic_flush())
        
    async def stop(self):
        """Stop the logger and flush remaining events."""
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        await self._flush_buffer()
        
    async def log(
        self,
        event_type: EventType,
        severity: Severity,
        actor: Actor,
        target: Optional[Target] = None,
        action: Optional[Action] = None,
        context: Optional[Context] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log an audit event.
        Returns the event ID.
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Calculate integrity hash
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type.value,
            "severity": severity.value,
            "actor": asdict(actor),
            "target": asdict(target) if target else None,
            "action": asdict(action) if action else None,
            "context": asdict(context) if context else None,
            "metadata": metadata,
        }
        
        integrity_hash = self._calculate_hash(event_data)
        integrity = {
            "hash": integrity_hash,
            "algorithm": "SHA-256",
            "previous_hash": self._last_hash
        }
        self._last_hash = integrity_hash
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=timestamp,
            event_type=event_type.value,
            severity=severity.value,
            actor=asdict(actor),
            target=asdict(target) if target else None,
            action=asdict(action) if action else None,
            context=asdict(context) if context else None,
            metadata=metadata,
            integrity=integrity
        )
        
        async with self._lock:
            self._buffer.append(event)
            
            # Immediate flush for critical events
            if severity == Severity.CRITICAL:
                await self._flush_buffer()
            elif len(self._buffer) >= self.buffer_size:
                await self._flush_buffer()
        
        return event_id
    
    def _calculate_hash(self, data: Dict) -> str:
        """Calculate SHA-256 hash of event data."""
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    async def _periodic_flush(self):
        """Periodically flush the buffer."""
        while True:
            try:
                await asyncio.sleep(self.flush_interval)
                async with self._lock:
                    if self._buffer:
                        await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Flush error: {e}")
                
    async def _flush_buffer(self):
        """Flush buffer to storage backends."""
        if not self._buffer:
            return
        
        events = self._buffer.copy()
        self._buffer = []
        
        # Write to ClickHouse for hot storage
        await self._write_to_clickhouse(events)
        
        # Write to S3 for archival
        await self._write_to_s3(events)
        
    async def _write_to_clickhouse(self, events: List[AuditEvent]):
        """Write events to ClickHouse."""
        try:
            # Convert to INSERT format
            rows = []
            for event in events:
                row = {
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "actor_id": event.actor.get("id"),
                    "actor_type": event.actor.get("type"),
                    "target_id": event.target.get("id") if event.target else None,
                    "action_name": event.action.get("name") if event.action else None,
                    "result": event.action.get("result") if event.action else None,
                    "metadata": json.dumps(event.metadata) if event.metadata else None,
                    "integrity_hash": event.integrity.get("hash") if event.integrity else None,
                }
                rows.append(row)
            
            # Bulk insert to ClickHouse
            async with aiohttp.ClientSession() as session:
                # Implementation depends on ClickHouse setup
                pass
                
        except Exception as e:
            # Log error and potentially retry
            print(f"ClickHouse write error: {e}")
            
    async def _write_to_s3(self, events: List[AuditEvent]):
        """Write events to S3 for archival."""
        import boto3
        
        try:
            s3 = boto3.client('s3')
            
            # Group by date for partitioning
            date_prefix = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            
            # Create NDJSON file
            lines = []
            for event in events:
                lines.append(json.dumps({
                    "event_id": event.event_id,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "severity": event.severity,
                    "actor": event.actor,
                    "target": event.target,
                    "action": event.action,
                    "context": event.context,
                    "metadata": event.metadata,
                    "integrity": event.integrity,
                }))
            
            content = '\n'.join(lines)
            key = f"audit/{date_prefix}/{uuid.uuid4()}.ndjson"
            
            s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=content,
                ContentType="application/x-ndjson"
            )
            
        except Exception as e:
            print(f"S3 write error: {e}")

    # Convenience methods for common events
    async def log_agent_action(
        self,
        agent_id: str,
        action_name: str,
        result: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Log an agent action."""
        return await self.log(
            event_type=EventType.AGENT_ACTION,
            severity=Severity.INFO,
            actor=Actor(type="agent", id=agent_id),
            action=Action(name=action_name, result=result),
            metadata=metadata
        )
    
    async def log_code_execution(
        self,
        agent_id: str,
        sandbox_id: str,
        code_hash: str,
        result: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Log code execution."""
        return await self.log(
            event_type=EventType.CODE_EXECUTED,
            severity=Severity.INFO,
            actor=Actor(type="agent", id=agent_id),
            target=Target(type="sandbox", id=sandbox_id),
            action=Action(name="execute_code", parameters={"code_hash": code_hash}, result=result),
            context=Context(sandbox_id=sandbox_id),
            metadata=metadata
        )
    
    async def log_security_alert(
        self,
        alert_type: str,
        severity: Severity,
        actor: Actor,
        description: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Log a security alert."""
        return await self.log(
            event_type=EventType.SECURITY_ALERT,
            severity=severity,
            actor=actor,
            action=Action(name=alert_type, parameters={"description": description}),
            metadata=metadata
        )
```

### 7.4 Audit Query Interface

```python
# audit_query.py
"""
Query interface for audit logs.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class AuditQuery:
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    severities: Optional[List[str]] = None
    actor_ids: Optional[List[str]] = None
    sandbox_ids: Optional[List[str]] = None
    experiment_ids: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0

class AuditQueryEngine:
    """
    Query engine for audit events.
    """
    
    def __init__(self, clickhouse_client):
        self.clickhouse = clickhouse_client
        
    async def query(self, query: AuditQuery) -> List[Dict]:
        """Execute audit query."""
        
        conditions = []
        params = {}
        
        if query.start_time:
            conditions.append("timestamp >= %(start_time)s")
            params["start_time"] = query.start_time.isoformat()
            
        if query.end_time:
            conditions.append("timestamp <= %(end_time)s")
            params["end_time"] = query.end_time.isoformat()
            
        if query.event_types:
            conditions.append("event_type IN %(event_types)s")
            params["event_types"] = query.event_types
            
        if query.severities:
            conditions.append("severity IN %(severities)s")
            params["severities"] = query.severities
            
        if query.actor_ids:
            conditions.append("actor_id IN %(actor_ids)s")
            params["actor_ids"] = query.actor_ids
            
        if query.sandbox_ids:
            conditions.append("sandbox_id IN %(sandbox_ids)s")
            params["sandbox_ids"] = query.sandbox_ids
            
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"""
        SELECT *
        FROM audit_events
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT %(limit)s
        OFFSET %(offset)s
        """
        params["limit"] = query.limit
        params["offset"] = query.offset
        
        return await self.clickhouse.query(sql, params)
    
    async def get_agent_activity(
        self,
        agent_id: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get comprehensive activity report for an agent."""
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        # Event counts by type
        events_sql = """
        SELECT event_type, count() as count
        FROM audit_events
        WHERE actor_id = %(agent_id)s
          AND timestamp >= %(start_time)s
        GROUP BY event_type
        """
        
        # Resource usage
        resources_sql = """
        SELECT 
            countIf(event_type = 'RESOURCE_ALLOCATED') as allocations,
            countIf(event_type = 'RESOURCE_EXCEEDED') as violations,
            countIf(event_type = 'RESOURCE_RELEASED') as releases
        FROM audit_events
        WHERE actor_id = %(agent_id)s
          AND timestamp >= %(start_time)s
        """
        
        # Security events
        security_sql = """
        SELECT *
        FROM audit_events
        WHERE actor_id = %(agent_id)s
          AND severity IN ('ERROR', 'CRITICAL')
          AND timestamp >= %(start_time)s
        ORDER BY timestamp DESC
        LIMIT 100
        """
        
        params = {"agent_id": agent_id, "start_time": start_time.isoformat()}
        
        return {
            "agent_id": agent_id,
            "period_hours": hours,
            "events": await self.clickhouse.query(events_sql, params),
            "resources": await self.clickhouse.query(resources_sql, params),
            "security_events": await self.clickhouse.query(security_sql, params),
        }
    
    async def detect_anomalies(
        self,
        hours: int = 1
    ) -> List[Dict]:
        """Detect anomalous behavior patterns."""
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        # High frequency events (potential DoS)
        high_freq_sql = """
        SELECT actor_id, event_type, count() as event_count
        FROM audit_events
        WHERE timestamp >= %(start_time)s
        GROUP BY actor_id, event_type
        HAVING event_count > 1000
        """
        
        # Multiple failures (potential attack)
        failures_sql = """
        SELECT actor_id, 
               countIf(action.result = 'FAILURE') as failures,
               countIf(action.result = 'BLOCKED') as blocked
        FROM audit_events
        WHERE timestamp >= %(start_time)s
        GROUP BY actor_id
        HAVING failures > 10 OR blocked > 5
        """
        
        params = {"start_time": start_time.isoformat()}
        
        anomalies = []
        
        high_freq = await self.clickhouse.query(high_freq_sql, params)
        for row in high_freq:
            anomalies.append({
                "type": "HIGH_FREQUENCY",
                "actor_id": row["actor_id"],
                "event_type": row["event_type"],
                "count": row["event_count"],
                "severity": "WARNING"
            })
        
        failures = await self.clickhouse.query(failures_sql, params)
        for row in failures:
            anomalies.append({
                "type": "MULTIPLE_FAILURES",
                "actor_id": row["actor_id"],
                "failures": row["failures"],
                "blocked": row["blocked"],
                "severity": "ERROR"
            })
        
        return anomalies
```

---

## 8. Deployment Configuration

### 8.1 Kubernetes Security Context

```yaml
# security-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autoconstitution-control
  namespace: autoconstitution
spec:
  replicas: 3
  selector:
    matchLabels:
      app: control-plane
  template:
    metadata:
      labels:
        app: control-plane
    spec:
      serviceAccountName: autoconstitution-control
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: control-plane
        image: autoconstitution/control:latest
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        resources:
          limits:
            cpu: "2"
            memory: 4Gi
          requests:
            cpu: 500m
            memory: 1Gi
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: config
          mountPath: /config
          readOnly: true
      volumes:
      - name: tmp
        emptyDir: {}
      - name: config
        configMap:
          name: autoconstitution-config
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: autoconstitution-control
  namespace: autoconstitution
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
- apiGroups: ["networking.k8s.io"]
  resources: ["networkpolicies"]
  verbs: ["get", "list", "watch", "create", "update", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: autoconstitution-control
  namespace: autoconstitution
subjects:
- kind: ServiceAccount
  name: autoconstitution-control
  namespace: autoconstitution
roleRef:
  kind: Role
  name: autoconstitution-control
  apiGroup: rbac.authorization.k8s.io
```

### 8.2 Pod Security Standards

```yaml
# pod-security.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: autoconstitution
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
    pod-security.kubernetes.io/enforce-version: latest
```

---

## 9. Security Monitoring and Alerting

### 9.1 Alert Rules

```yaml
# prometheus-alerts.yaml
groups:
- name: autoconstitution-security
  rules:
  # High rate of blocked actions
  - alert: HighBlockedActionRate
    expr: |
      rate(audit_events_total{result="BLOCKED"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High rate of blocked actions"
      description: "Agent {{ $labels.actor_id }} has high rate of blocked actions"

  # Resource limit violations
  - alert: ResourceLimitViolation
    expr: |
      audit_events_total{event_type="RESOURCE_EXCEEDED"} > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Resource limit violated"
      description: "Agent {{ $labels.actor_id }} exceeded resource limits"

  # Privilege escalation attempts
  - alert: PrivilegeEscalationAttempt
    expr: |
      audit_events_total{event_type="PRIVILEGE_ESCALATION"} > 0
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "Privilege escalation detected"
      description: "Agent {{ $labels.actor_id }} attempted privilege escalation"

  # Suspicious network activity
  - alert: SuspiciousNetworkActivity
    expr: |
      rate(audit_events_total{event_type="NETWORK_ACCESS"}[5m]) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Suspicious network activity"
      description: "High rate of network connections from {{ $labels.actor_id }}"

  # Code validation failures
  - alert: CodeValidationFailure
    expr: |
      rate(audit_events_total{event_type="CODE_BLOCKED"}[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Code validation failures"
      description: "Agent {{ $labels.actor_id }} generating blocked code"
```

---

## 10. Incident Response

### 10.1 Response Playbook

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INCIDENT RESPONSE FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DETECTION ──► CONTAINMENT ──► ANALYSIS ──► REMEDIATION ──► RECOVERY      │
│       │              │            │              │             │            │
│       ▼              ▼            ▼              ▼             ▼            │
│   Alert fired   Isolate agent  Collect logs  Kill process  Restore svc    │
│   Auto-triage   Block network   Forensics    Patch vuln    Verify fix     │
│   Escalate?     Quarantine      Correlate    Update rules  Post-mortem    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Automated Response Actions

```python
# incident_response.py
"""
Automated incident response system.
"""

from enum import Enum, auto
from typing import Dict, List, Optional
import asyncio

class IncidentSeverity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

class ResponseAction(Enum):
    LOG = auto()
    ALERT = auto()
    THROTTLE = auto()
    ISOLATE = auto()
    KILL = auto()
    QUARANTINE = auto()

class IncidentResponder:
    """
    Automated incident response system.
    """
    
    RESPONSE_MATRIX = {
        # (event_type, severity) -> [actions]
        ("PRIVILEGE_ESCALATION", IncidentSeverity.CRITICAL): [
            ResponseAction.LOG,
            ResponseAction.ALERT,
            ResponseAction.ISOLATE,
            ResponseAction.KILL,
            ResponseAction.QUARANTINE,
        ],
        ("RESOURCE_EXCEEDED", IncidentSeverity.HIGH): [
            ResponseAction.LOG,
            ResponseAction.THROTTLE,
            ResponseAction.ALERT,
        ],
        ("CODE_BLOCKED", IncidentSeverity.MEDIUM): [
            ResponseAction.LOG,
            ResponseAction.THROTTLE,
        ],
        ("NETWORK_ANOMALY", IncidentSeverity.HIGH): [
            ResponseAction.LOG,
            ResponseAction.ALERT,
            ResponseAction.ISOLATE,
        ],
    }
    
    def __init__(self, k8s_client, audit_logger):
        self.k8s = k8s_client
        self.audit = audit_logger
        
    async def handle_incident(
        self,
        event_type: str,
        severity: IncidentSeverity,
        agent_id: str,
        details: Dict
    ):
        """Handle a security incident."""
        
        # Determine response actions
        actions = self.RESPONSE_MATRIX.get(
            (event_type, severity),
            [ResponseAction.LOG, ResponseAction.ALERT]
        )
        
        for action in actions:
            try:
                await self._execute_action(action, agent_id, details)
            except Exception as e:
                await self.audit.log_security_alert(
                    alert_type="RESPONSE_FAILED",
                    severity=Severity.ERROR,
                    actor=Actor(type="system", id="incident-responder"),
                    description=f"Failed to execute {action}: {e}"
                )
    
    async def _execute_action(
        self,
        action: ResponseAction,
        agent_id: str,
        details: Dict
    ):
        """Execute a response action."""
        
        if action == ResponseAction.LOG:
            await self.audit.log_security_alert(
                alert_type="INCIDENT",
                severity=Severity.WARNING,
                actor=Actor(type="agent", id=agent_id),
                description=f"Incident: {details}"
            )
            
        elif action == ResponseAction.ALERT:
            await self._send_alert(agent_id, details)
            
        elif action == ResponseAction.THROTTLE:
            await self._throttle_agent(agent_id)
            
        elif action == ResponseAction.ISOLATE:
            await self._isolate_agent(agent_id)
            
        elif action == ResponseAction.KILL:
            await self._kill_agent(agent_id)
            
        elif action == ResponseAction.QUARANTINE:
            await self._quarantine_agent(agent_id)
    
    async def _isolate_agent(self, agent_id: str):
        """Network isolate an agent."""
        # Apply deny-all network policy
        policy = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"isolate-{agent_id}",
                "namespace": "autoconstitution"
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {"agent-id": agent_id}
                },
                "policyTypes": ["Ingress", "Egress"]
            }
        }
        await self.k8s.create_network_policy(policy)
    
    async def _kill_agent(self, agent_id: str):
        """Kill all agent processes."""
        await self.k8s.delete_pod(
            namespace="autoconstitution",
            label_selector=f"agent-id={agent_id}"
        )
    
    async def _quarantine_agent(self, agent_id: str):
        """Move agent to quarantine namespace."""
        # Create quarantine pod with full isolation
        pass
    
    async def _throttle_agent(self, agent_id: str):
        """Throttle agent resources."""
        # Patch deployment to reduce resources
        pass
    
    async def _send_alert(self, agent_id: str, details: Dict):
        """Send alert to security team."""
        # Send to PagerDuty/Slack/etc
        pass
```

---

## 11. Compliance and Governance

### 11.1 Security Controls Mapping

| Control | Implementation | Evidence |
|---------|---------------|----------|
| AC-2 | Agent identity management | Audit logs |
| AC-3 | Capability-based access control | Policy configs |
| AC-6 | Least privilege | Seccomp profiles |
| AU-6 | Audit log analysis | ClickHouse queries |
| CM-7 | Least functionality | Container hardening |
| SC-7 | Boundary protection | Network policies |
| SC-39 | Process isolation | Container sandboxes |
| SI-4 | System monitoring | Prometheus/Grafana |

### 11.2 Security Checklist

- [ ] Container images scanned for vulnerabilities
- [ ] Seccomp profiles applied to all containers
- [ ] Network policies restrict traffic flow
- [ ] Resource limits configured for all workloads
- [ ] Audit logging enabled for all components
- [ ] Code validation pipeline operational
- [ ] Incident response procedures documented
- [ ] Security monitoring dashboards configured
- [ ] Regular security assessments scheduled
- [ ] Backup and recovery procedures tested

---

## Appendix A: Configuration Files

### A.1 Complete Seccomp Profile

See section 3.3 for the base profile. Additional rules for specific workloads:

```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": [
        "bpf",
        "clone",
        "clone3",
        "fanotify_init",
        "fanotify_mark",
        "fsconfig",
        "fsmount",
        "fsopen",
        "fspick",
        "mount",
        "mount_setattr",
        "move_mount",
        "open_tree",
        "perf_event_open",
        "pidfd_getfd",
        "pidfd_open",
        "pivot_root",
        "quotactl",
        "quotactl_fd",
        "setdomainname",
        "sethostname",
        "syslog",
        "umount2",
        "unshare",
        "vm86",
        "vm86old"
      ],
      "action": "SCMP_ACT_ERRNO"
    }
  ]
}
```

### A.2 AppArmor Profile

```
#include <tunables/global>

profile autoconstitution-sandbox flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  
  # Deny dangerous capabilities
  deny capability sys_admin,
  deny capability sys_module,
  deny capability sys_rawio,
  deny capability sys_time,
  deny capability sys_nice,
  deny capability sys_resource,
  deny capability audit_control,
  deny capability audit_write,
  deny capability setpcap,
  deny capability mknod,
  deny capability fsetid,
  deny capability setgid,
  deny capability setuid,
  deny capability net_bind_service,
  deny capability net_admin,
  deny capability net_raw,
  deny capability ipc_lock,
  deny capability ipc_owner,
  deny capability sys_pacct,
  deny capability syslog,
  deny capability wake_alarm,
  deny capability block_suspend,
  
  # Allow basic file access
  / r,
  /usr/** r,
  /lib/** r,
  /lib64/** r,
  /bin/** r,
  /sbin/** r,
  /etc/** r,
  
  # Workspace access
  /workspace/** rwk,
  /tmp/** rw,
  
  # Deny sensitive paths
  deny /etc/shadow r,
  deny /etc/passwd r,
  deny /etc/group r,
  deny /proc/** w,
  deny /sys/** w,
  deny /dev/** w,
}
```

---

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Author | Security Architecture Team |
| Date | 2024 |
| Classification | Internal |
| Review Cycle | Quarterly |
