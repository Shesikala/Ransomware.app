import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # Load the scaler used during training

# Selected feature names (Exclude 'Class' as it's the target variable)
selected_feature_names = [
    "__arm_nr_cacheflush", "__arm_nr_set_tls", "_llseek", "_newselect", "accept",
    "access", "bind", "brk", "capset", "chdir", "chmod", "chown32", "clock_getres",
    "clock_gettime", "clone", "close", "connect", "dup", "dup2", "epoll_create",
    "epoll_ctl", "epoll_wait", "eventfd2", "execve", "exit", "exit_group", "faccessat",
    "fchmod", "fchown32", "fcntl", "fcntl64", "fdatasync", "flock", "fork", "fstat64",
    "fstatfs64", "fsync", "ftruncate", "ftruncate64", "futex", "getcwd", "getdents64",
    "getegid32", "geteuid32", "getgid32", "getgroups32", "getpeername", "getpgid",
    "getpid", "getppid", "getpriority", "getresgid32", "getresuid32", "getrusage",
    "getsockname", "getsockopt", "gettid", "gettimeofday", "getuid32", "inotify_add_watch",
    "inotify_init", "inotify_rm_watch", "ioctl", "lgetxattr", "link", "listen", "lseek",
    "lstat64", "madvise", "mkdir", "mknod", "mlock", "mmap2", "mount", "mprotect",
    "mremap", "msync", "munlock", "munmap", "nanosleep", "open", "pipe", "pipe2", "poll",
    "prctl", "pread64", "ptrace", "pwrite64", "read", "readlink", "recvfrom", "recvmsg",
    "rename", "rmdir", "rt_sigprocmask", "rt_sigtimedwait", "sched_get_priority_max",
    "sched_get_priority_min", "sched_getparam", "sched_getscheduler", "sched_setaffinity",
    "sched_setscheduler", "sched_yield", "sendmsg", "sendto", "setgid32", "setgroups32",
    "setitimer", "setpgid", "setpriority", "setresgid32", "setresuid32", "setrlimit",
    "setsid", "setsockopt", "setuid32", "shutdown", "sigaction", "sigaltstack",
    "sigprocmask", "sigsuspend", "socket", "socketpair", "stat64", "statfs64", "symlink",
    "sysinfo", "timer_create", "timer_settime", "truncate", "ugetrlimit", "umask", "uname",
    "unlink", "utimes", "vfork", "wait4", "write", "writev"
]

# Streamlit App UI
st.title("Malware Attack Detection")
st.write("Enter feature values to predict malware category.")

# User Inputs
user_inputs = {col: st.number_input(f"{col}", value=0.0, step=0.01) for col in selected_feature_names}

# Convert input dictionary to DataFrame
input_df = pd.DataFrame([user_inputs])

# Ensure input matches trained feature order
input_df = input_df.reindex(columns=selected_feature_names, fill_value=0)

# Apply scaling
input_scaled = scaler.transform(input_df)

# Predict Button
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"The predicted malware category is: **{prediction[0]}**")
