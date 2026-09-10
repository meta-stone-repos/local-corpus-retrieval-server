from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    unique_mapping,
)


def mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def keys(value, allowed, label):
    mapping(value, label)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label}: unknown fields {sorted(unknown)}")


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def cli_arguments(options):
    result = []
    for key, value in options.items():
        if not isinstance(key, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", key
        ):
            raise ValueError(f"Invalid argument name: {key}")
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            result.append(flag)
        elif isinstance(value, (str, int, float)):
            result.append(f"{flag}={value}")
        else:
            raise ValueError(f"args.{key} must be a scalar")
    return result


def environment(runtime):
    keys(
        runtime,
        {"hf_home", "hf_offline", "cuda_visible_devices", "env"},
        "runtime",
    )
    env = dict(os.environ)
    extra = runtime.get("env", {})
    mapping(extra, "runtime.env")
    for key, value in extra.items():
        if not isinstance(key, str) or not isinstance(value, (str, int)):
            raise ValueError("runtime.env requires string names and scalar values")
        env[key] = str(value)

    if runtime.get("hf_home"):
        home = Path(runtime["hf_home"])
        if not home.is_absolute():
            home = ROOT / home
        env["HF_HOME"] = str(home)
        env["HF_HUB_CACHE"] = str(home / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = str(home / "hub")
        env["HF_DATASETS_CACHE"] = str(home / "datasets")
        # 避免旧变量将 tokenizer 指向另一个缓存位置。
        env.pop("TRANSFORMERS_CACHE", None)

    if "hf_offline" in runtime:
        if type(runtime["hf_offline"]) is not bool:
            raise ValueError("runtime.hf_offline must be true or false")
        value = "1" if runtime["hf_offline"] else "0"
        env["HF_HUB_OFFLINE"] = value
        env["TRANSFORMERS_OFFLINE"] = value

    if "cuda_visible_devices" in runtime:
        env["CUDA_VISIBLE_DEVICES"] = str(runtime["cuda_visible_devices"])

    env["PYTHONUNBUFFERED"] = "1"
    return env


def check_paths(options, backend):
    import glob

    corpus = options.get("corpus")
    if corpus:
        conflicting = {
            "index_path", "model_name", "pooling", "normalize",
            "torch_dtype", "lookup_path", "index_type", "nprobe",
            "task_prefix", "max_length",
        } & set(options)
        if conflicting:
            raise ValueError(
                f"corpus mode conflicts with explicit settings: {sorted(conflicting)}"
            )
        path = ROOT / "corpora" / corpus / "manifest.json"
        if not path.is_file():
            raise ValueError(f"Missing manifest: {path}")
        return

    for key in ("document_store", "corpus_path", "lookup_path"):
        value = options.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise ValueError(f"Missing {key}: {path}")

    if backend == "online":
        return

    value = options.get("index_path")
    if not value:
        raise ValueError("Offline service requires corpus or index_path")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path

    if backend == "bm25":
        if not path.is_dir():
            raise ValueError(f"Missing BM25 index directory: {path}")
    elif backend == "faiss":
        kind = options.get("index_type", "flat")
        if not options.get("model_name"):
            raise ValueError("Dense explicit mode requires model_name")
        if kind == "flat":
            if not glob.glob(str(path)):
                raise ValueError(f"No Dense shards match: {path}")
        elif kind == "ivf-flat":
            if not path.is_file() or not options.get("lookup_path"):
                raise ValueError("ivf-flat requires index_path and lookup_path")
        else:
            raise ValueError(f"Unsupported Dense index_type: {kind}")


def stop_processes(processes):
    for _, proc, _ in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if all(proc.poll() is not None for _, proc, _ in processes):
            break
        time.sleep(0.2)

    for _, proc, _ in processes:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()



# 运行记录按配置文件绝对路径隔离，不通过端口选择要停止的进程。
def process_identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return fields[19]
    except (OSError, IndexError):
        return None


def state_paths(config_path):
    import hashlib

    folder = ROOT / "logs" / "services" / ".state"
    folder.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        str(Path(config_path).resolve()).encode()
    ).hexdigest()
    return folder / f"{digest}.json", folder / f"{digest}.lock"


def manage_group(config_path, stop=False):
    state_path, _ = state_paths(config_path)
    if not state_path.exists():
        print("没有找到该配置的启动记录。")
        return

    data = json.loads(state_path.read_text())
    pid = data["pid"]
    if process_identity(pid) != data["identity"]:
        print("该配置的启动器已退出；不操作旧记录中的进程。")
        return

    if not stop:
        print(f"启动器运行中：pid={pid}")
        for name, item in data.get("services", {}).items():
            alive = process_identity(item["pid"]) == item["identity"]
            status = item["status"] if alive else "EXITED"
            print(
                f"{name}: {status}, pid={item['pid']}, "
                f"port={item['port']}, log={item['log']}"
            )
        print("READY 表示已通过启动就绪检查，不代表持续健康检查。")
        return

    # pidfd 绑定具体进程，避免 PID 被复用后误发信号。
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        print("启动器已退出。")
        return
    try:
        if process_identity(pid) != data["identity"]:
            print("进程身份已变化，未发送停止信号。")
            return
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)

    print("已请求停止该配置启动的整组服务……", flush=True)
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if process_identity(pid) != data["identity"]:
            print("启动器已退出。")
            return
        time.sleep(0.2)
    print("仍在退出中，请查看原启动终端；未向其他进程发送信号。")


class RunState:
    def __init__(self, config_path):
        import fcntl

        self.path, lock_path = state_paths(config_path)
        self.lock = lock_path.open("a")
        try:
            fcntl.flock(
                self.lock.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError(
                "该配置已有启动器运行，请先使用 --status 或 --stop"
            ) from None
        self.data = {
            "pid": os.getpid(),
            "identity": process_identity(os.getpid()),
            "services": {},
        }
        self.save()

    def save(self):
        temp = self.path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(json.dumps(self.data))
        os.replace(temp, self.path)

    def add(self, name, proc, port, logfile):
        self.data["services"][name] = {
            "pid": proc.pid,
            "identity": process_identity(proc.pid),
            "port": port,
            "log": str(logfile),
            "status": "STARTING",
        }
        self.save()

    def ready(self, name):
        self.data["services"][name]["status"] = "READY"
        self.save()


PROBE_CODE = """
import asyncio
import sys
from fastmcp import Client
from fastmcp.client.transports import SSETransport

async def main():
    url, transport, read_enabled = sys.argv[1:]
    target = SSETransport(url) if transport == "sse" else url
    async with Client(target) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools}
        expected = {"search"}
        if read_enabled == "1":
            expected.add("read_document")
        if not expected <= names:
            raise RuntimeError("Required tools are missing")

asyncio.run(main())
"""


def probe_service(options, read_enabled):
    host = options.get("host", "127.0.0.1")
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    url = f"http://{host}:{options['port']}/mcp"
    proc = subprocess.Popen(
        [
            sys.executable, "-c", PROBE_CODE,
            url, options.get("transport", "sse"),
            "1" if read_enabled else "0",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        return proc.wait(timeout=8) == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def monitor_services(processes, specs, state, startup_timeout):
    options_by_name = {
        name: (backend, options)
        for name, backend, options, _, _ in specs
    }
    ready = set()
    started = time.monotonic()
    last_update = started
    announced = False

    while True:
        for name, proc, logfile in processes:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"{name} exited with code {code}; log: {logfile}"
                )

        for name, proc, logfile in processes:
            if name in ready:
                continue
            backend, options = options_by_name[name]
            read_enabled = (
                backend == "online" or options.get("get_document", False)
            )
            if probe_service(options, read_enabled):
                if proc.poll() is not None:
                    raise RuntimeError(f"{name} exited during readiness check")
                ready.add(name)
                state.ready(name)
                print(
                    f"READY {name}: port={options['port']}, "
                    f"elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        if len(ready) == len(processes):
            if not announced:
                print(
                    "全部服务已就绪。Ctrl+C 或另一个终端使用 --stop 停止。",
                    flush=True,
                )
                announced = True
        else:
            elapsed = time.monotonic() - started
            pending = [name for name, _, _ in processes if name not in ready]
            if elapsed >= startup_timeout:
                raise RuntimeError(
                    f"启动超时：{', '.join(pending)}；请查看对应日志"
                )
            if time.monotonic() - last_update >= 15:
                print(
                    f"WAITING {', '.join(pending)}，已等待 {elapsed:.0f}s",
                    flush=True,
                )
                last_update = time.monotonic()

        time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--services", nargs="+")
    selection.add_argument("--all", action="store_true")
    parser.add_argument(
        "--check", action="store_true",
        help="Validate configuration and local paths without starting services",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true")
    action.add_argument("--stop", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=900)
    args = parser.parse_args()

    if args.status or args.stop:
        if args.services or args.all or args.check:
            parser.error("--status/--stop 管理整组，不与启动选项混用")
        manage_group(args.config, stop=args.stop)
        return
    if not args.services and not args.all:
        parser.error("启动时请指定 --services 或 --all")
    if args.startup_timeout <= 0:
        parser.error("--startup-timeout 必须大于 0")

    config = yaml.load(
        Path(args.config).read_text(encoding="utf-8"),
        Loader=UniqueLoader,
    )
    keys(config, {"version", "defaults", "services"}, "config")
    if config.get("version") != 1:
        raise ValueError("Expected version: 1")
    defaults = config.get("defaults", {})
    keys(defaults, {"runtime", "args"}, "defaults")
    services = mapping(config.get("services"), "services")
    names = list(services) if args.all else args.services
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one service, without duplicates")

    specs = []
    ports = set()
    for name in names:
        if name not in services:
            raise ValueError(f"Unknown service {name}; available: {list(services)}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Service names may contain letters, digits, _ and -")

        section = services[name]
        keys(section, {"backend", "runtime", "args", "provider"}, name)
        merged = merge(defaults, section)
        backend = merged.get("backend")
        if backend not in {"bm25", "faiss", "online"}:
            raise ValueError(f"{name}: unsupported backend {backend}")
        options = dict(mapping(merged.get("args", {}), f"{name}.args"))
        if "config" in options or "searcher_type" in options:
            raise ValueError("Use backend/provider instead of args.config/searcher_type")
        if "hf_token" in options:
            raise ValueError("Place HF_TOKEN in runtime.env instead of args")
        port = options.get("port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError(f"{name}: port must be an integer in 1..65535")
        if port in ports:
            raise ValueError(f"Selected services share port {port}")
        ports.add(port)
        if options.get("transport", "sse") not in {"sse", "streamable-http"}:
            raise ValueError("Grouped launcher supports sse or streamable-http")
        if backend == "online":
            provider = mapping(merged.get("provider"), f"{name}.provider")
            if not provider.get("name"):
                raise ValueError("Online provider requires name")
        elif "provider" in merged:
            raise ValueError("provider is only valid for online services")

        cli_arguments(options)
        check_paths(options, backend)
        env = environment(merged.get("runtime", {}))
        specs.append((name, backend, options, merged.get("provider"), env))
        print(f"CHECK {name}: backend={backend}, port={port}", flush=True)

    if args.check:
        print("配置和本地路径检查通过；未加载模型、未调用 API。")
        return

    run_state = RunState(args.config)

    # 启动前检查端口，不接管已存在的服务。
    for name, _, options, _, _ in specs:
        with socket.socket() as sock:
            try:
                sock.bind(("0.0.0.0", options["port"]))
            except OSError as exc:
                raise ValueError(
                    f"{name}: port {options['port']} unavailable"
                ) from exc

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = ROOT / "logs" / "services" / f"{stamp}-{os.getpid()}"
    logdir.mkdir(parents=True)
    processes = []
    handles = []

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        with tempfile.TemporaryDirectory(prefix="retrieval-config-") as temp:
            try:
                for name, backend, options, provider, env in specs:
                    options = dict(options)
                    if backend == "online":
                        provider_options = dict(provider)
                        provider_name = provider_options.pop("name")
                        internal = {
                            "provider": provider_name,
                            "providers": {provider_name: provider_options},
                        }
                        path = Path(temp) / f"{name}.json"
                        fd = os.open(
                            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                        )
                        with os.fdopen(fd, "w") as stream:
                            json.dump(internal, stream)
                        options["config"] = str(path)
                        script = ROOT / "searcher" / "online_mcp_server.py"
                    else:
                        options["searcher_type"] = backend
                        script = ROOT / "searcher" / "mcp_server.py"

                    logfile = logdir / f"{name}.log"
                    handle = logfile.open("w")
                    handles.append(handle)
                    proc = subprocess.Popen(
                        [sys.executable, str(script), *cli_arguments(options)],
                        cwd=ROOT,
                        env=env,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    processes.append((name, proc, logfile))
                    run_state.add(name, proc, options["port"], logfile)
                    print(
                        f"STARTING {name}: pid={proc.pid}, "
                        f"port={options['port']}, log={logfile}",
                        flush=True,
                    )

                monitor_services(
                    processes, specs, run_state, args.startup_timeout
                )
            finally:
                stop_processes(processes)
    except KeyboardInterrupt:
        print("\n本次启动的服务已停止。", flush=True)
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
