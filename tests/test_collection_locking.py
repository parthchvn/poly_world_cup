"""Both shipped collectors must keep their writer lock portable and exclusive."""
from __future__ import annotations

import builtins
import errno
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULES = ("poly_world_cup/trades.py", "scripts/build_actor_dataset.py")


def load_collector(relative):
    name = "collection_lock_test_" + relative.replace("/", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=MODULES)
def collector(request):
    return load_collector(request.param), request.param


CHILD = r'''
import importlib.util, os, pathlib, sys
spec = importlib.util.spec_from_file_location("collector_lock_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
try:
    with module._writer_lock(pathlib.Path(sys.argv[2])):
        print("acquired", flush=True)
        if len(sys.argv) > 3:
            sys.stdin.buffer.read(1)
            os._exit(0)
except module.TradeIngestionError:
    print("contended", flush=True)
'''


def child_attempt(relative, directory):
    result = subprocess.run(
        [sys.executable, "-c", CHILD, str(ROOT / relative), str(directory)],
        capture_output=True, text=True, timeout=30, check=True, cwd=ROOT,
    )
    return result.stdout.strip()


def test_separate_process_contention_and_release(collector, tmp_path):
    module, relative = collector
    with module._writer_lock(tmp_path):
        assert child_attempt(relative, tmp_path) == "contended"
    assert child_attempt(relative, tmp_path) == "acquired"


def test_exception_releases_real_lock(collector, tmp_path):
    module, relative = collector
    with pytest.raises(RuntimeError, match="body failed"):
        with module._writer_lock(tmp_path):
            raise RuntimeError("body failed")
    assert child_attempt(relative, tmp_path) == "acquired"


def test_process_exit_releases_real_lock(collector, tmp_path):
    module, relative = collector
    process = subprocess.Popen(
        [sys.executable, "-c", CHILD, str(ROOT / relative), str(tmp_path), "crash"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=ROOT,
    )
    try:
        assert process.stdout.readline().strip() == "acquired"
        with pytest.raises(module.TradeIngestionError):
            with module._writer_lock(tmp_path):
                pytest.fail("A live child already owns this lock")
        process.communicate("x", timeout=30)
        assert process.returncode == 0
        with module._writer_lock(tmp_path):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=30)


def install_windows_mock(monkeypatch, module, fail_errno=None):
    calls = []
    acquire, release = 101, 102

    def locking(fd, operation, length):
        calls.append((operation, length, os.lseek(fd, 0, os.SEEK_CUR)))
        if operation == acquire and fail_errno is not None:
            raise OSError(fail_errno, "mock Windows lock failure")

    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        locking=locking, LK_NBLCK=acquire, LK_UNLCK=release,
    ))
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "fcntl":
            raise AssertionError("Windows path imported POSIX-only fcntl")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return calls, acquire, release


@pytest.mark.parametrize("body_error", [False, True])
def test_windows_locks_byte_zero_and_releases(collector, tmp_path, monkeypatch, body_error):
    module, _ = collector
    # Existing nonempty markers must not move the locked byte to the file end.
    (tmp_path / ".writer.lock").write_bytes(b"old marker")
    calls, acquire, release = install_windows_mock(monkeypatch, module)
    if body_error:
        with pytest.raises(RuntimeError, match="body failed"):
            with module._writer_lock(tmp_path):
                raise RuntimeError("body failed")
    else:
        with module._writer_lock(tmp_path):
            assert calls == [(acquire, 1, 0)]
    assert calls == [(acquire, 1, 0), (release, 1, 0)]


@pytest.mark.parametrize("error_number", sorted({errno.EACCES, errno.EAGAIN, errno.EDEADLK}))
def test_windows_contention_never_enters_or_unlocks(collector, tmp_path, monkeypatch, error_number):
    module, _ = collector
    calls, acquire, _ = install_windows_mock(monkeypatch, module, error_number)
    with pytest.raises(module.TradeIngestionError):
        with module._writer_lock(tmp_path):
            pytest.fail("Contended lock must not enter its protected body")
    assert calls == [(acquire, 1, 0)]


def test_windows_other_os_error_is_preserved(collector, tmp_path, monkeypatch):
    module, _ = collector
    calls, acquire, _ = install_windows_mock(monkeypatch, module, errno.EBADF)
    with pytest.raises(OSError) as caught:
        with module._writer_lock(tmp_path):
            pytest.fail("Failed lock must not enter its protected body")
    assert caught.value.errno == errno.EBADF
    assert calls == [(acquire, 1, 0)]
