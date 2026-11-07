"""Sample process RSS during a stage; reports measured memory, never tensor-only estimates."""

import os
import threading
import time
from pathlib import Path


def current_rss_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                          "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                          "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        kernel, process = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        process.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if process.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize
        return None
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return None


class StageResources:
    def __enter__(self):
        self.start, self.peak = time.perf_counter(), current_rss_bytes()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(.05):
                value = current_rss_bytes()
                if value is not None:
                    self.peak = max(self.peak or 0, value)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        final = current_rss_bytes()
        if final is not None:
            self.peak = max(self.peak or 0, final)
        self.report = {"wall_seconds": time.perf_counter() - self.start,
                       "peak_sampled_process_rss_bytes": self.peak, "sampling_interval_seconds": .05,
                       "memory_scope": "whole-process host RSS; includes previously loaded objects; sampled peak may miss transients"}
