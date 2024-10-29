#!/usr/bin/env python
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import BinaryIO

import argparse
import io
import json
from pathlib import Path
import re
from tqdm import tqdm
import sys


class EventInfo:
    __slots__ = [
        "name",
        "args",
        "desc",
        "scope",
        "ts_start",
        "ts_end",
        "pass_name",
        "pattern_name",
        "iteration",
    ]

    def __init__(self, ts_start: int, name: str, args: dict):
        self.ts_start = ts_start
        self.ts_end = None
        self.name = name
        self.scope: str | None = None
        self.args = args
        desc = self.desc = args.get("desc")
        self.pass_name = None
        self.pattern_name = None
        self.iteration = None
        # Unfortunately, MLIR emits all useful information free form with
        # desc strings. These even have typos.
        if name == "pass-execution" and (
            m := re.search(r"`pass-execution` running `([^`])`", desc)
        ):
            self.pass_name = m.group(1)
        elif name == "apply-pattern" and (
            m := re.search(r"apply-pattern pattern: (.*)", desc)
        ):
            self.pattern_name = m.group(1)
        elif name == "GreedyPatternRewriteIteration" and (
            m := re.search(r"GreedyPatternRewriteIteration\(([0-9]+)\)", desc)
        ):
            self.iteration = int(m.group(1))

    @property
    def duration(self) -> float:
        return (self.ts_end - self.ts_start) / 1000000.0


class ThreadState:
    __slots__ = [
        "tid",
        "event_stack",
        "pass_stack",
    ]

    def __init__(self, tid: int):
        self.tid = tid
        self.event_stack: list[EventInfo] = []
        self.pass_stack: list[str] = []

    def begin(self, event: EventInfo):
        scope = ""
        if self.pass_stack:
            scope = self.pass_stack[-1]
        event.scope = scope
        self.event_stack.append(event)
        if event.pass_name is not None:
            self.pass_stack.append(event.pass_name)

    def end(self, ts_end: int):
        event = self.event_stack.pop()
        event.ts_end = ts_end
        if event.pass_name is not None:
            self.pass_stack.pop()
        return event


class EventScope:
    def __init__(self, name: str):
        self.name = name

    def record_event(self, event: EventInfo): ...


class EventDb:
    def __init__(self):
        self.threads: dict[ThreadState] = {}
        self.scopes: dict[str, EventScope] = {}

    def get_scope(self, name: str) -> EventScope:
        try:
            scope = self.scopes[name]
        except KeyError:
            scope = self.scopes[name] = EventScope(name)
        return scope

    def record_event(self, event: EventInfo):
        self.get_scope("__GLOBAL__").record_event(event)
        scope = event.scope
        if scope is not None:
            self.get_scope(scope).record_event(event)

    def load_file(self, f: BinaryIO):
        # Because Chrome tracing files are mammoth when spewed onto this great
        # earth from MLIR, we take advantage of records being written one
        # per line and parse each line individually.
        f.seek(0, io.SEEK_END)
        file_size = f.tell()
        f.seek(0)
        print(f"Loading file: {f.name}")
        line_number = 0
        with tqdm(total=file_size, unit="bytes", unit_scale=True) as pbar:
            while line := f.readline():
                line_number += 1
                pbar.update(len(line))
                line = line.strip()
                # First/last line.
                if line.startswith(b"["):
                    line = line[1:]
                if line.endswith(b"]"):
                    line = line[0:-1]
                if line.endswith(b","):
                    line = line[0:-1]
                try:
                    record = json.loads(line)
                    assert isinstance(record, dict)
                except Exception as e:
                    print("error decoding record:", e)
                self.handle_record(record)

                # TEMP: Early exit for debugging
                if line_number > 1000000:
                    break

    def handle_record(self, record: dict):
        cat = record["cat"]
        if cat != "PERF":
            return
        name = record["name"]
        phase = record["ph"]
        tid = int(record["tid"])
        ts = int(record["ts"])
        args = record.get("args")
        if phase == "B":
            self.get_thread(tid).begin(EventInfo(ts, name, args))
        elif phase == "E":
            event = self.get_thread(tid).end(ts)
            self.record_event(event)

    def get_thread(self, tid: int) -> ThreadState:
        try:
            t = self.threads[tid]
        except KeyError:
            t = self.threads[tid] = ThreadState(tid)
        return t
    
    def write_reports(self, dest_dir: Path):
        ...


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        nargs="+",
        type=argparse.FileType("rb"),
        help="Chrome tracing input file",
    )
    parser.add_argument(
        "--dest", required=True, type=Path, help="Destination directory"
    )
    args = parser.parse_args(argv)
    db = EventDb()
    for f in args.input_file:
        db.load_file(f)

    dest_dir: Path = args.dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    db.write_reports(dest_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
