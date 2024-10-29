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
import math
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
        "ident",
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
        self.ident = None
        # Unfortunately, MLIR emits all useful information free form with
        # desc strings. These even have typos.
        if name == "pass-execution" and (
            m := re.search(r"`pass-execution` running `([^`]*)`", desc)
        ):
            self.pass_name = m.group(1)
            self.ident = f"pass.{self.pass_name}"
        elif name == "apply-pattern" and (
            m := re.search(r"apply-pattern pattern: (.*)", desc)
        ):
            self.pattern_name = m.group(1)
            self.ident = f"pattern.{self.pattern_name}"
        elif name == "GreedyPatternRewriteIteration" and (
            m := re.search(r"GreedyPatternRewriteIteration\(([0-9]+)\)", desc)
        ):
            self.iteration = int(m.group(1))
            self.ident = f"GreedyPatternRewriteIteration"
        else:
            print(f"warning: failed to categorize event: {self}")

    @property
    def duration(self) -> float:
        return (self.ts_end - self.ts_start) / 1000000.0

    def __repr__(self):
        return f"EventInfo(name={self.name}, args={self.args})"


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
            self.pass_stack.append(event.ident)

    def end(self, ts_end: int):
        event = self.event_stack.pop()
        event.ts_end = ts_end
        if event.pass_name is not None:
            self.pass_stack.pop()
        return event


class Stats:
    __slots__ = [
        "ident",
        "min",
        "max",
        "count",
        "sum",
        "sum_x2",
    ]

    def __init__(self, event: EventInfo):
        self.ident = event.ident
        initial = event.duration
        self.min = self.max = initial
        self.count = 1
        self.sum = initial
        self.sum_x2 = initial * initial

    def add(self, event: EventInfo):
        d = event.duration
        self.count += 1
        self.min = min(self.min, d)
        self.max = max(self.max, d)
        self.sum += d
        self.sum_x2 += d * d

    @property
    def mean(self) -> float:
        return self.sum / self.count

    @property
    def stddev(self) -> float:
        mean = self.mean
        return math.sqrt((self.sum_x2 / self.count) - (mean * mean))


class EventScope:
    __slots__ = [
        "db",
        "name",
        "stats",
    ]

    def __init__(self, db: "EventDb", name: str):
        self.db = db
        self.name = name
        self.stats: dict[str, Stats] = {}  # Keyed by event.ident

    def record_event(self, event: EventInfo):
        ident = event.ident
        if ident is None:
            print(f"warning: dropped event {event}")
            return
        row = self.stats.get(ident)
        if row is None:
            row = Stats(event)
            self.stats[ident] = row
        else:
            row.add(event)

    def write_report(self, report_path: Path):
        def w(s):
            out.write(s)

        def wl(s):
            out.write(s)
            out.write("\n")

        def link_scope(row: Stats):
            if row.ident in self.db.scopes:
                return f"<a href='{row.ident}.html'>{row.ident}</a>"
            else:
                return row.ident

        def start_stats_table():
            wl("<table>")
            wl("<tr>")
            wl("<th>Name</th>")
            wl("<th>Total</th>")
            wl("<th>Count</th>")
            wl("<th>MEAN</th>")
            wl("<th>MIN</th>")
            wl("<th>MAX</th>")
            wl("<th>STDDEV</th>")
            wl("</tr>")

        def end_stats_table():
            wl("</table>")

        def stats_row(row: Stats):
            sum_scale, sum_label = human_scale(row.sum)
            mean_scale, mean_label = human_scale(row.mean)
            wl("<tr>")
            wl(f"<td>{link_scope(row)}</td>")
            wl(f"<td>{row.sum * sum_scale} {sum_label}</td>")
            wl(f"<td>{row.count}</td>")
            wl(f"<td>{row.mean * mean_scale} {mean_label}</td>")
            wl(f"<td>{row.min * mean_scale}</td>")
            wl(f"<td>{row.max * mean_scale}</td>")
            wl(f"<td>{row.stddev * mean_scale}</td>")
            wl("</tr>")

        def stats_table(rows):
            start_stats_table()
            for row in rows:
                stats_row(row)
            end_stats_table()

        with open(report_path, "wt") as out:
            wl("<html>")
            wl(f"<head><title>Series report: {self.name}</title></head>")
            wl(f"<style>")
            wl(
                "table {border-width: thin; border-spacing: 2px; border-style: none; border-color: black; border-collapse: collapse;}"
            )
            wl("td, th { border: 1px solid black; }")
            wl(f"</style>")
            wl(f"<body>")

            pass_rows = [s for s in self.stats.values() if s.ident.startswith("pass.")]
            if pass_rows:
                wl(f"<h1>Passes in Execution Order:</h1>")
                stats_table(pass_rows)

                # Show in descending order of total cost
                rows_desc = sorted(pass_rows, key=lambda row: row.sum, reverse=True)
                wl(f"<h1>Passes in descending order of duration:</h1>")
                stats_table(rows_desc)

            iteration_rows = [
                s
                for s in self.stats.values()
                if s.ident in ["GreedyPatternRewriteIteration"]
            ]
            if iteration_rows:
                # Show in descending order of total cost
                rows_desc = sorted(
                    iteration_rows, key=lambda row: row.sum, reverse=True
                )
                wl(f"<h1>Iteration actions:</h1>")
                stats_table(rows_desc)

            pattern_rows = [
                s for s in self.stats.values() if s.ident.startswith("pattern.")
            ]
            if pattern_rows:
                # Show in descending order of total cost
                rows_desc = sorted(pattern_rows, key=lambda row: row.sum, reverse=True)
                wl(f"<h1>Patterns in descending order of duration:</h1>")
                stats_table(rows_desc)

            wl(f"</body>")
            wl("</html>")


class EventDb:
    __slots__ = [
        "threads",
        "scopes",
    ]

    def __init__(self):
        self.threads: dict[ThreadState] = {}
        self.scopes: dict[str, EventScope] = {}

    def get_scope(self, name: str) -> EventScope:
        try:
            scope = self.scopes[name]
        except KeyError:
            scope = self.scopes[name] = EventScope(self, name)
        return scope

    def record_event(self, event: EventInfo):
        self.get_scope("__GLOBAL__").record_event(event)
        scope = event.scope
        if scope is not None:
            self.get_scope(scope if scope else "__UNNAMED__").record_event(event)

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

                # DEBUGGING: Early exit for debugging
                # if line_number > 1000000:
                #     break

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
        for scope in self.scopes.values():
            if scope.name == "__GLOBAL__":
                file_name = "index.html"
            else:
                file_name = f"{scope.name}.html"
            report_path = dest_dir / file_name
            print(f"Writing report {report_path}")
            scope.write_report(report_path)


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


def human_scale(value: float) -> tuple[float, str]:
    if value > 0.1:
        return 1.0, "SECONDS"
    elif value > 0.0009:
        return 1000.0, "MILLIS"
    else:
        return 1000000.0, "MICROS"


if __name__ == "__main__":
    main(sys.argv[1:])
