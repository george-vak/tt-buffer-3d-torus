from __future__ import annotations

import csv
import heapq
import math
import random
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

Node = Tuple[int, int, int]
Direction = Tuple[int, int, int]
PortKey = Tuple[Node, Direction]

DIRECTIONS: List[Direction] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


class Scenario(str, Enum):
    SP_FP = "SP+FP"
    TAS = "TAS"
    CQF = "CQF"


class TrafficClass(str, Enum):
    TT = "TT"
    BE = "BE"


@dataclass
class Flow:
    flow_id: str
    traffic_class: TrafficClass
    src: Node
    dst: Node
    period: float
    frame_size_bits: int
    start_time: float = 0.0
    phase: float = 0.0


@dataclass
class Frame:
    flow_id: str
    seq: int
    traffic_class: TrafficClass
    src: Node
    dst: Node
    size_bits: int
    remaining_bits: int
    gen_time: float
    path: List[Node]
    hop_index: int = 0
    regulated_port: Optional[PortKey] = None


@dataclass
class PortStats:
    max_tt_queue_bits: int = 0
    max_tt_total_bits: int = 0
    tt_drops: int = 0
    tt_transmitted: int = 0
    be_transmitted: int = 0


@dataclass
class Port:
    node: Node
    direction: Direction
    tt_queue: Deque[Frame] = field(default_factory=deque)
    be_queue: Deque[Frame] = field(default_factory=deque)
    cqf_queues: List[Deque[Frame]] = field(default_factory=lambda: [deque(), deque()])
    tt_queue_bits: int = 0
    cqf_queue_bits: List[int] = field(default_factory=lambda: [0, 0])
    busy_until: float = 0.0
    next_wakeup_time: Optional[float] = None
    tt_reg_next_time: Dict[str, float] = field(default_factory=dict)
    ats_queues: Dict[str, Deque[Frame]] = field(default_factory=lambda: defaultdict(deque))
    ats_release_scheduled: Dict[str, bool] = field(default_factory=dict)
    stats: PortStats = field(default_factory=PortStats)


@dataclass
class SimulationStats:
    scenario: str
    n_gen_tt: int
    delivered_tt: int
    dropped_tt: int
    loss_ratio: float
    min_delay: Optional[float]
    avg_delay: Optional[float]
    max_delay: Optional[float]
    max_tt_queue_bits: int
    max_tt_queue_bytes: float
    max_tt_total_bits: int
    max_tt_total_bytes: float
    per_flow_delays: Dict[str, List[float]]
    per_port_stats: Dict[PortKey, PortStats]


class TorusNetworkSimulator:
    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        link_rate_bps: float,
        d_prop: float,
        d_sw: float,
        tt_buffer_bits: int,
        scenario: Scenario,
        l_np_bits: Optional[int] = None,
        tas_period: Optional[float] = None,
        tas_tt_window: Optional[float] = None,
        cqf_cycle: Optional[float] = None,
        enable_tt_regulation: bool = True,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.C = link_rate_bps
        self.d_prop = d_prop
        self.d_sw = d_sw
        self.tt_buffer_bits = tt_buffer_bits
        self.scenario = scenario
        self.l_np_bits = l_np_bits
        self.tas_period = tas_period
        self.tas_tt_window = tas_tt_window
        self.cqf_cycle = cqf_cycle
        self.enable_tt_regulation = enable_tt_regulation

        if scenario == Scenario.SP_FP and l_np_bits is None:
            raise ValueError("Для SP+FP необходимо задать l_np_bits.")
        if scenario == Scenario.TAS:
            if tas_period is None or tas_tt_window is None:
                raise ValueError("Для TAS необходимо задать tas_period и tas_tt_window.")
            if tas_tt_window >= tas_period:
                raise ValueError("Для TAS должно выполняться tas_tt_window < tas_period.")
        if scenario == Scenario.CQF and cqf_cycle is None:
            raise ValueError("Для CQF необходимо задать cqf_cycle.")

        self.ports: Dict[PortKey, Port] = {}
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    node = (x, y, z)
                    for direction in DIRECTIONS:
                        self.ports[(node, direction)] = Port(node=node, direction=direction)

        self.flows: List[Flow] = []
        self.flow_periods: Dict[str, float] = {}
        self.event_queue: List[Tuple[float, int, str, object]] = []
        self.event_seq = 0
        self.per_flow_delays: Dict[str, List[float]] = defaultdict(list)

        self.n_gen_tt = 0
        self.n_gen_tt_target: Optional[int] = None
        self.generation_closed = False
        self.generation_stop_time: Optional[float] = None

        self.dropped_tt = 0
        self.delivered_tt = 0

    def add_flow(self, flow: Flow) -> None:
        if flow.src == flow.dst:
            raise ValueError("src и dst должны различаться.")
        self.flows.append(flow)
        self.flow_periods[flow.flow_id] = flow.period

    @staticmethod
    def _axis_step(src_coord: int, dst_coord: int, n: int) -> int:
        forward = (dst_coord - src_coord) % n
        backward = (src_coord - dst_coord) % n
        if forward == 0:
            return 0
        return 1 if forward <= backward else -1

    def xyz_route(self, src: Node, dst: Node) -> List[Node]:
        path = [src]
        cur = src
        while cur[0] != dst[0]:
            step = self._axis_step(cur[0], dst[0], self.nx)
            cur = ((cur[0] + step) % self.nx, cur[1], cur[2])
            path.append(cur)
        while cur[1] != dst[1]:
            step = self._axis_step(cur[1], dst[1], self.ny)
            cur = (cur[0], (cur[1] + step) % self.ny, cur[2])
            path.append(cur)
        while cur[2] != dst[2]:
            step = self._axis_step(cur[2], dst[2], self.nz)
            cur = (cur[0], cur[1], (cur[2] + step) % self.nz)
            path.append(cur)
        return path

    @staticmethod
    def direction_between(a: Node, b: Node, nx: int, ny: int, nz: int) -> Direction:
        ax, ay, az = a
        bx, by, bz = b
        if ay == by and az == bz:
            if (ax + 1) % nx == bx:
                return (1, 0, 0)
            if (ax - 1) % nx == bx:
                return (-1, 0, 0)
        if ax == bx and az == bz:
            if (ay + 1) % ny == by:
                return (0, 1, 0)
            if (ay - 1) % ny == by:
                return (0, -1, 0)
        if ax == bx and ay == by:
            if (az + 1) % nz == bz:
                return (0, 0, 1)
            if (az - 1) % nz == bz:
                return (0, 0, -1)
        raise ValueError(f"Узлы {a} и {b} не являются соседями.")

    def schedule(self, time: float, event_type: str, payload: object) -> None:
        heapq.heappush(self.event_queue, (time, self.event_seq, event_type, payload))
        self.event_seq += 1

    def schedule_port_wakeup(self, time: float, port: Port) -> None:
        if time <= 0:
            return
        if port.next_wakeup_time is None or abs(port.next_wakeup_time - time) > 1e-15:
            port.next_wakeup_time = time
            self.schedule(time, "port_wakeup", port)

    def initialize_generation(self) -> None:
        for flow in self.flows:
            first_time = flow.start_time + flow.phase
            self.schedule(first_time, "generate_frame", (flow, 0))

    def run(self, n_gen_tt_target: int, drain_time: float = 0.0) -> SimulationStats:
        self.n_gen_tt_target = n_gen_tt_target
        self.initialize_generation()

        while self.event_queue:
            time, _, event_type, payload = heapq.heappop(self.event_queue)

            if self.generation_closed and self.generation_stop_time is not None:
                if time > self.generation_stop_time + drain_time:
                    break

            if event_type == "generate_frame":
                flow, seq = payload
                self.handle_generate_frame(time, flow, seq)
            elif event_type == "enqueue_frame":
                self.handle_enqueue_frame(time, payload)
            elif event_type == "tx_complete":
                port, frame, bits_sent = payload
                self.handle_tx_complete(time, port, frame, bits_sent)
            elif event_type == "ats_release":
                node, direction, flow_id = payload
                self.handle_ats_release(time, node, direction, flow_id)
            elif event_type == "port_wakeup":
                port = payload
                port.next_wakeup_time = None
                self.try_start_transmission(time, port)
            else:
                raise ValueError(f"Неизвестный тип события: {event_type}")

        all_delays = [d for delays in self.per_flow_delays.values() for d in delays]
        per_port_stats = {key: port.stats for key, port in self.ports.items()}
        loss_ratio = self.dropped_tt / self.n_gen_tt if self.n_gen_tt > 0 else 0.0
        max_tt_queue_bits = max((p.stats.max_tt_queue_bits for p in self.ports.values()), default=0)
        max_tt_total_bits = max((p.stats.max_tt_total_bits for p in self.ports.values()), default=0)

        return SimulationStats(
            scenario=self.scenario.value,
            n_gen_tt=self.n_gen_tt,
            delivered_tt=self.delivered_tt,
            dropped_tt=self.dropped_tt,
            loss_ratio=loss_ratio,
            min_delay=min(all_delays) if all_delays else None,
            avg_delay=sum(all_delays) / len(all_delays) if all_delays else None,
            max_delay=max(all_delays) if all_delays else None,
            max_tt_queue_bits=max_tt_queue_bits,
            max_tt_queue_bytes=max_tt_queue_bits / 8,
            max_tt_total_bits=max_tt_total_bits,
            max_tt_total_bytes=max_tt_total_bits / 8,
            per_flow_delays=dict(self.per_flow_delays),
            per_port_stats=per_port_stats,
        )

    def handle_generate_frame(self, time: float, flow: Flow, seq: int) -> None:
        if self.generation_closed:
            return

        if flow.traffic_class == TrafficClass.TT:
            assert self.n_gen_tt_target is not None
            if self.n_gen_tt >= self.n_gen_tt_target:
                self.close_generation(time)
                return
            self.n_gen_tt += 1

        path = self.xyz_route(flow.src, flow.dst)
        frame = Frame(
            flow_id=flow.flow_id,
            seq=seq,
            traffic_class=flow.traffic_class,
            src=flow.src,
            dst=flow.dst,
            size_bits=flow.frame_size_bits,
            remaining_bits=flow.frame_size_bits,
            gen_time=time,
            path=path,
            hop_index=0,
        )
        self.schedule(time + self.d_sw, "enqueue_frame", frame)

        if flow.traffic_class == TrafficClass.TT and self.n_gen_tt >= self.n_gen_tt_target:
            self.close_generation(time)
            return

        if not self.generation_closed:
            self.schedule(time + flow.period, "generate_frame", (flow, seq + 1))

    def close_generation(self, time: float) -> None:
        self.generation_closed = True
        self.generation_stop_time = time

    def handle_enqueue_frame(self, time: float, frame: Frame) -> None:
        current_node = frame.path[frame.hop_index]

        if current_node == frame.dst:
            if frame.traffic_class == TrafficClass.TT:
                self.delivered_tt += 1
                self.per_flow_delays[frame.flow_id].append(time - frame.gen_time)
            return

        next_node = frame.path[frame.hop_index + 1]
        direction = self.direction_between(current_node, next_node, self.nx, self.ny, self.nz)
        port = self.ports[(current_node, direction)]

        if self.enqueue_to_ats_if_needed(time, port, frame):
            return

        accepted = self.enqueue_to_port_at_time(time, port, frame)
        if not accepted:
            if frame.traffic_class == TrafficClass.TT:
                self.dropped_tt += 1
                port.stats.tt_drops += 1
            return

        self.try_start_transmission(time, port)

    def enqueue_to_ats_if_needed(self, time: float, port: Port, frame: Frame) -> bool:
        if frame.traffic_class != TrafficClass.TT or not self.enable_tt_regulation:
            return False

        port.ats_queues[frame.flow_id].append(frame)

        if not port.ats_release_scheduled.get(frame.flow_id, False):
            self.schedule_next_ats_release(time, port, frame.flow_id)

        return True

    def schedule_next_ats_release(self, time: float, port: Port, flow_id: str) -> None:
        queue = port.ats_queues.get(flow_id)
        if not queue:
            port.ats_release_scheduled[flow_id] = False
            return

        period = self.flow_periods.get(flow_id)
        if period is None:
            release_time = time
        else:
            release_time = max(time, port.tt_reg_next_time.get(flow_id, time))
            port.tt_reg_next_time[flow_id] = release_time + period

        port.ats_release_scheduled[flow_id] = True
        self.schedule(release_time, "ats_release", (port.node, port.direction, flow_id))

    def handle_ats_release(self, time: float, node: Node, direction: Direction, flow_id: str) -> None:
        port = self.ports[(node, direction)]
        queue = port.ats_queues.get(flow_id)

        if not queue:
            port.ats_release_scheduled[flow_id] = False
            return

        frame = queue.popleft()
        accepted = self.enqueue_to_port_at_time(time, port, frame)

        if not accepted:
            self.dropped_tt += 1
            port.stats.tt_drops += 1
        else:
            self.try_start_transmission(time, port)

        port.ats_release_scheduled[flow_id] = False
        if queue:
            self.schedule_next_ats_release(time, port, flow_id)

    def need_delay_by_regulator(
        self,
        time: float,
        current_node: Node,
        direction: Direction,
        port: Port,
        frame: Frame,
    ) -> bool:
        if frame.traffic_class != TrafficClass.TT or not self.enable_tt_regulation:
            return False

        port_key = (current_node, direction)
        if frame.regulated_port == port_key:
            frame.regulated_port = None
            return False

        period = self.flow_periods.get(frame.flow_id)
        if period is None:
            return False

        previous_next_time = port.tt_reg_next_time.get(frame.flow_id, time)
        release_time = max(time, previous_next_time)
        port.tt_reg_next_time[frame.flow_id] = release_time + period

        if release_time > time:
            frame.regulated_port = port_key
            self.schedule(release_time, "enqueue_frame", frame)
            return True
        return False

    def handle_tx_complete(self, time: float, port: Port, frame: Frame, bits_sent: int) -> None:
        frame.remaining_bits -= bits_sent

        if frame.remaining_bits > 0:
            port.be_queue.appendleft(frame)
            self.try_start_transmission(time, port)
            return

        if frame.traffic_class == TrafficClass.TT:
            port.stats.tt_transmitted += 1
        else:
            port.stats.be_transmitted += 1

        next_node = frame.path[frame.hop_index + 1]
        frame.hop_index += 1
        arrival_time = time + self.d_prop

        if next_node == frame.dst:
            self.schedule(arrival_time, "enqueue_frame", frame)
        else:
            self.schedule(arrival_time + self.d_sw, "enqueue_frame", frame)

        self.try_start_transmission(time, port)

    def enqueue_to_port(self, port: Port, frame: Frame) -> bool:
        if frame.traffic_class == TrafficClass.BE:
            port.be_queue.append(frame)
            return True

        if self.scenario == Scenario.CQF:
            rx_index = self.cqf_rx_index(frame.gen_time if False else 0.0)
            raise RuntimeError("Internal CQF enqueue must receive current time. Use enqueue_to_port_at_time.")

        if port.tt_queue_bits + frame.remaining_bits > self.tt_buffer_bits:
            return False
        port.tt_queue.append(frame)
        port.tt_queue_bits += frame.remaining_bits
        self.update_standard_tt_stats(port)
        return True

    def enqueue_to_port_at_time(self, time: float, port: Port, frame: Frame) -> bool:
        if frame.traffic_class == TrafficClass.BE:
            port.be_queue.append(frame)
            return True

        if self.scenario == Scenario.CQF:
            rx_index = self.cqf_rx_index(time)
            per_queue_limit_bits = self.tt_buffer_bits // 2
            if port.cqf_queue_bits[rx_index] + frame.remaining_bits > per_queue_limit_bits:
                return False
            port.cqf_queues[rx_index].append(frame)
            port.cqf_queue_bits[rx_index] += frame.remaining_bits
            self.update_cqf_stats(port)
            return True

        if port.tt_queue_bits + frame.remaining_bits > self.tt_buffer_bits:
            return False
        port.tt_queue.append(frame)
        port.tt_queue_bits += frame.remaining_bits
        self.update_standard_tt_stats(port)
        return True

    def try_start_transmission(self, time: float, port: Port) -> None:
        if time < port.busy_until:
            return

        if self.scenario == Scenario.SP_FP:
            frame, bits_to_send, next_wakeup = self.select_sp_fp(port)
        elif self.scenario == Scenario.TAS:
            frame, bits_to_send, next_wakeup = self.select_tas(time, port)
        elif self.scenario == Scenario.CQF:
            frame, bits_to_send, next_wakeup = self.select_cqf(time, port)
        else:
            raise ValueError(f"Неизвестный сценарий: {self.scenario}")

        if frame is None:
            if next_wakeup is not None and next_wakeup > time:
                self.schedule_port_wakeup(next_wakeup, port)
            return

        tx_time = bits_to_send / self.C
        finish_time = time + tx_time
        port.busy_until = finish_time
        self.schedule(finish_time, "tx_complete", (port, frame, bits_to_send))

    def select_sp_fp(self, port: Port) -> Tuple[Optional[Frame], int, Optional[float]]:
        if port.tt_queue:
            frame = port.tt_queue.popleft()
            port.tt_queue_bits -= frame.remaining_bits
            self.update_standard_tt_stats(port)
            return frame, frame.remaining_bits, None

        if port.be_queue:
            frame = port.be_queue.popleft()
            bits_to_send = min(frame.remaining_bits, self.l_np_bits)
            return frame, bits_to_send, None

        return None, 0, None

    def select_tas(self, time: float, port: Port) -> Tuple[Optional[Frame], int, Optional[float]]:
        assert self.tas_period is not None
        assert self.tas_tt_window is not None

        phase = time % self.tas_period
        tt_open = phase < self.tas_tt_window

        if tt_open:
            remaining_window = self.tas_tt_window - phase
            if port.tt_queue:
                frame = port.tt_queue[0]
                tx_time = frame.remaining_bits / self.C
                if tx_time <= remaining_window:
                    port.tt_queue.popleft()
                    port.tt_queue_bits -= frame.remaining_bits
                    self.update_standard_tt_stats(port)
                    return frame, frame.remaining_bits, None
                next_time = time + remaining_window + (self.tas_period - self.tas_tt_window)
                return None, 0, next_time
            return None, 0, time + remaining_window

        time_to_next_tt_window = self.tas_period - phase
        next_tt_window = time + time_to_next_tt_window
        if port.be_queue:
            frame = port.be_queue[0]
            tx_time = frame.remaining_bits / self.C
            if tx_time <= time_to_next_tt_window:
                port.be_queue.popleft()
                return frame, frame.remaining_bits, None
        return None, 0, next_tt_window

    def select_cqf(self, time: float, port: Port) -> Tuple[Optional[Frame], int, Optional[float]]:
        tx_index = self.cqf_tx_index(time)
        rx_index = self.cqf_rx_index(time)
        next_boundary = self.next_cqf_boundary(time)

        if port.cqf_queues[tx_index]:
            frame = port.cqf_queues[tx_index][0]
            tx_time = frame.remaining_bits / self.C
            if time + tx_time <= next_boundary:
                port.cqf_queues[tx_index].popleft()
                port.cqf_queue_bits[tx_index] -= frame.remaining_bits
                self.update_cqf_stats(port)
                return frame, frame.remaining_bits, None
            return None, 0, next_boundary

        if port.cqf_queues[rx_index]:
            return None, 0, next_boundary

        if port.be_queue:
            frame = port.be_queue[0]
            tx_time = frame.remaining_bits / self.C
            if time + tx_time <= next_boundary:
                port.be_queue.popleft()
                return frame, frame.remaining_bits, None
            return None, 0, next_boundary

        return None, 0, None

    def cqf_cycle_number(self, time: float) -> int:
        assert self.cqf_cycle is not None
        return int(math.floor(time / self.cqf_cycle))

    def cqf_rx_index(self, time: float) -> int:
        return self.cqf_cycle_number(time) % 2

    def cqf_tx_index(self, time: float) -> int:
        return 1 - self.cqf_rx_index(time)

    def next_cqf_boundary(self, time: float) -> float:
        assert self.cqf_cycle is not None
        return (self.cqf_cycle_number(time) + 1) * self.cqf_cycle

    def update_standard_tt_stats(self, port: Port) -> None:
        q = port.tt_queue_bits
        port.stats.max_tt_queue_bits = max(port.stats.max_tt_queue_bits, q)
        port.stats.max_tt_total_bits = max(port.stats.max_tt_total_bits, q)

    def update_cqf_stats(self, port: Port) -> None:
        q0 = port.cqf_queue_bits[0]
        q1 = port.cqf_queue_bits[1]
        port.stats.max_tt_queue_bits = max(port.stats.max_tt_queue_bits, q0, q1)
        port.stats.max_tt_total_bits = max(port.stats.max_tt_total_bits, q0 + q1)



def all_nodes(nx: int, ny: int, nz: int) -> List[Node]:
    return [(x, y, z) for x in range(nx) for y in range(ny) for z in range(nz)]


def create_random_flows(
    nx: int,
    ny: int,
    nz: int,
    count: int,
    traffic_class: TrafficClass,
    period_range: Tuple[float, float],
    frame_size_bits: int,
    seed: int,
    flow_prefix: str,
) -> List[Flow]:
    rng = random.Random(seed)
    nodes = all_nodes(nx, ny, nz)
    flows: List[Flow] = []

    for i in range(count):
        src = rng.choice(nodes)
        dst = rng.choice(nodes)
        while dst == src:
            dst = rng.choice(nodes)

        period = rng.uniform(period_range[0], period_range[1])
        flows.append(
            Flow(
                flow_id=f"{flow_prefix}_{i}",
                traffic_class=traffic_class,
                src=src,
                dst=dst,
                period=period,
                frame_size_bits=frame_size_bits,
                phase=rng.uniform(0.0, period),
            )
        )

    return flows


def ceil_to_frame_bytes(bits: float, frame_size_bits: int) -> int:
    bytes_value = math.ceil(bits / 8)
    frame_bytes = math.ceil(frame_size_bits / 8)
    return int(math.ceil(bytes_value / frame_bytes) * frame_bytes)


def candidate_buffers_from_analytical(
    b_analytical_bytes: int,
    frame_size_bits: int,
) -> List[Tuple[float, int]]:
    factors = [0.50, 1.00, 1.15]
    result: List[Tuple[float, int]] = []
    used = set()

    for factor in factors:
        candidate_bits = b_analytical_bytes * 8 * factor
        value = max(
            math.ceil(frame_size_bits / 8),
            ceil_to_frame_bytes(candidate_bits, frame_size_bits),
        )
        if value not in used:
            result.append((factor, value))
            used.add(value)

    return result


def compute_port_traffic_parameters(
    nx: int,
    ny: int,
    nz: int,
    flows: List[Flow],
    router: TorusNetworkSimulator,
) -> Dict[PortKey, Dict[str, float]]:
    rho = defaultdict(float)
    sum_l = defaultdict(float)

    for flow in flows:
        if flow.traffic_class != TrafficClass.TT:
            continue
        path = router.xyz_route(flow.src, flow.dst)
        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]
            direction = router.direction_between(current_node, next_node, nx, ny, nz)
            key = (current_node, direction)
            rho[key] += flow.frame_size_bits / flow.period
            sum_l[key] += flow.frame_size_bits

    params: Dict[PortKey, Dict[str, float]] = {}
    for key in set(rho.keys()) | set(sum_l.keys()):
        params[key] = {
            "rho": rho[key],
            "b": sum_l[key],
        }
    return params


def analytical_buffer_fp_bits(
    port_params: Dict[PortKey, Dict[str, float]],
    l_np_bits: int,
    C: float,
) -> float:
    return max(
        (p["b"] + p["rho"] * (l_np_bits / C) for p in port_params.values()),
        default=0.0,
    )


def analytical_buffer_tas_bits(
    port_params: Dict[PortKey, Dict[str, float]],
    tas_period: float,
    tas_window: float,
) -> float:
    theta = tas_period - tas_window
    return max(
        (p["b"] + p["rho"] * theta for p in port_params.values()),
        default=0.0,
    )


def analytical_buffer_cqf_bytes(
    port_params: Dict[PortKey, Dict[str, float]],
    cqf_cycle: float,
    frame_size_bits: int,
) -> int:
    one_queue_bits = max(
        (p["b"] + p["rho"] * cqf_cycle for p in port_params.values()),
        default=0.0,
    )
    one_queue_bytes = ceil_to_frame_bytes(one_queue_bits, frame_size_bits)
    return 2 * one_queue_bytes


def check_tas_stability(
    port_params: Dict[PortKey, Dict[str, float]],
    C: float,
    tas_period: float,
    tas_window: float,
) -> bool:
    c_eff = C * tas_window / tas_period
    return all(p["rho"] < c_eff for p in port_params.values())


def check_cqf_feasibility(
    port_params: Dict[PortKey, Dict[str, float]],
    C: float,
    cqf_cycle: float,
) -> bool:
    return all(p["rho"] * cqf_cycle + p["b"] <= C * cqf_cycle for p in port_params.values())


def run_single_configuration(
    *,
    scenario: Scenario,
    nx: int,
    ny: int,
    nz: int,
    C: float,
    d_prop: float,
    d_sw: float,
    tt_buffer_bits: int,
    tt_flows: List[Flow],
    be_flows: List[Flow],
    n_gen_tt_target: int,
    drain_time: float,
    l_np_bits: Optional[int] = None,
    tas_period: Optional[float] = None,
    tas_tt_window: Optional[float] = None,
    cqf_cycle: Optional[float] = None,
    enable_tt_regulation: bool = True,
) -> SimulationStats:
    sim = TorusNetworkSimulator(
        nx=nx,
        ny=ny,
        nz=nz,
        link_rate_bps=C,
        d_prop=d_prop,
        d_sw=d_sw,
        tt_buffer_bits=tt_buffer_bits,
        scenario=scenario,
        l_np_bits=l_np_bits,
        tas_period=tas_period,
        tas_tt_window=tas_tt_window,
        cqf_cycle=cqf_cycle,
        enable_tt_regulation=enable_tt_regulation,
    )

    for flow in tt_flows + be_flows:
        sim.add_flow(flow)

    return sim.run(n_gen_tt_target=n_gen_tt_target, drain_time=drain_time)


def make_row(
    seed_index: int,
    scenario: str,
    scenario_param: str,
    factor: float,
    buffer_bytes: int,
    analytical_buffer_bytes: int,
    condition_ok: bool,
    stats: SimulationStats,
) -> Dict[str, object]:
    return {
        "seed": seed_index,
        "scenario": scenario,
        "scenario_param": scenario_param,
        "buffer_factor": factor,
        "buffer_bytes": buffer_bytes,
        "buffer_kbytes": buffer_bytes / 1024,
        "analytical_buffer_bytes": analytical_buffer_bytes,
        "analytical_buffer_kbytes": analytical_buffer_bytes / 1024,
        "condition_ok": condition_ok,
        "n_gen_tt": stats.n_gen_tt,
        "delivered_tt": stats.delivered_tt,
        "dropped_tt": stats.dropped_tt,
        "loss_ratio": stats.loss_ratio,
        "avg_delay_us": stats.avg_delay * 1e6 if stats.avg_delay is not None else "",
        "max_delay_us": stats.max_delay * 1e6 if stats.max_delay is not None else "",
    }


def print_result(
    seed_index: int,
    scenario_label: str,
    factor: float,
    buffer_bytes: int,
    analytical_buffer_bytes: int,
    stats: SimulationStats,
) -> None:
    max_delay_us = stats.max_delay * 1e6 if stats.max_delay is not None else 0.0
    avg_delay_us = stats.avg_delay * 1e6 if stats.avg_delay is not None else 0.0

    print(
        f"seed={seed_index:>2} | "
        f"{scenario_label:<20} | "
        f"k={factor:>4.2f} | "
        f"B={buffer_bytes / 1024:>6.1f} KB | "
        f"B_an={analytical_buffer_bytes / 1024:>6.1f} KB | "
        f"NgenTT={stats.n_gen_tt:>7} | "
        f"dropTT={stats.dropped_tt:>5} | "
        f"lossTT={stats.loss_ratio:.3e} | "
        f"avgD={avg_delay_us:.3f} us | "
        f"maxD={max_delay_us:.3f} us"
    )


def save_results_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment() -> None:
    nx, ny, nz = 6, 6, 6
    C = 10_000_000_000
    d_prop = 1e-6
    d_sw = 0.5e-6

    tt_frame_size_bits = 1500 * 8
    be_frame_size_bits = 1500 * 8

    tt_flow_count = 100
    be_flow_count = 100

    tt_period_range = (100e-6, 500e-6)
    be_period_range = (50e-6, 200e-6)

    l_np_bits = 128 * 8

    tas_period = 100e-6
    tas_windows = [70e-6, 80e-6, 90e-6]

    cqf_cycles = [120e-6, 160e-6, 200e-6]

    epsilon_target = 1e-5
    n_gen_tt_target = 30000

    # n_gen_tt_target = int(3 / epsilon_target)

    drain_time = 0.02
    seeds = [1, 3]
    enable_tt_regulation = True

    output_path = Path("simulation_results_final_ats_6x6x6.csv")
    rows: List[Dict[str, object]] = []

    print("Параметры эксперимента")
    print(f"Размер тора: {nx}x{ny}x{nz}, N_sw={nx * ny * nz}")
    print(f"TT flows={tt_flow_count}, BE flows={be_flow_count}")
    print(f"Целевое N_gen^TT={n_gen_tt_target}")
    print()

    for seed_index, seed in enumerate(seeds, start=1):
        tt_flows = create_random_flows(
            nx=nx,
            ny=ny,
            nz=nz,
            count=tt_flow_count,
            traffic_class=TrafficClass.TT,
            period_range=tt_period_range,
            frame_size_bits=tt_frame_size_bits,
            seed=seed,
            flow_prefix="tt",
        )

        be_flows = create_random_flows(
            nx=nx,
            ny=ny,
            nz=nz,
            count=be_flow_count,
            traffic_class=TrafficClass.BE,
            period_range=be_period_range,
            frame_size_bits=be_frame_size_bits,
            seed=1000 + seed,
            flow_prefix="be",
        )

        router = TorusNetworkSimulator(
            nx=nx,
            ny=ny,
            nz=nz,
            link_rate_bps=C,
            d_prop=d_prop,
            d_sw=d_sw,
            tt_buffer_bits=10**18,
            scenario=Scenario.SP_FP,
            l_np_bits=l_np_bits,
            enable_tt_regulation=False,
        )

        port_params = compute_port_traffic_parameters(
            nx=nx,
            ny=ny,
            nz=nz,
            flows=tt_flows,
            router=router,
        )

        b_fp_bits = analytical_buffer_fp_bits(port_params=port_params, l_np_bits=l_np_bits, C=C)
        b_fp_bytes = ceil_to_frame_bytes(b_fp_bits, tt_frame_size_bits)

        for factor, buffer_bytes in candidate_buffers_from_analytical(b_fp_bytes, tt_frame_size_bits):
            stats = run_single_configuration(
                scenario=Scenario.SP_FP,
                nx=nx,
                ny=ny,
                nz=nz,
                C=C,
                d_prop=d_prop,
                d_sw=d_sw,
                tt_buffer_bits=buffer_bytes * 8,
                tt_flows=tt_flows,
                be_flows=be_flows,
                n_gen_tt_target=n_gen_tt_target,
                drain_time=drain_time,
                l_np_bits=l_np_bits,
                enable_tt_regulation=enable_tt_regulation,
            )

            rows.append(make_row(seed_index, Scenario.SP_FP.value, f"L_np={l_np_bits / 8:.0f} bytes", factor, buffer_bytes, b_fp_bytes, True, stats))
            print_result(seed_index, "SP+FP", factor, buffer_bytes, b_fp_bytes, stats)

        for tas_window in tas_windows:
            b_tas_bits = analytical_buffer_tas_bits(port_params=port_params, tas_period=tas_period, tas_window=tas_window)
            b_tas_bytes = ceil_to_frame_bytes(b_tas_bits, tt_frame_size_bits)
            tas_ok = check_tas_stability(port_params=port_params, C=C, tas_period=tas_period, tas_window=tas_window)

            for factor, buffer_bytes in candidate_buffers_from_analytical(b_tas_bytes, tt_frame_size_bits):
                stats = run_single_configuration(
                    scenario=Scenario.TAS,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    C=C,
                    d_prop=d_prop,
                    d_sw=d_sw,
                    tt_buffer_bits=buffer_bytes * 8,
                    tt_flows=tt_flows,
                    be_flows=be_flows,
                    n_gen_tt_target=n_gen_tt_target,
                    drain_time=drain_time,
                    tas_period=tas_period,
                    tas_tt_window=tas_window,
                    enable_tt_regulation=enable_tt_regulation,
                )

                rows.append(make_row(seed_index, Scenario.TAS.value, f"W={tas_window * 1e6:.0f} us", factor, buffer_bytes, b_tas_bytes, tas_ok, stats))
                print_result(seed_index, f"TAS W={tas_window * 1e6:.0f}us", factor, buffer_bytes, b_tas_bytes, stats)

        for cqf_cycle in cqf_cycles:
            b_cqf_bytes = analytical_buffer_cqf_bytes(
                port_params=port_params,
                cqf_cycle=cqf_cycle,
                frame_size_bits=tt_frame_size_bits,
            )
            cqf_ok = check_cqf_feasibility(port_params=port_params, C=C, cqf_cycle=cqf_cycle)

            for factor, buffer_bytes in candidate_buffers_from_analytical(b_cqf_bytes, tt_frame_size_bits):
                stats = run_single_configuration(
                    scenario=Scenario.CQF,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    C=C,
                    d_prop=d_prop,
                    d_sw=d_sw,
                    tt_buffer_bits=buffer_bytes * 8,
                    tt_flows=tt_flows,
                    be_flows=be_flows,
                    n_gen_tt_target=n_gen_tt_target,
                    drain_time=drain_time,
                    cqf_cycle=cqf_cycle,
                    enable_tt_regulation=enable_tt_regulation,
                )

                rows.append(make_row(seed_index, Scenario.CQF.value, f"Tc={cqf_cycle * 1e6:.0f} us", factor, buffer_bytes, b_cqf_bytes, cqf_ok, stats))
                print_result(seed_index, f"CQF Tc={cqf_cycle * 1e6:.0f}us", factor, buffer_bytes, b_cqf_bytes, stats)

    save_results_csv(output_path, rows)
    print()
    print(f"Результаты сохранены в файл: {output_path.resolve()}")


if __name__ == "__main__":
    run_experiment()
