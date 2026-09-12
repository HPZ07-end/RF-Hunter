"""B题模拟器客户端（附件1、附件2协议；Python 3.9+，仅标准库）。

先在模拟器界面登录并启动测试，接口就绪后：
    client = SimulatorClient(robot_id="你的参赛队号")
    info = client.enter()
    result = client.measure(300, 400, 1)
    if result["measure_result"] == "near":
        client.clear(300, 400, 1)
    # 在问题三/四策略确认任务完成后：
    client.exit()

每局新建一个实例；同一局只用一个实例。import 或直接运行此文件不发送请求。
响应保留附件原始字段，异常不会伪装成 no_signal 或清除失败。
日志通过标准 logging 输出；主程序可配置 FileHandler 持久保存。
"""
import json
import logging
import math
import threading
import time
import unicodedata
import uuid
from http.client import HTTPException
from numbers import Integral, Real
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler, HTTPRedirectHandler

__all__ = ["SimulatorClient", "ClientError", "TransportError", "ProtocolError",
           "HTTPStatusError", "ActionRejected", "PendingActionError"]
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ClientError(RuntimeError):
    """所有客户端运行异常的基类。"""


class TransportError(ClientError):
    """未获得完整响应，动作是否执行未知；只能 retry_pending() 重试原动作。"""


class ProtocolError(ClientError):
    """响应不符合协议，保留待确认动作。"""


class PendingActionError(ClientError):
    """上一动作尚未确认，禁止发送不同的新动作。"""


class HTTPStatusError(ClientError):
    def __init__(self, status, response, request_id):
        self.status = status
        self.response = response
        self.request_id = request_id
        super().__init__(f"HTTP {status}, request_id={request_id}, response={response}")


class ActionRejected(ClientError):
    def __init__(self, response, request_id):
        self.response = response
        self.request_id = request_id
        super().__init__(f"模拟器拒绝动作（accepted=false），request_id={request_id}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _identifier(value, name, maximum):
    if not isinstance(value, str) or not 1 <= len(value.encode("utf-8")) <= maximum:
        raise ValueError(f"{name} 必须是 UTF-8 长度 1..{maximum} 字节的字符串")
    if any(unicodedata.category(c) in ("Cc", "Cf") for c in value):
        raise ValueError(f"{name} 不能包含控制字符或不可见格式字符")
    return value


def _coordinate(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("坐标必须是实数，不能是布尔值")
    value = float(value)
    if not math.isfinite(value) or abs(value) > 2_000_000:
        raise ValueError("坐标必须有限，且绝对值不超过 2000000 米")
    return value


def _channel(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("频道必须是 1..20 的整数")
    if not math.isfinite(float(value)) or not 1 <= value <= 20 or int(value) != value:
        raise ValueError("频道必须是 1..20 的整数")
    return int(value)


def _number(data, key):
    v = data.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ProtocolError(f"响应字段 {key} 应为有限非负数")
    return v


class SimulatorClient:
    """通信层；不包含搜索、交会定位或“全部清除”的判定。

    timeout: 单次网络等待秒数。retries: 网络异常后额外重试次数。
    重试复用完全相同的字节和 request_id，不对 HTTP 错误自动重试。
    remaining_real_time_s 是依据 enter 请求首次发送时刻计算的保守预算，
    并非模拟器实时查询值。virtual_time_s 以成功响应为准。
    """
    def __init__(self, robot_id, base_url="http://127.0.0.1:2026", *,
                 timeout=5.0, retries=2, retry_delay=0.25):
        self.robot_id = _identifier(robot_id, "robot_id", 64)
        url = urlsplit(base_url)
        if (url.scheme != "http" or not url.hostname or url.path not in ("", "/")
                or url.query or url.fragment or url.username or url.password):
            raise ValueError("base_url 应为 http://主机:端口，不含路径、查询参数或认证信息")
        if not isinstance(retries, Integral) or isinstance(retries, bool) or retries < 0:
            raise ValueError("retries 必须是非负整数")
        if not math.isfinite(timeout) or timeout <= 0 or not math.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError("timeout 必须为有限正数，retry_delay 必须为有限非负数")
        self.base_url = base_url.rstrip("/")
        self.timeout, self.retries, self.retry_delay = timeout, int(retries), retry_delay
        # 本机通信不经过系统代理，也不接受重定向。
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())
        self._lock = threading.Lock()
        self._pending = None
        self._state = "new"
        self.position = None
        self.current_channel = None
        self.virtual_time_s = None
        self.max_virtual_duration_s = None
        self.max_real_duration_s = None
        self._deadline = None
        self.cleared_channels = set()
        self.action_counts = {"enter": 0, "measure": 0, "clear": 0, "exit": 0}

    @property
    def remaining_real_time_s(self):
        """本地保守剩余现实秒数；enter 前为 None。不会发送 HTTP。"""
        return None if self._deadline is None else max(0.0, self._deadline - time.monotonic())

    @property
    def pending_request_id(self):
        return None if self._pending is None else self._pending[1]["request_id"]

    def enter(self):
        """POST /enter：开始本局，返回限时等原始字段。"""
        return self._new_action("/enter")

    def measure(self, x, y, channel):
        """POST /measure：移动、必要时切换频道、检测；返回三种 measure_result。"""
        return self._new_action("/measure", x, y, channel)

    def clear(self, x, y, channel):
        """POST /clear：移动并尝试清除；不改变 current_channel。"""
        return self._new_action("/clear", x, y, channel)

    def exit(self):
        """POST /exit：主动结束本局，不用于查询超时原因。"""
        return self._new_action("/exit")

    def retry_pending(self):
        """结果未知时重发原动作；原字节/ID均保持不变，不重复累加本地统计。"""
        with self._lock:
            if self._pending is None:
                raise PendingActionError("没有需要重试的动作")
            return self._send_pending()

    def _new_action(self, path, x=None, y=None, channel=None):
        # 锁覆盖发送、读取及更新，防止同一实例并发发送不同动作。
        with self._lock:
            if self._pending is not None:
                raise PendingActionError(f"请先 retry_pending()：{self.pending_request_id}")
            if (path == "/enter" and self._state != "new") or (path != "/enter" and self._state != "active"):
                raise ClientError(f"当前状态 {self._state} 不能执行 {path}；每局须新建实例并 enter")
            payload = {"arena_id": "default", "robot_id": self.robot_id,
                       "request_id": uuid.uuid4().hex}
            if path in ("/measure", "/clear"):
                payload.update(position={"x": _coordinate(x), "y": _coordinate(y)},
                               channel=_channel(channel))
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8")
            self._pending = (path, payload, body, time.monotonic())
            return self._send_pending()

    def _send_pending(self):
        path, payload, body, first_sent = self._pending
        for attempt in range(self.retries + 1):
            request = Request(self.base_url + path, data=body, method="POST",
                              headers={"Content-Type": "application/json"})
            logger.info("request %s", json.dumps({"path": path, "payload": payload,
                        "attempt": attempt + 1}, ensure_ascii=False))
            try:
                try:
                    with self._opener.open(request, timeout=self.timeout) as response:
                        status, raw = response.status, response.read()
                except HTTPError as error:
                    with error:
                        status, raw = error.code, error.read()
            except (URLError, OSError, HTTPException) as error:
                logger.warning("transport_error request_id=%s error=%r", payload["request_id"], error)
                if attempt < self.retries:
                    time.sleep(self.retry_delay)
                    continue
                raise TransportError(
                    f"{path} 未收到完整响应，动作执行状态未知；request_id={payload['request_id']}。"
                    "确认模拟器状态后可 retry_pending()；不要发送不同的新动作。"
                ) from error
            break
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as error:
            raise ProtocolError(f"HTTP {status} 响应不是有效 UTF-8 JSON；动作保留待确认") from error
        logger.info("response %s", json.dumps({"request_id": payload["request_id"],
                    "http_status": status, "body": data}, ensure_ascii=False))
        if not isinstance(data, dict) or type(data.get("accepted")) is not bool:
            raise ProtocolError("响应必须是包含布尔 accepted 的对象")
        _number(data, "real_timestamp_ms")
        _number(data, "virtual_time_s")
        if status != 200:
            # 400等确定的拒绝可开始新动作；409/429/5xx保留原动作以便核实。
            if status in (400, 404, 405, 413, 415) and data["accepted"] is False:
                self._pending = None
            raise HTTPStatusError(status, data, payload["request_id"])
        if data["accepted"] is False:
            self._pending = None
            # 附件规定此时返回的 virtual_time_s=0 不是当前时间。
            raise ActionRejected(data, payload["request_id"])
        self._validate_success(path, data)
        self._apply_success(path, payload, data, first_sent)
        self._pending = None
        return data

    @staticmethod
    def _validate_success(path, data):
        if path == "/enter":
            for key in ("max_virtual_duration_s", "max_real_duration_s", "remaining_real_duration_s"):
                _number(data, key)
            remaining = data["remaining_real_duration_s"]
            if not 0 <= remaining <= 1200 or int(remaining) != remaining:
                raise ProtocolError("remaining_real_duration_s 不在整数范围 0..1200")
        elif path == "/measure":
            if data.get("measure_result") not in ("no_signal", "near", "direction"):
                raise ProtocolError("未知 measure_result")
            if data["measure_result"] == "direction":
                if not 0 <= _number(data, "svd_deg") < 360:
                    raise ProtocolError("svd_deg 不在 [0,360) 范围内")
        elif path == "/clear":
            if data.get("clear_result") not in ("success", "no_target_in_range"):
                raise ProtocolError("未知 clear_result")
        elif data.get("exit_reason") != "user_exit":
            raise ProtocolError("未知 exit_reason")

    def _apply_success(self, path, payload, data, first_sent):
        self.virtual_time_s = float(data["virtual_time_s"])
        self.action_counts[path[1:]] += 1
        if path == "/enter":
            self._state = "active"
            self.position, self.current_channel = (0.0, 0.0), 1
            self.max_virtual_duration_s = data["max_virtual_duration_s"]
            self.max_real_duration_s = data["max_real_duration_s"]
            # 首次发送早于服务端响应，故这个预算略保守，也适用于enter响应丢失后的重试。
            self._deadline = first_sent + data["remaining_real_duration_s"]
        elif path in ("/measure", "/clear"):
            self.position = (payload["position"]["x"], payload["position"]["y"])
            if path == "/measure":
                self.current_channel = payload["channel"]
            elif data["clear_result"] == "success":
                self.cleared_channels.add(payload["channel"])
        else:
            self._state = "exited"
