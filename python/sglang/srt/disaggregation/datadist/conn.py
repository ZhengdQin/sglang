import dataclasses
import json
import logging
import os
import queue
import re
import subprocess
import threading
from collections import defaultdict
from typing import Optional, Dict, List, Set

import llm_datadist
import numpy as np
import torch
import zmq
from llm_datadist import CacheDesc, BlocksCacheKey, Cache
from llm_datadist import LLMDataDist, LLMRole, LLMConfig
from numpy import typing as npt

from sglang.srt.disaggregation.base import KVArgs, KVPoll, BaseKVSender
from sglang.srt.disaggregation.common import CommonKVManager, CommonKVReceiver, CommonKVBootstrapServer
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_local_ip_by_remote

logger = logging.getLogger(__name__)

GUARD = "DataDistMsgGuard".encode("ascii")


@dataclasses.dataclass
class TransferKVChunk:
    room: int
    prefill_kv_indices: npt.NDArray[np.int32]
    index_slice: slice
    is_last: bool
    prefill_aux_index: Optional[int]


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int
    cluster_id: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]) -> "TransferInfo":
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            dst_kv_indices=np.frombuffer(msg[3], dtype=np.int32),
            dst_aux_index=int(msg[4].decode("ascii")),
            required_dst_info_num=int(msg[5].decode("ascii")),
            cluster_id=int(msg[6].decode("ascii"))
        )


def get_device_info(device_id):
    npu_info = subprocess.run(
        ["npu-smi info -m | awk '{if ($3 ~ /[0-9]+/) print $1, $2, $3}'"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    hccn_path = '/usr/local/Ascend/driver/tools/hccn_tool'
    if npu_info.returncode != 0 or not os.path.exists(hccn_path):
        raise RuntimeError("no npu-smi/hccn tools provided for NPU.")
    device_infos = [line.split(' ') for line in npu_info.stdout.splitlines()]
    info = device_infos[device_id]
    # info: npuId, chipId, deviceId, deviceIp, super_device_id, super_pod_id
    device_ip_info = subprocess.run(
        [hccn_path, '-i', f'{info[2]}', '-ip', '-g'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    re_result = re.match(r'ipaddr:(.*)\n', device_ip_info.stdout)
    if re_result is None:
        raise RuntimeError("Can't find npu ip")
    device_ip = re_result.group(1)
    info.append(device_ip)
    pod_info = subprocess.run(
        [f"npu-smi info -t spod-info -i {info[0]} -c {info[1]}"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    re_result = re.search(r'SDID *: (\d+)', pod_info.stdout)
    if re_result is None:
        raise RuntimeError("Can't find super device id")
    super_device_id = re_result.group(1)
    re_result = re.search(r'Super Pod ID *: (\d+)', pod_info.stdout)
    if re_result is None:
        raise RuntimeError("Can't find super pod id")
    super_pod_id = re_result.group(1)
    info.extend([super_device_id, super_pod_id])
    return info, len(device_infos)


def generate_rank_table_a3(device_id):
    device_info, device_count = get_device_info(device_id)
    rank_info = {
        "status": "completed",
        "version": "1.2",
        "server_count": "1",
        "server_list": [
            {
                "server_id": f"{get_local_ip_by_remote()}",
                "device": [
                    {
                        "device_id": f"{device_info[2]}",
                        "super_device_id": f"{device_info[4]}",
                        "device_ip": f"{device_info[3]}"
                    }
                ]
            }
        ],
        "super_pod_list": [
            {
                "super_pod_id": f"{device_info[5]}",
                "server_list": [
                    {"server_id": f"{get_local_ip_by_remote()}"}
                ]
            }
        ]
    }
    return json.dumps(rank_info), device_count


TORCH_DTYPE_TO_NPU_DTYPE = {
    torch.half: llm_datadist.DataType.DT_FLOAT16,
    torch.float16: llm_datadist.DataType.DT_FLOAT16,
    torch.bfloat16: llm_datadist.DataType.DT_BF16,
    torch.float: llm_datadist.DataType.DT_FLOAT,
    torch.float32: llm_datadist.DataType.DT_FLOAT,
    torch.int8: llm_datadist.DataType.DT_INT8,
    torch.int64: llm_datadist.DataType.DT_INT64,
    torch.int32: llm_datadist.DataType.DT_INT32
}


class DataDistKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend)
        self.registered_kv_caches: List[Cache] = []
        self.device_id = self.kv_args.gpu_id
        self.local_host_ip = get_local_ip_by_remote()
        # bootstrap_room到状态的映射
        self.request_status: Dict[int, int] = {}
        # 初始化datadist
        llm_config = LLMConfig()
        llm_config.device_id = self.device_id
        llm_config.sync_kv_timeout = 20000
        rank_table, self.world_size = generate_rank_table_a3(self.device_id)
        llm_config.local_comm_res = rank_table
        # 加上个node_rank偏移保证cluster_id不冲突
        self.cluster_id = self.device_id + self.world_size * ServerArgs.node_rank
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.role = LLMRole.PROMPT
            # p侧监听，D侧link_clusters
            llm_config.listen_ip_info = f"{self.local_host_ip}:{26000 + self.kv_args.gpu_id}"
        elif self.disaggregation_mode == DisaggregationMode.DECODE:
            self.role = LLMRole.DECODER
            self.cluster_id += self.world_size * ServerArgs.nnodes
        else:
            raise ValueError(
                f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
            )
        self.llm_datadist = LLMDataDist(self.role, self.cluster_id)
        engine_options = llm_config.generate_options()
        self.llm_datadist.init(engine_options)

        # 注册内存
        self.cache_manager = self.llm_datadist.cache_manager
        self.register_buffer_to_engine()

        self.server_socket = zmq.Context().socket(zmq.PULL)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.transfer_infos: Dict[int, Dict[int, TransferInfo]] = {}  # room到D侧传输信息的映射
            # P侧创建zmq监听，接受D侧的握手信息
            self.start_prefill_thread()
            # 启动多线程异步传输kv_cache，用队列保证请求处理的顺序性
            queue_size = 12
            self.transfer_queues = [queue.Queue() for _ in range(queue_size)]
            [threading.Thread(target=self.transfer_worker, args=[q], daemon=True).start() for q in self.transfer_queues]
        if self.disaggregation_mode == DisaggregationMode.DECODE:
            # 用来接受prefill发送的完成状态
            self.need_response_num: Dict[int, int] = {}
            self.response_tracker: Dict[int, Set[int]] = defaultdict(set)
            # D侧创建zmq监听接收P侧返回的传输完成消息
            self.start_decode_thread()
            self.register_link_lock = threading.Lock()
            self.link_clusters_dict = {}

    def register_buffer_to_engine(self):
        cache_desc = CacheDesc(
            num_tensors=len(self.kv_args.kv_data_ptrs),
            shape=tuple(self.kv_args.kv_data_shape),
            data_type=TORCH_DTYPE_TO_NPU_DTYPE[self.kv_args.kv_data_dtype]
        )
        cache_addrs = self.kv_args.kv_data_ptrs
        cache_key = BlocksCacheKey(self.cluster_id, 0)
        self.registered_kv_caches.append(
            self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
        )

        # metadata aux只注册output_ids
        output_ids_addr = self.kv_args.aux_data_ptrs[0]
        cache_desc = CacheDesc(
            num_tensors=1,
            shape=(self.kv_args.metadata_buffer_size, 16),
            data_type=TORCH_DTYPE_TO_NPU_DTYPE[torch.int32]
        )
        cache_addrs = [output_ids_addr]
        cache_key = BlocksCacheKey(self.cluster_id, 1)
        self.registered_kv_caches.append(
            self.cache_manager.register_blocks_cache(cache_desc, cache_addrs, cache_key)
        )

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: int):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def register_link(self, link_register_key, bootstrap_infos):
        # receiver触发建链，只用建一次
        with self.register_link_lock:
            if link_register_key in self.link_clusters_dict:
                return
            cluster_list = []
            for bootstrap_info in bootstrap_infos:
                cluster = llm_datadist.LLMClusterInfo()
                cluster.append_remote_ip_info(bootstrap_info["rank_ip"], 26000 + bootstrap_info["gpu_id"])
                cluster_list.append(cluster)
            ret, rets = self.llm_datadist.link_clusters(cluster_list, 20000)
            if ret != llm_datadist.LLMStatusCode.LLM_SUCCESS:
                raise Exception(f"link cluster failure {ret} {rets}")
            self.link_clusters_dict[link_register_key] = cluster_list

    def start_prefill_thread(self):
        self.server_socket.bind(f"tcp://{self.local_host_ip}:{self.rank_port}")

        def bootstrap_thread():
            while True:
                # receive and process handshake msg from KVReceiver bootstrap thread
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}, Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")

                required_dst_info_num = int(waiting_req_bytes[5].decode("ascii"))
                room = int(room)
                cluster_id = int(waiting_req_bytes[6].decode("ascii"))
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][cluster_id] = TransferInfo.from_zmq(waiting_req_bytes)
                # 多个D对应一个P场景，等待D全部握手，开始传输
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(f"{room=} is bootstrapped")
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()

    def sync_status_to_decode(
        self, remote: str, dst_port: int, room: int, status: int, prefill_rank: int
    ):
        if ":" in remote:
            remote = remote.split(":")[0]
        self._connect("tcp://" + remote + ":" + str(dst_port)).send_multipart(
            [
                str(room).encode("ascii"),
                str(status).encode("ascii"),
                str(prefill_rank).encode("ascii"),
            ]
        )

    def start_decode_thread(self):
        self.server_socket.bind(f"tcp://{self.local_host_ip}:{self.rank_port}")

        def decode_thread():
            while True:
                bootstrap_room, status, prefill_rank = self.server_socket.recv_multipart()
                status = int(status.decode("ascii"))
                bootstrap_room = int(bootstrap_room.decode("ascii"))
                prefill_rank = int(prefill_rank.decode("ascii"))

                if status == KVPoll.Success:
                    if bootstrap_room in self.request_status:
                        self.response_tracker[bootstrap_room].add(prefill_rank)
                        expected_response_num = self.need_response_num[bootstrap_room]
                        arrived_response_num = len(self.response_tracker[bootstrap_room])
                        if (
                            self.is_mla_backend
                            or arrived_response_num == expected_response_num
                        ):
                            self.update_status(bootstrap_room, KVPoll.Success)
                elif status == KVPoll.Failed:
                    self.update_status(bootstrap_room, KVPoll.Failed)

        threading.Thread(target=decode_thread).start()

    def transfer_worker(self, q: queue.Queue):
        while True:
            kv_chunk: TransferKVChunk = q.get()
            bootstrap_room = kv_chunk.room
            index_slice = kv_chunk.index_slice
            prefill_kv_indices = kv_chunk.prefill_kv_indices
            is_last = kv_chunk.is_last
            prefill_aux_index = kv_chunk.prefill_aux_index

            reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
            for req in reqs_to_be_processed:
                if req.is_dummy():
                    continue
                chunked_dst_kv_indices = req.dst_kv_indices[index_slice]
                decode_cache_key = BlocksCacheKey(req.cluster_id, 0)
                self.cache_manager.push_blocks(
                    decode_cache_key,
                    self.registered_kv_caches[0],
                    prefill_kv_indices.tolist(),
                    chunked_dst_kv_indices.tolist(),
                    range(len(self.kv_args.kv_data_ptrs)),
                    range(len(self.kv_args.kv_data_ptrs))
                )

                # Only the last chunk we need to send the aux data.
                if is_last:
                    decode_cache_key = BlocksCacheKey(req.cluster_id, 1)
                    self.cache_manager.push_blocks(
                        decode_cache_key,
                        self.registered_kv_caches[1],
                        [prefill_aux_index],
                        [req.dst_aux_index],
                        range(1),
                        range(1)
                    )
            if is_last:
                # 全部发送完成，同步状态到decoder
                self.update_status(bootstrap_room, KVPoll.Success)
                for req in reqs_to_be_processed:
                    self.sync_status_to_decode(
                        req.endpoint,
                        req.dst_port,
                        req.room,
                        KVPoll.Success,
                        self.kv_args.engine_rank
                    )
                del self.transfer_infos[bootstrap_room]

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        aux_index: Optional[int] = None
    ):
        assert bootstrap_room in self.transfer_infos
        queue_index = bootstrap_room % len(self.transfer_queues)
        self.transfer_queues[queue_index].put(
            TransferKVChunk(
                room=bootstrap_room,
                prefill_kv_indices=kv_indices,
                index_slice=index_slice,
                is_last=is_last,
                prefill_aux_index=aux_index,
            )
        )


class DataDistKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: DataDistKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        data_parallel_rank: Optional[int] = None,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, data_parallel_rank)
        # 触发llm-datadist建链
        link_register_key = f"{self.bootstrap_addr}_{self.target_dp_group}"
        mgr.register_link(link_register_key, self.bootstrap_infos)
        mgr.update_status(self.bootstrap_room, KVPoll.WaitingForInput)

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        kv_mgr: DataDistKVManager = self.kv_mgr
        kv_mgr.need_response_num[self.bootstrap_room] = len(self.bootstrap_infos)  # 记录D侧需要P侧响应的个数
        for bootstrap_info in self.bootstrap_infos:
            prefill_server_url = f"{bootstrap_info['rank_ip']}:{bootstrap_info['rank_port']}"
            is_dummy = bootstrap_info["is_dummy"]
            sock, lock = self._connect("tcp://" + prefill_server_url)

            with lock:
                sock.send_multipart([
                    GUARD,
                    str(self.bootstrap_room).encode("ascii"),
                    kv_mgr.local_host_ip.encode("ascii"),
                    str(kv_mgr.rank_port).encode("ascii"),
                    kv_indices.tobytes() if not is_dummy else b"",
                    str(aux_index).encode("ascii"),
                    str(self.required_dst_info_num).encode("ascii"),
                    str(kv_mgr.cluster_id).encode("ascii"),
                ])

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        kv_mgr: DataDistKVManager = self.kv_mgr
        if self.bootstrap_room in kv_mgr.request_status:
            kv_mgr.request_status.pop(self.bootstrap_room)
        if self.bootstrap_room in kv_mgr.need_response_num:
            kv_mgr.need_response_num.pop(self.bootstrap_room)
        if self.bootstrap_room in kv_mgr.response_tracker:
            kv_mgr.response_tracker.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()
        raise Exception("KVReceiver failure")


class DataDistKVSender(BaseKVSender):
    def __init__(
        self,
        mgr: DataDistKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int
    ):
        self.kv_mgr = mgr
        self.bootstrap_room = bootstrap_room
        self.aux_index = None
        self.num_kv_indices = None
        self.bootstrap_server_url = bootstrap_addr
        self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Bootstrapping)
        self.curr_idx = 0

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None):
        self.num_kv_indices = num_kv_indices
        self.aux_index = aux_index

    def send(self, kv_indices: npt.NDArray[np.int32]):
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices
        self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.aux_index
        )

    def poll(self) -> KVPoll:
        return self.kv_mgr.check_status(self.bootstrap_room)

    def clear(self) -> None:
        if self.bootstrap_room in self.kv_mgr.request_status:
            self.kv_mgr.request_status.pop(self.bootstrap_room)

    def failure_exception(self):
        self.clear()
        raise Exception("KVSender failure")


class DataDistKVBootstrapServer(CommonKVBootstrapServer):
    pass
