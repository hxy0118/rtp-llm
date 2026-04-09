import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed import collective_torch as ct
from rtp_llm.models_py.distributed import rocm_rccl as hr


class TestCollectiveTorchHipGraphUnit(unittest.TestCase):
    def setUp(self):
        self._orig_is_rocm_runtime = hr._is_rocm_runtime
        self._orig_rccl_comm = hr._rccl_comm
        self._orig_rccl_world_size = hr._rccl_world_size
        self._orig_rccl_lib = hr._rccl_lib
        self._orig_cache = dict(hr._hipgraph_allgather_outputs)
        hr._hipgraph_allgather_outputs.clear()

    def tearDown(self):
        hr._is_rocm_runtime = self._orig_is_rocm_runtime
        hr._rccl_comm = self._orig_rccl_comm
        hr._rccl_world_size = self._orig_rccl_world_size
        hr._rccl_lib = self._orig_rccl_lib
        hr._hipgraph_allgather_outputs.clear()
        hr._hipgraph_allgather_outputs.update(self._orig_cache)

    def test_should_use_hipgraph_capture_rccl(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertTrue(hr.should_use_hipgraph_capture_rccl(True))
            self.assertFalse(hr.should_use_hipgraph_capture_rccl(False))

    def test_should_not_use_hipgraph_capture_rccl_without_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            self.assertFalse(hr.should_use_hipgraph_capture_rccl(True))

    def test_get_nccl_dtype_map_and_error(self):
        fp16_tensor = torch.zeros(1, dtype=torch.float16)
        self.assertEqual(
            hr._get_nccl_dtype(fp16_tensor), hr._NCCL_DTYPE_MAP[torch.float16]
        )

        with self.assertRaises(TypeError):
            hr._get_nccl_dtype(torch.zeros(1, dtype=torch.bool))

    def test_get_or_create_allgather_output_cache_reuse(self):
        hr._rccl_world_size = 3
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            out1 = hr._get_or_create_allgather_output(src)
            out2 = hr._get_or_create_allgather_output(src)

        self.assertIs(out1, out2)
        self.assertEqual(tuple(out1.shape), (6, 4))
        self.assertEqual(out1.dtype, src.dtype)
        self.assertEqual(out1.device, src.device)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 1)

    def test_get_or_create_allgather_output_rejects_inactive_capture(self):
        hr._rccl_world_size = 2
        src = torch.zeros((2, 4), dtype=torch.float16)

        with patch.object(hr, "_is_hipgraph_capture_active", return_value=False):
            with self.assertRaises(RuntimeError):
                hr._get_or_create_allgather_output(src)

        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_set_graph_capture_nccl_comm_clears_cache(self):
        hr._is_rocm_runtime = True
        hr._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )
        fake_lib = object()

        with patch.object(hr, "_load_rccl", return_value=fake_lib), patch.object(
            hr, "_setup_rccl_api"
        ) as setup_api:
            hr.set_graph_capture_nccl_comm(456, 4, 0)

        setup_api.assert_called_once_with(fake_lib)
        self.assertIsInstance(hr._rccl_comm, ctypes.c_void_p)
        self.assertEqual(hr._rccl_comm.value, 456)
        self.assertEqual(hr._rccl_world_size, 4)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_enter_mode_without_handle_keeps_existing_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 4

        hr.enter_graph_capture_mode(0, 0, 0)

        self.assertIsNotNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_comm.value, 123)
        self.assertEqual(hr._rccl_world_size, 4)

    def test_set_graph_capture_nccl_comm_zero_handle_clears_state(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = ctypes.c_void_p(123)
        hr._rccl_world_size = 4
        hr._hipgraph_allgather_outputs[(tuple([2, 4]), torch.float16, "cpu", -1)] = (
            torch.zeros((2, 4), dtype=torch.float16)
        )

        hr.set_graph_capture_nccl_comm(0, 0, 0)

        self.assertIsNone(hr._rccl_comm)
        self.assertEqual(hr._rccl_world_size, 1)
        self.assertEqual(len(hr._hipgraph_allgather_outputs), 0)

    def test_prepare_hipgraph_capture_rccl_comm_bootstraps_on_tp(self):
        hr._is_rocm_runtime = True
        parallelism_config = SimpleNamespace(tp_size=2, world_size=2)
        tp_group = object()
        with patch.object(
            hr, "bootstrap_hipgraph_capture_rccl_comm_from_tp_group"
        ) as bootstrap:
            hr.prepare_hipgraph_capture_rccl_comm_if_needed(
                parallelism_config, tp_group
            )
        bootstrap.assert_called_once_with(tp_group)

    def test_all_gather_fail_fast_when_capture_has_no_rccl_comm(self):
        hr._is_rocm_runtime = True
        hr._rccl_comm = None
        tensor = torch.zeros((1, 2), dtype=torch.float16)
        with patch.object(hr, "_is_hipgraph_capture_active", return_value=True):
            with self.assertRaises(RuntimeError):
                ct.all_gather(tensor, ct.Group.TP)


if __name__ == "__main__":
    unittest.main()
