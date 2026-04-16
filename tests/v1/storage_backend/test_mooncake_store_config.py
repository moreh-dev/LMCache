# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MooncakeStoreConfig.enable_ssd_offload field."""
import json
import tempfile
from pathlib import Path

import pytest

from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakeStoreConfig,
)


@pytest.fixture()
def base_config_dict():
    """Minimal valid config dict for MooncakeStoreConfig."""
    return {
        "local_hostname": "localhost",
        "metadata_server": "P2PHANDSHAKE",
        "global_segment_size": 3355443200,
        "local_buffer_size": 1073741824,
        "protocol": "tcp",
        "device_name": "",
        "master_server_address": "localhost:50051",
        "transfer_timeout": 1,
        "storage_root_dir": "",
    }


class TestMooncakeStoreConfigFromFile:
    def test_enable_ssd_offload_true(self, base_config_dict, tmp_path):
        base_config_dict["enable_ssd_offload"] = True
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(base_config_dict))

        config = MooncakeStoreConfig.from_file(str(config_file))
        assert config.enable_ssd_offload is True

    def test_enable_ssd_offload_false(self, base_config_dict, tmp_path):
        base_config_dict["enable_ssd_offload"] = False
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(base_config_dict))

        config = MooncakeStoreConfig.from_file(str(config_file))
        assert config.enable_ssd_offload is False

    def test_enable_ssd_offload_default(self, base_config_dict, tmp_path):
        # Key not present → defaults to False
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(base_config_dict))

        config = MooncakeStoreConfig.from_file(str(config_file))
        assert config.enable_ssd_offload is False


class TestMooncakeStoreConfigFromLMCacheConfig:
    def test_enable_ssd_offload_from_extra_config(self, base_config_dict):
        base_config_dict["enable_ssd_offload"] = True

        class FakeConfig:
            extra_config = base_config_dict

        config = MooncakeStoreConfig.load_from_lmcache_config(FakeConfig())
        assert config.enable_ssd_offload is True

    def test_enable_ssd_offload_default_from_extra_config(self, base_config_dict):
        # Key not present → defaults to False
        class FakeConfig:
            extra_config = base_config_dict

        config = MooncakeStoreConfig.load_from_lmcache_config(FakeConfig())
        assert config.enable_ssd_offload is False
