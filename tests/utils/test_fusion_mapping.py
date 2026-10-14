# Copyright 2026 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import load_file

import transformers.conversion_mapping as conversion_mapping
import transformers.fusion_mapping as fusion_mapping
import transformers.monkey_patching as monkey_patching
from transformers import PretrainedConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    get_model_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import Conv3dToLinear, WeightConverter, WeightRenaming
from transformers.fusion_mapping import register_fusion_patches
from transformers.modeling_utils import PreTrainedModel
from transformers.monkey_patching import (
    apply_patches,
    get_patch_mapping,
    register_patch_mapping,
    unregister_patch_mapping,
)
from transformers.registration_scope import load_registration_scope


DUMMY_TRANSFORMERS_MODULE_NAME = "transformers.test_fusion_mapping_dummy"
# `apply_patches()` scans `sys.modules` and only rewrites class attributes exposed
# from `transformers.*` modules, so this dummy class must be reachable through a
# fake `transformers` module instead of only through a local symbol.
DUMMY_TRANSFORMERS_MODULE = types.ModuleType(DUMMY_TRANSFORMERS_MODULE_NAME)
sys.modules[DUMMY_TRANSFORMERS_MODULE_NAME] = DUMMY_TRANSFORMERS_MODULE


class DummyVisionConfig(PretrainedConfig):
    model_type = "dummy_fusion_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        patch_embed_stride=(2, 2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.patch_embed_stride = patch_embed_stride


class DummyFusionConfig(PretrainedConfig):
    model_type = "dummy_fusion"
    sub_configs = {"vision_config": DummyVisionConfig}

    def __init__(self, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = DummyVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = DummyVisionConfig(**vision_config)

        self.vision_config = vision_config


class DummyPatchEmbedding(nn.Module):
    def __init__(self, stride=(2, 2, 2), bias=False):
        super().__init__()
        self.embed_dim = 8
        self.proj = nn.Conv3d(3, self.embed_dim, kernel_size=(2, 2, 2), stride=stride, bias=bias)


DUMMY_PATCHABLE_CLASSES = {"DummyPatchEmbedding": DummyPatchEmbedding}

for class_name, patchable_class in DUMMY_PATCHABLE_CLASSES.items():
    setattr(DUMMY_TRANSFORMERS_MODULE, class_name, patchable_class)


class DummyFusionModel(PreTrainedModel):
    config_class = DummyFusionConfig

    def __init__(self, config):
        super().__init__(config)
        # Resolve the class through the fake `transformers.*` module so monkey patching
        # can replace it before instantiation.
        self.patch_embed = DUMMY_TRANSFORMERS_MODULE.DummyPatchEmbedding(
            stride=config.vision_config.patch_embed_stride, bias=True
        )
        self.post_init()


class FusionMappingTest(unittest.TestCase):
    """Covers registration, no-match, and conflict handling for fusion mapping."""

    fusion_config = {"patch_embeddings": True}

    def setUp(self):
        self.patch_mapping_patcher = patch.object(monkey_patching, "_monkey_patch_mapping_cache", {})
        self.patch_mapping_patcher.start()
        self.discovery_cache_patcher = patch.object(fusion_mapping, "_FUSION_DISCOVERY_CACHE", {})
        self.discovery_cache_patcher.start()
        self.checkpoint_conversion_mapping_cache = deepcopy(conversion_mapping._checkpoint_conversion_mapping_cache)
        self.user_registered_mappings = set(conversion_mapping.USER_REGISTERED_MAPPINGS)

    def tearDown(self):
        self.patch_mapping_patcher.stop()
        self.discovery_cache_patcher.stop()
        conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(self.checkpoint_conversion_mapping_cache)
        conversion_mapping.USER_REGISTERED_MAPPINGS.clear()
        conversion_mapping.USER_REGISTERED_MAPPINGS.update(self.user_registered_mappings)

    def _projection(self, model):
        patch_embed = model.patch_embed
        if hasattr(patch_embed, "linear_proj"):
            return patch_embed.linear_proj
        return patch_embed.proj

    def _source_patterns(self, conversions):
        return {pattern for conversion in conversions for pattern in conversion.source_patterns}

    def test_register_fusion_patches_is_effective_on_dummy_model(self):
        # Registers and applies a fusion on a dummy model.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig()

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(config.model_type))
        self.assertIsInstance(DummyFusionModel(config).patch_embed.proj, nn.Conv3d)

        register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

        self.assertEqual(len(get_patch_mapping()), 1)
        self.assertEqual(len(get_checkpoint_conversion_mapping(config.model_type)), 2)

        with apply_patches():
            fused_model = DummyFusionModel(config)

        fused_projection = getattr(
            fused_model.patch_embed, "linear_proj", getattr(fused_model.patch_embed, "proj", None)
        )
        self.assertIsInstance(fused_projection, nn.Linear)

    def test_register_fusion_patches_skips_when_no_modules_match(self):
        # Leaves registries untouched when nothing is fusable.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig(vision_config={"patch_embed_stride": (1, 1, 1)})

        register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(config.model_type))

    def test_register_fusion_patches_raises_on_transform_conflicts(self):
        # Rejects transforms that would shadow an existing source pattern.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"
        config = DummyFusionConfig()
        model_type = config.model_type

        # build a conflicting conversion mapping with the same source pattern but different target pattern
        register_checkpoint_conversion_mapping(
            model_type,
            [
                WeightConverter(
                    source_patterns=r"patch_embed\.proj\.weight$",
                    target_patterns=r"patch_embed\.other_linear_proj\.weight$",
                    operations=[Conv3dToLinear(in_channels=3, kernel_size=(2, 2, 2))],
                )
            ],
            overwrite=True,
        )

        with self.assertRaisesRegex(ValueError, "conflicts with an existing conversion mapping"):
            register_fusion_patches(DummyFusionModel, config, fusion_config=self.fusion_config)

    def test_from_pretrained_uses_serialized_fusion_config(self):
        # A serialized `fusion_config` is reused on a later load.
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"

        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as fused_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)

            fused_model = DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            fused_model.save_pretrained(fused_dir)

            # Simulate a fresh process so the second load comes only from the serialized config.
            monkey_patching._monkey_patch_mapping_cache.clear()
            fusion_mapping._FUSION_DISCOVERY_CACHE.clear()
            conversion_mapping._checkpoint_conversion_mapping_cache = deepcopy(
                self.checkpoint_conversion_mapping_cache
            )

            reloaded_model = DummyFusionModel.from_pretrained(fused_dir)

        fused_projection = getattr(
            reloaded_model.patch_embed, "linear_proj", getattr(reloaded_model.patch_embed, "proj", None)
        )
        self.assertIsInstance(fused_projection, nn.Linear)

    def test_registrations_outside_a_load_stay_global(self):
        # Entries written with no load in progress survive a later load, and a load cannot replace them.
        class GlobalReplacement(nn.Module):
            pass

        class ScopedReplacement(nn.Linear):
            pass

        register_patch_mapping({"KeptGlobal": GlobalReplacement})
        register_checkpoint_conversion_mapping("kept_global", [WeightRenaming("src", "tgt")])

        with load_registration_scope():
            register_patch_mapping({"ScopedOnly": ScopedReplacement})
            register_checkpoint_conversion_mapping("scoped_only", [WeightRenaming("fused.src", "fused.tgt")])
            register_patch_mapping({"KeptGlobal": ScopedReplacement}, overwrite=True)

            mapping = get_patch_mapping()
            self.assertIs(mapping["KeptGlobal"], ScopedReplacement)
            self.assertIs(mapping["ScopedOnly"], ScopedReplacement)
            self.assertEqual(len(get_checkpoint_conversion_mapping("kept_global")), 1)
            self.assertEqual(len(get_checkpoint_conversion_mapping("scoped_only")), 1)
            self.assertNotIn("scoped_only", conversion_mapping.USER_REGISTERED_MAPPINGS)
            self.assertIn("kept_global", conversion_mapping.USER_REGISTERED_MAPPINGS)

            # Dropping the scoped shadow reveals the global class again.
            unregister_patch_mapping(["KeptGlobal"])
            self.assertIs(get_patch_mapping()["KeptGlobal"], GlobalReplacement)
            self.assertIs(get_patch_mapping()["ScopedOnly"], ScopedReplacement)

        self.assertEqual(get_patch_mapping(), {"KeptGlobal": GlobalReplacement})
        self.assertIsNone(get_checkpoint_conversion_mapping("scoped_only"))
        self.assertEqual(len(get_checkpoint_conversion_mapping("kept_global")), 1)
        self.assertNotIn("scoped_only", conversion_mapping.USER_REGISTERED_MAPPINGS)

    def test_nested_load_scopes_do_not_see_each_other(self):
        class Replacement(nn.Module):
            pass

        with load_registration_scope():
            register_patch_mapping({"OuterPatch": Replacement})
            register_checkpoint_conversion_mapping("outer_type", [WeightRenaming("outer", "outer_tgt")])
            with load_registration_scope():
                register_patch_mapping({"InnerPatch": Replacement})
                self.assertIn("InnerPatch", get_patch_mapping())
                self.assertNotIn("OuterPatch", get_patch_mapping())
                self.assertIsNone(get_checkpoint_conversion_mapping("outer_type"))
            self.assertIn("OuterPatch", get_patch_mapping())
            self.assertNotIn("InnerPatch", get_patch_mapping())
            self.assertIsNotNone(get_checkpoint_conversion_mapping("outer_type"))

        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping("outer_type"))

    def test_fused_then_plain_loads_do_not_share_registrations(self):
        # Two fused loads and one plain load in the same process.
        model_type = f"dummy_fusion_{self._testMethodName}"
        DummyFusionConfig.model_type = model_type
        register_checkpoint_conversion_mapping(
            model_type,
            [WeightRenaming("global.keep", "global.kept")],
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as source_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)
            original = load_file(os.path.join(source_dir, "model.safetensors"))

            first, first_info = DummyFusionModel.from_pretrained(
                source_dir, fusion_config=self.fusion_config, output_loading_info=True
            )
            second, second_info = DummyFusionModel.from_pretrained(
                source_dir, fusion_config=self.fusion_config, output_loading_info=True
            )
            plain, plain_info = DummyFusionModel.from_pretrained(source_dir, output_loading_info=True)

        for info in (first_info, second_info, plain_info):
            self.assertEqual(info["missing_keys"], set())
            self.assertEqual(info["unexpected_keys"], set())
            self.assertEqual(info["mismatched_keys"], set())

        self.assertIsInstance(self._projection(first), nn.Linear)
        self.assertIsInstance(self._projection(second), nn.Linear)
        self.assertIsInstance(self._projection(plain), nn.Conv3d)
        torch.testing.assert_close(
            first.patch_embed.linear_proj.weight.cpu(), original["patch_embed.proj.weight"].reshape(8, 24)
        )
        torch.testing.assert_close(second.patch_embed.linear_proj.bias.cpu(), original["patch_embed.proj.bias"])

        # The load did not publish its fusion rules into the process-wide registries.
        self.assertEqual(get_patch_mapping(), {})
        global_sources = self._source_patterns(get_checkpoint_conversion_mapping(model_type))
        self.assertEqual(global_sources, {"global.keep"})

        fused_sources = self._source_patterns(get_model_conversion_mapping(first, add_legacy=False))
        self.assertIn(r"patch_embed\.proj\.weight$", fused_sources)
        self.assertIn(r"patch_embed\.proj\.bias$", fused_sources)
        self.assertIn("global.keep", fused_sources)

        plain_sources = self._source_patterns(get_model_conversion_mapping(plain, add_legacy=False))
        self.assertNotIn(r"patch_embed\.proj\.weight$", plain_sources)
        self.assertEqual(plain_sources, {"global.keep"})

    def test_failed_fused_load_leaves_no_registrations(self):
        model_type = f"dummy_fusion_{self._testMethodName}"
        DummyFusionConfig.model_type = model_type
        seen = {}
        calls = {"n": 0}
        real_init = DummyFusionModel.__init__

        def exploding_init(self, config):
            calls["n"] += 1
            # Discovery meta-inits the class once. The real load is the next construction.
            if calls["n"] > 1:
                seen["patch"] = "DummyPatchEmbedding" in get_patch_mapping()
                seen["conversions"] = get_checkpoint_conversion_mapping(model_type)
                raise RuntimeError("load failed after fusion registration")
            return real_init(self, config)

        with tempfile.TemporaryDirectory() as source_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)
            with patch.object(DummyFusionModel, "__init__", exploding_init):
                with self.assertRaisesRegex(RuntimeError, "load failed after fusion registration"):
                    DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)

            self.assertGreaterEqual(calls["n"], 2)
            self.assertTrue(seen["patch"])
            self.assertIsNotNone(seen["conversions"])
            self.assertEqual(get_patch_mapping(), {})
            self.assertIsNone(get_checkpoint_conversion_mapping(model_type))
            self.assertNotIn(model_type, conversion_mapping.USER_REGISTERED_MAPPINGS)

            # A failed load must not make the next fused or plain load disagree with its own config.
            fused = DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            plain = DummyFusionModel.from_pretrained(source_dir)

        self.assertIsInstance(self._projection(fused), nn.Linear)
        self.assertIsInstance(self._projection(plain), nn.Conv3d)
        self.assertEqual(get_patch_mapping(), {})
        self.assertIsNone(get_checkpoint_conversion_mapping(model_type))

    def test_conflicting_fusion_load_drops_the_partial_patch(self):
        # The patch is registered before the converter conflict is raised. A failed load drops both.
        model_type = f"dummy_fusion_{self._testMethodName}"
        DummyFusionConfig.model_type = model_type
        register_checkpoint_conversion_mapping(
            model_type,
            [
                WeightConverter(
                    source_patterns=r"patch_embed\.proj\.weight$",
                    target_patterns=r"patch_embed\.other_linear_proj\.weight$",
                    operations=[Conv3dToLinear(in_channels=3, kernel_size=(2, 2, 2))],
                )
            ],
            overwrite=True,
        )

        with tempfile.TemporaryDirectory() as source_dir:
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)
            with self.assertRaisesRegex(ValueError, "conflicts with an existing conversion mapping"):
                DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)

        self.assertEqual(get_patch_mapping(), {})
        remaining = get_checkpoint_conversion_mapping(model_type)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].target_patterns, ["patch_embed.other_linear_proj.weight"])

    def test_save_pretrained_writes_original_layout_after_load_scope_exits(self):
        DummyFusionConfig.model_type = f"dummy_fusion_{self._testMethodName}"

        with (
            tempfile.TemporaryDirectory() as source_dir,
            tempfile.TemporaryDirectory() as saved_dir,
            tempfile.TemporaryDirectory() as fallback_dir,
        ):
            DummyFusionModel(DummyFusionConfig()).save_pretrained(source_dir)
            original = load_file(os.path.join(source_dir, "model.safetensors"))

            fused = DummyFusionModel.from_pretrained(source_dir, fusion_config=self.fusion_config)
            self.assertIsInstance(fused.patch_embed.linear_proj, nn.Linear)
            # The global registries no longer hold the fusion rules the saver needs.
            self.assertEqual(get_patch_mapping(), {})
            self.assertIsNone(get_checkpoint_conversion_mapping(fused.config.model_type))

            fused.save_pretrained(saved_dir)
            saved = load_file(os.path.join(saved_dir, "model.safetensors"))

            # Force the save path that looks the converters up again from the model.
            fused._weight_conversions = None
            fused.save_pretrained(fallback_dir)
            fallback = load_file(os.path.join(fallback_dir, "model.safetensors"))

        for rewritten in (saved, fallback):
            self.assertEqual(set(rewritten), set(original))
            for key, value in original.items():
                torch.testing.assert_close(rewritten[key], value)
