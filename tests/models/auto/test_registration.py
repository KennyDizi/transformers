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

import unittest
from collections import OrderedDict

from transformers import AutoConfig, AutoModel, AutoTokenizer, BertConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto.auto_factory import _LazyAutoMapping
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_MAPPING
from transformers.models.auto.tokenization_auto import (
    _SLOW_TOKENIZER_REGISTRATIONS,
    REGISTERED_TOKENIZER_CLASSES,
    TOKENIZER_MAPPING,
)
from transformers.pipelines import PIPELINE_REGISTRY
from transformers.quantizers.auto import (
    AUTO_QUANTIZATION_CONFIG_MAPPING,
    AUTO_QUANTIZER_MAPPING,
    register_quantization_config,
    register_quantizer,
)
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin


def _config(model_type: str):
    class Config(PreTrainedConfig):
        pass

    Config.model_type = model_type
    return Config


class RegistrationRuleTest(unittest.TestCase):
    def test_custom_config_same_class_is_noop_and_different_class_raises(self):
        model_type = "reg-config-collision"
        config_cls = _config(model_type)
        other = _config(model_type)

        try:
            AutoConfig.register(model_type, config_cls)
            AutoConfig.register(model_type, config_cls)
            self.assertIs(CONFIG_MAPPING[model_type], config_cls)

            with self.assertRaisesRegex(ValueError, "cannot be overwritten with") as ctx:
                AutoConfig.register(model_type, other)
            message = str(ctx.exception)
            self.assertIn(model_type, message)
            self.assertIn(other.__name__, message)
            self.assertIs(CONFIG_MAPPING[model_type], config_cls)

            AutoConfig.register(model_type, other, exist_ok=True)
            self.assertIs(CONFIG_MAPPING[model_type], other)
        finally:
            CONFIG_MAPPING._extra_content.pop(model_type, None)

    def test_redefined_config_class_needs_exist_ok(self):
        model_type = "reg-config-redefined"

        def make():
            class Config(PreTrainedConfig):
                pass

            Config.model_type = model_type
            return Config

        first = make()
        second = make()
        self.assertIsNot(first, second)
        self.assertEqual(first.__name__, second.__name__)

        try:
            AutoConfig.register(model_type, first)
            with self.assertRaisesRegex(ValueError, second.__name__):
                AutoConfig.register(model_type, second)
            self.assertIs(CONFIG_MAPPING[model_type], first)
            AutoConfig.register(model_type, second, exist_ok=True)
            self.assertIs(CONFIG_MAPPING[model_type], second)
        finally:
            CONFIG_MAPPING._extra_content.pop(model_type, None)

    def test_builtin_config_stays_protected(self):
        class BertShadow(BertConfig):
            pass

        try:
            with self.assertRaisesRegex(ValueError, "already used by a Transformers config"):
                AutoConfig.register("bert", BertConfig)
            with self.assertRaisesRegex(ValueError, "already used by a Transformers config"):
                AutoConfig.register("bert", BertShadow)
            self.assertIs(CONFIG_MAPPING["bert"], BertConfig)

            AutoConfig.register("bert", BertShadow, exist_ok=True)
            self.assertIs(CONFIG_MAPPING["bert"], BertShadow)
        finally:
            CONFIG_MAPPING._extra_content.pop("bert", None)

    def test_custom_model_and_builtin_model(self):
        model_type = "reg-model-collision"
        config_cls = _config(model_type)

        class Model:
            config_class = config_cls

        class Other:
            config_class = config_cls

        class PretendBert:
            config_class = BertConfig

        try:
            AutoConfig.register(model_type, config_cls)
            AutoModel.register(config_cls, Model)
            AutoModel.register(config_cls, Model)
            self.assertIs(MODEL_MAPPING[config_cls], Model)

            with self.assertRaisesRegex(ValueError, "cannot be overwritten with") as ctx:
                AutoModel.register(config_cls, Other)
            self.assertIn(repr(config_cls), str(ctx.exception))
            self.assertIn(Other.__name__, str(ctx.exception))
            self.assertIs(MODEL_MAPPING[config_cls], Model)

            AutoModel.register(config_cls, Other, exist_ok=True)
            self.assertIs(MODEL_MAPPING[config_cls], Other)

            with self.assertRaisesRegex(ValueError, "already used by a Transformers model"):
                AutoModel.register(BertConfig, PretendBert)
            with self.assertRaisesRegex(ValueError, "already used by a Transformers model"):
                AutoModel.register(BertConfig, PretendBert, exist_ok=False)
            AutoModel.register(BertConfig, PretendBert, exist_ok=True)
            self.assertIs(MODEL_MAPPING._extra_content[BertConfig], PretendBert)
        finally:
            CONFIG_MAPPING._extra_content.pop(model_type, None)
            MODEL_MAPPING._extra_content.pop(config_cls, None)
            MODEL_MAPPING._extra_content.pop(BertConfig, None)

    def test_native_config_registration_returns_early_unless_exist_ok(self):
        class FakeModel:
            pass

        untouched = _LazyAutoMapping(OrderedDict({"bert": "BertConfig"}), OrderedDict())
        untouched.register(BertConfig, FakeModel)
        self.assertNotIn(BertConfig, untouched._extra_content)
        untouched.register(BertConfig, FakeModel, exist_ok=True)
        self.assertIs(untouched._extra_content[BertConfig], FakeModel)

        protected = _LazyAutoMapping(OrderedDict({"bert": "BertConfig"}), OrderedDict({"bert": "BertModel"}))
        with self.assertRaisesRegex(ValueError, "already used by a Transformers model"):
            protected.register(BertConfig, FakeModel)
        protected.register(BertConfig, FakeModel, exist_ok=True)
        self.assertIs(protected._extra_content[BertConfig], FakeModel)

    def test_mapping_compares_class_objects_inside_containers(self):
        class Cfg(PreTrainedConfig):
            model_type = "reg-container"

        class A:
            pass

        class B:
            pass

        class A2:
            pass

        mapping = _LazyAutoMapping(OrderedDict(), OrderedDict())
        mapping.register(Cfg, {"pil": A})
        stored = mapping._extra_content[Cfg]
        mapping.register(Cfg, {"pil": A})
        self.assertIs(mapping._extra_content[Cfg], stored)

        mapping.register(Cfg, {"tv": B})
        self.assertIs(mapping._extra_content[Cfg]["pil"], A)
        self.assertIs(mapping._extra_content[Cfg]["tv"], B)

        with self.assertRaisesRegex(ValueError, "A2") as ctx:
            mapping.register(Cfg, {"pil": A2})
        self.assertIn("pil", str(ctx.exception))
        self.assertIs(mapping._extra_content[Cfg]["pil"], A)
        self.assertIs(mapping._extra_content[Cfg]["tv"], B)

        mapping.register(Cfg, {"pil": A2}, exist_ok=True)
        self.assertIs(mapping._extra_content[Cfg]["pil"], A2)
        self.assertIs(mapping._extra_content[Cfg]["tv"], B)

        class TupleCfg(PreTrainedConfig):
            model_type = "reg-tuple"

        mapping.register(TupleCfg, (A, B))
        stored_tuple = mapping._extra_content[TupleCfg]
        mapping.register(TupleCfg, (A, B))
        self.assertIs(mapping._extra_content[TupleCfg], stored_tuple)
        with self.assertRaisesRegex(ValueError, "A2"):
            mapping.register(TupleCfg, (A, A2))
        self.assertIs(mapping._extra_content[TupleCfg], stored_tuple)

    def test_tokenizer_slow_then_fast_and_other_collisions(self):
        model_type = "reg-tokenizer"
        config_cls = _config(model_type)

        class Slow:
            pass

        class Fast:
            pass

        class OtherSlow:
            pass

        try:
            AutoConfig.register(model_type, config_cls)
            AutoTokenizer.register(config_cls, slow_tokenizer_class=Slow)
            self.assertIs(TOKENIZER_MAPPING[config_cls], Slow)
            AutoTokenizer.register(config_cls, slow_tokenizer_class=Slow)
            self.assertIs(TOKENIZER_MAPPING[config_cls], Slow)

            with self.assertRaisesRegex(ValueError, "OtherSlow"):
                AutoTokenizer.register(config_cls, slow_tokenizer_class=OtherSlow)
            self.assertIs(TOKENIZER_MAPPING[config_cls], Slow)

            AutoTokenizer.register(config_cls, fast_tokenizer_class=Fast)
            self.assertIs(TOKENIZER_MAPPING[config_cls], Fast)

            with self.assertRaisesRegex(ValueError, "already used by a Transformers model"):
                AutoTokenizer.register(BertConfig, slow_tokenizer_class=Slow)
        finally:
            CONFIG_MAPPING._extra_content.pop(model_type, None)
            TOKENIZER_MAPPING._extra_content.pop(config_cls, None)
            _SLOW_TOKENIZER_REGISTRATIONS.pop(config_cls, None)
            REGISTERED_TOKENIZER_CLASSES.pop("Slow", None)
            REGISTERED_TOKENIZER_CLASSES.pop("Fast", None)
            REGISTERED_TOKENIZER_CLASSES.pop("OtherSlow", None)

    def test_quantizer_and_quantization_config_registration(self):
        class CustomConfig(QuantizationConfigMixin):
            pass

        class OtherConfig(QuantizationConfigMixin):
            pass

        class CustomQuantizer(HfQuantizer):
            pass

        class OtherQuantizer(HfQuantizer):
            pass

        def make_quantizer():
            class CustomQuantizer(HfQuantizer):
                pass

            return CustomQuantizer

        original_quantizer = AUTO_QUANTIZER_MAPPING["awq"]
        original_config = AUTO_QUANTIZATION_CONFIG_MAPPING["awq"]
        try:
            register_quantizer("reg-quantizer")(CustomQuantizer)
            register_quantizer("reg-quantizer")(CustomQuantizer)
            self.assertIs(AUTO_QUANTIZER_MAPPING["reg-quantizer"], CustomQuantizer)

            with self.assertRaisesRegex(ValueError, "cannot be overwritten with") as ctx:
                register_quantizer("reg-quantizer")(OtherQuantizer)
            self.assertIn("reg-quantizer", str(ctx.exception))
            self.assertIn(OtherQuantizer.__name__, str(ctx.exception))
            self.assertIs(AUTO_QUANTIZER_MAPPING["reg-quantizer"], CustomQuantizer)

            redefined = make_quantizer()
            self.assertEqual(redefined.__name__, CustomQuantizer.__name__)
            self.assertIsNot(redefined, CustomQuantizer)
            with self.assertRaisesRegex(ValueError, redefined.__name__):
                register_quantizer("reg-quantizer")(redefined)

            register_quantizer("reg-quantizer", exist_ok=True)(OtherQuantizer)
            self.assertIs(AUTO_QUANTIZER_MAPPING["reg-quantizer"], OtherQuantizer)

            with self.assertRaisesRegex(ValueError, "Quantizer 'awq' already registered"):
                register_quantizer("awq")(CustomQuantizer)
            with self.assertRaisesRegex(ValueError, "Quantizer 'awq' already registered"):
                register_quantizer("awq")(original_quantizer)
            register_quantizer("awq", exist_ok=True)(CustomQuantizer)
            self.assertIs(AUTO_QUANTIZER_MAPPING["awq"], CustomQuantizer)

            register_quantization_config("reg-quantizer")(CustomConfig)
            register_quantization_config("reg-quantizer")(CustomConfig)
            with self.assertRaisesRegex(ValueError, "cannot be overwritten with") as config_error:
                register_quantization_config("reg-quantizer")(OtherConfig)
            self.assertIn("reg-quantizer", str(config_error.exception))
            self.assertIn(OtherConfig.__name__, str(config_error.exception))
            self.assertIs(AUTO_QUANTIZATION_CONFIG_MAPPING["reg-quantizer"], CustomConfig)
            register_quantization_config("reg-quantizer", exist_ok=True)(OtherConfig)
            self.assertIs(AUTO_QUANTIZATION_CONFIG_MAPPING["reg-quantizer"], OtherConfig)

            with self.assertRaisesRegex(ValueError, "Config 'awq' already registered"):
                register_quantization_config("awq")(CustomConfig)
            register_quantization_config("awq", exist_ok=True)(CustomConfig)
            self.assertIs(AUTO_QUANTIZATION_CONFIG_MAPPING["awq"], CustomConfig)
        finally:
            AUTO_QUANTIZER_MAPPING.pop("reg-quantizer", None)
            AUTO_QUANTIZATION_CONFIG_MAPPING.pop("reg-quantizer", None)
            AUTO_QUANTIZER_MAPPING["awq"] = original_quantizer
            AUTO_QUANTIZATION_CONFIG_MAPPING["awq"] = original_config

    def test_pipeline_builtin_requires_exist_ok_and_custom_uses_class_identity(self):
        class Pipe:
            pass

        class Other:
            pass

        def make_pipe():
            class Pipe:
                pass

            return Pipe

        builtin = "text-classification"
        _, original_task, _ = PIPELINE_REGISTRY.check_task(builtin)
        custom = "reg-pipeline-collision"
        try:
            with self.assertRaisesRegex(ValueError, "text-classification") as ctx:
                PIPELINE_REGISTRY.register_pipeline(builtin, Pipe)
            self.assertIn("already used by a Transformers pipeline", str(ctx.exception))
            self.assertIn(Pipe.__name__, str(ctx.exception))
            _, unchanged, _ = PIPELINE_REGISTRY.check_task(builtin)
            self.assertEqual(unchanged["impl"], original_task["impl"])

            with self.assertRaisesRegex(ValueError, "already used by a Transformers pipeline"):
                PIPELINE_REGISTRY.register_pipeline(builtin, original_task["impl"])

            PIPELINE_REGISTRY.register_pipeline(builtin, Pipe, exist_ok=True)
            _, overridden, _ = PIPELINE_REGISTRY.check_task(builtin)
            self.assertIs(overridden["impl"], Pipe)

            PIPELINE_REGISTRY.register_pipeline(custom, pipeline_class=Pipe, type="text")
            stored = PIPELINE_REGISTRY.supported_tasks[custom]
            PIPELINE_REGISTRY.register_pipeline(custom, pipeline_class=Pipe, type="audio")
            self.assertIs(PIPELINE_REGISTRY.supported_tasks[custom], stored)
            self.assertEqual(stored["type"], "text")

            with self.assertRaisesRegex(ValueError, "Other") as custom_error:
                PIPELINE_REGISTRY.register_pipeline(custom, pipeline_class=Other)
            self.assertIn(custom, str(custom_error.exception))
            self.assertIs(PIPELINE_REGISTRY.supported_tasks[custom]["impl"], Pipe)

            redefined = make_pipe()
            self.assertIsNot(redefined, Pipe)
            self.assertEqual(redefined.__name__, Pipe.__name__)
            with self.assertRaisesRegex(ValueError, "cannot be overwritten with"):
                PIPELINE_REGISTRY.register_pipeline(custom, pipeline_class=redefined)

            PIPELINE_REGISTRY.register_pipeline(custom, pipeline_class=Other, type="audio", exist_ok=True)
            self.assertIs(PIPELINE_REGISTRY.supported_tasks[custom]["impl"], Other)
            self.assertEqual(PIPELINE_REGISTRY.supported_tasks[custom]["type"], "audio")
        finally:
            PIPELINE_REGISTRY.supported_tasks[builtin] = original_task
            PIPELINE_REGISTRY.supported_tasks.pop(custom, None)
