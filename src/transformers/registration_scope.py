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

"""Load-scoped patch and checkpoint-conversion registrations.

`PreTrainedModel.from_pretrained` opens one scope while it registers fusions and kernel
replacements, builds the model, and collects weight transforms. Registrations made in that
window stay on the scope instead of the process-wide registries. The scope is dropped when
the load returns, including when it fails. Conversions collected for a successful load are
copied onto the model so later `get_model_conversion_mapping(model)` calls still see them.

Calls made while no scope is active keep writing to the global registries.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from copy import deepcopy
from typing import Any


class LoadRegistrationScope:
    """Registrations created by the `from_pretrained` call that is currently on the stack."""

    def __init__(self) -> None:
        self.patch_mapping: dict[str, Any] = {}
        self.conversion_mappings: dict[str, list] = {}
        self.registered_conversion_keys: set[str] = set()

    def attach_conversions(self, model: Any) -> None:
        """Copy this load's conversion rules onto `model` without replacing rules it already has.

        Nested loads attach first. Keys already stored on `model` belong to that inner load and
        stay in place; this scope only fills keys it introduced.
        """
        if not self.conversion_mappings:
            return

        existing = getattr(model, "_load_conversion_mappings", None)
        merged: dict[str, list] = dict(existing) if isinstance(existing, dict) else {}
        for key, mapping in self.conversion_mappings.items():
            if key not in merged:
                merged[key] = deepcopy(mapping)
        model._load_conversion_mappings = merged


_load_scopes: ContextVar[tuple[LoadRegistrationScope, ...]] = ContextVar("load_registration_scopes", default=())


def current_load_scope() -> LoadRegistrationScope | None:
    """Return the innermost active load scope, or `None` when no load is in progress."""
    scopes = _load_scopes.get()
    if not scopes:
        return None
    return scopes[-1]


def conversion_key_registered_for_load(key: str) -> bool:
    """Whether the current load registered a conversion mapping under `key`."""
    scope = current_load_scope()
    return scope is not None and key in scope.registered_conversion_keys


@contextmanager
def load_registration_scope() -> Iterator[LoadRegistrationScope]:
    """Collect patch and conversion registrations for the duration of one model load.

    Nested loads push another scope. Only the innermost scope is visible, so an inner load
    cannot observe or collide with registrations the outer load has not committed yet.
    """
    scope = LoadRegistrationScope()
    token: Token[tuple[LoadRegistrationScope, ...]] = _load_scopes.set((*_load_scopes.get(), scope))
    try:
        yield scope
    finally:
        _load_scopes.reset(token)
