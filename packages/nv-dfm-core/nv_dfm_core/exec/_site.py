# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import appdirs  # pyright: ignore[reportMissingTypeStubs]
from typing_extensions import Self
from upath import UPath

from nv_dfm_core.telemetry import telemetry_enabled

from ._dfm_context import DfmContext
from ._fsspec_config import FsspecConfig
from ._secrets_vault import SecretsVault
from ._secrets_vault_config import SecretsVaultConfig

if TYPE_CHECKING:
    from nv_dfm_core.telemetry import SpanBuilder


class Site(ABC):
    """
    Base class for all sites. The apigen tool will generate a concrete
    implementation of this class for each federation site.
    """

    def __init__(
        self,
        dfm_context: DfmContext,
        cache_config: FsspecConfig,
        secrets_vault_config: SecretsVaultConfig,
    ):
        self._dfm_context: DfmContext = dfm_context
        self._cache_conf: FsspecConfig = cache_config
        try:
            self._secrets_vault: SecretsVault = SecretsVault.from_config(
                secrets_vault_config
            )
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to instantiate secrets vault for site {dfm_context.federation_name} from config '{repr(secrets_vault_config)}'. Falling back to default env var secrets vault! Error was: {e}"
            )
            self._secrets_vault = SecretsVault()

    @property
    def dfm_context(self) -> DfmContext:
        return self._dfm_context

    @property
    def api_version(self) -> str:
        return self._dfm_context.api_version

    @property
    def federation_name(self) -> str:
        return self._dfm_context.federation_name

    @property
    def site(self) -> Self:
        """Returns the site itself"""
        return self

    def cache_storage(self, subpath: str | Path | UPath | None = None) -> UPath:
        """
        Returns a UPath, which is a pathlib.Path-like object that is backed by
        an fsspec filestystem, pointing at the location that should be used to
        cache data at this site. If subpath is provided, the returned path is
        already set to basepath/subpath, otherwise the location is the UPath of the
        top level cache directory for this site.

        If no base_path was configured for this site, the user cache directory is used.
        The user cache directory is platform dependent and is typically located in
        ~/.cache/dfm/<federation_name>.

        Returns:
            - The UPath object pointing at either the top level cache directory for this site or
              the subpath inside the cache directory if provided. It is ensured that the returned
              path is an existing directory, otherwise an error is raised.
        """
        # collect the pieces
        base_path = (  # pyright: ignore[reportUnknownVariableType]
            self._cache_conf.base_path
            if self._cache_conf.base_path
            else appdirs.user_cache_dir(self.federation_name)  # pyright: ignore[reportUnknownMemberType]
        )
        assert isinstance(base_path, str)
        subpath = subpath if subpath else ""

        # construct the path
        path = UPath(
            base_path,
            subpath,
            protocol=self._cache_conf.protocol,
            **self._cache_conf.storage_options,
        )

        # check it's all okay
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        assert path.exists()
        if not path.is_dir():
            raise FileExistsError(f"Cache directory {path} is not a directory")

        return path

    def secret_for_key(self, key: str) -> str:
        """
        Returns the secret for the given key.
        """
        if self._secrets_vault:
            return self._secrets_vault.secret_for_key(key)
        else:
            raise ValueError(
                f"No secrets vault configured for site {self.federation_name}"
            )

    @contextmanager
    def telemetry_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator["SpanBuilder | None", None, None]:
        """Create a telemetry span for tracking operations within adapters.

        This is the recommended way for adapter authors to add custom telemetry
        to their adapter implementations. Spans are only recorded when telemetry
        is enabled (DFM_TELEMETRY_ENABLED=true).

        Args:
            name: Name of the span (e.g., "fetch_data", "transform", "validate")
            attributes: Optional dict of attributes to attach to the span

        Yields:
            SpanBuilder if telemetry is enabled, None otherwise.
            Call span.set_ok() on success or span.set_error(msg) on failure.

        Example:
            class MyAdapter:
                def __init__(self, site):
                    self.site = site

                async def body(self, **kwargs):
                    # Track a database query
                    with self.site.telemetry_span("db_query", {"table": "users"}) as span:
                        results = await self.query_database()
                        if span:
                            span.set_attribute("row_count", len(results))
                            span.set_ok()
                        return results

                    # Track multiple operations
                    with self.site.telemetry_span("validate_input") as span:
                        self.validate(kwargs)
                        if span:
                            span.set_ok()

                    with self.site.telemetry_span("transform_data") as span:
                        result = self.transform(data)
                        if span:
                            span.set_ok()
                        return result
        """
        if not telemetry_enabled():
            yield None
            return

        collector = self._dfm_context.telemetry_collector
        if collector is None:
            yield None
            return

        with collector.span(name, attributes=attributes) as span:
            yield span
