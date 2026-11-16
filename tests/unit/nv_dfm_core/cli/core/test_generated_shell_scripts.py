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

"""Run generated deployment scripts against fake tools, never real infrastructure."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nv_dfm_core.cli.core._generator import Generator
from nv_dfm_core.gen.apigen._config_models import ProvisionInfoObject

PAYLOAD = "value ' \" $(touch PWNED) `touch PWNED2`; * [x]"
LEGACY_PASSWORD = "CONFIGURED_PASSWORD_MUST_NOT_BE_RENDERED"


@pytest.fixture
def shell_tools(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / "tools"
    tools.mkdir()
    log = tmp_path / "calls.jsonl"
    for tool in ("docker", "podman", "helm", "kubectl"):
        executable = tools / tool
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "call = {'tool': Path(sys.argv[0]).name, 'args': sys.argv[1:]}\n"
            "assert 'DFM_DOCKER_REGISTRY_PASSWORD' not in os.environ\n"
            "assert 'PASSWORD' not in os.environ\n"
            "if len(sys.argv) > 1 and sys.argv[1] == 'login':\n"
            "    call['password'] = sys.stdin.read()\n"
            "with open(os.environ['DFM_TEST_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(call) + '\\n')\n"
        )
        executable.chmod(0o755)
    env = os.environ.copy()
    env.pop("DFM_DOCKER_REGISTRY_PASSWORD", None)
    env.pop("DFM_DOCKER_REGISTRY_USERNAME", None)
    env.pop("PASSWORD", None)
    env.update(PATH=f"{tools}:{env['PATH']}", DFM_TEST_CALLS=str(log))
    return env, log


def make_generator(tmp_path, monkeypatch, provision, server_sites=(), client_sites=()):
    federation_dir = tmp_path / "federation"
    federation_dir.mkdir()
    config = SimpleNamespace(
        project_path=tmp_path / "project.yaml",
        federation_workspace_dir=tmp_path / "workspace with spaces",
        federation_dir=federation_dir,
        provision_info=ProvisionInfoObject.model_validate(provision),
        flare_project_name="project",
        server_sites=list(server_sites),
        client_sites=list(client_sites),
    )
    generator = Generator("test_fed", config)
    output = tmp_path / "generated"
    output.mkdir()
    monkeypatch.setattr(generator, "_check_conditions", lambda *args: output)
    return generator, output


def docker_provision(tmp_path, *, password="", engine="docker"):
    dockerfile = tmp_path / "Dockerfile with spaces"
    dockerfile.write_text("FROM scratch\n")
    return {
        "docker": {
            "image": PAYLOAD,
            "tag": PAYLOAD,
            "dockerfile": {"dir": str(tmp_path), "file": dockerfile.name},
            "build": {
                "engine": engine,
                "arch": "amd64",
                "context": PAYLOAD,
                "save": {"enabled": True, "file": f"saved images/{PAYLOAD}.tar"},
                "push": {
                    "enabled": True,
                    "registry": {
                        "url": PAYLOAD,
                        "username": PAYLOAD,
                        "password": password,
                    },
                },
            },
        }
    }


def run_script(script, env, *args):
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
    return subprocess.run(
        ["bash", str(script), *args], env=env, text=True, capture_output=True
    )


def read_calls(log):
    return (
        [json.loads(line) for line in log.read_text().splitlines()]
        if log.exists()
        else []
    )


def assert_no_injection(tmp_path):
    assert not (tmp_path / "PWNED").exists()
    assert not (tmp_path / "PWNED2").exists()


@pytest.mark.parametrize("engine", ["docker", "podman"])
def test_docker_scripts_preserve_arguments_and_read_password_at_runtime(
    tmp_path, monkeypatch, shell_tools, capsys, engine
):
    env, log = shell_tools
    generator, output = make_generator(
        tmp_path,
        monkeypatch,
        docker_provision(tmp_path, password=LEGACY_PASSWORD, engine=engine),
    )
    generator.dockerfile(force=True, skip="dockerfile")
    for file in output.iterdir():
        assert LEGACY_PASSWORD not in file.read_text()
    assert "DFM_DOCKER_REGISTRY_PASSWORD" in capsys.readouterr().err

    # A legacy password must prompt migration before doing any build/push work.
    result = run_script(output / "build_image.sh", env)
    assert result.returncode != 0
    assert "DFM_DOCKER_REGISTRY_PASSWORD" in result.stderr
    assert read_calls(log) == []

    runtime_password = "-n literal ' \" $(touch PWNED) `touch PWNED2`\\end"
    env["DFM_DOCKER_REGISTRY_PASSWORD"] = runtime_password
    # An exported caller variable of the same name must not remain exported.
    env["PASSWORD"] = "caller-password"
    env["DFM_DOCKER_REGISTRY_USERNAME"] = "runtime " + PAYLOAD
    result = run_script(output / "build_image.sh", env)
    assert result.returncode == 0, result.stderr
    calls = read_calls(log)
    assert [call["tool"] for call in calls] == [engine] * 5
    build, save, login, tag, push = calls
    expected_build = [
        "build",
        "-t",
        f"{PAYLOAD}:{PAYLOAD}",
        "--platform",
        "linux/amd64",
    ]
    if engine == "podman":
        expected_build.append("--format=docker")
    expected_build += ["-f", str(tmp_path / "Dockerfile with spaces"), PAYLOAD]
    assert build["args"] == expected_build
    assert save["args"] == [
        "save",
        "--output",
        f"saved images/{PAYLOAD}.tar",
        f"{PAYLOAD}:{PAYLOAD}",
    ]
    assert login["args"] == [
        "login",
        PAYLOAD,
        "--username",
        "runtime " + PAYLOAD,
        "--password-stdin",
    ]
    assert login["password"] == runtime_password
    assert runtime_password not in (result.stdout + result.stderr)
    name = f"{PAYLOAD}/{PAYLOAD}:{PAYLOAD}"
    assert tag["args"] == ["tag", f"{PAYLOAD}:{PAYLOAD}", name]
    assert push["args"] == ["push", name]
    assert_no_injection(tmp_path)


def test_docker_existing_login_needs_no_password(tmp_path, monkeypatch, shell_tools):
    env, log = shell_tools
    generator, output = make_generator(
        tmp_path, monkeypatch, docker_provision(tmp_path)
    )
    generator.dockerfile(force=True, skip="dockerfile")
    result = run_script(output / "build_image.sh", env)
    assert result.returncode == 0, result.stderr
    assert [call["args"][0] for call in read_calls(log)] == [
        "build",
        "save",
        "tag",
        "push",
    ]
    assert_no_injection(tmp_path)


@pytest.mark.parametrize("push", [None, {"enabled": False}])
def test_docker_build_without_save_or_push(tmp_path, monkeypatch, shell_tools, push):
    env, log = shell_tools
    provision = docker_provision(tmp_path)
    provision["docker"]["build"].update(save=None, push=push)
    generator, output = make_generator(tmp_path, monkeypatch, provision)
    generator.dockerfile(force=True, skip="dockerfile")
    result = run_script(output / "build_image.sh", env)
    assert result.returncode == 0, result.stderr
    assert [call["args"][0] for call in read_calls(log)] == ["build"]
    assert_no_injection(tmp_path)


@pytest.mark.parametrize("source", ["env", "workspace"])
def test_helm_scripts_preserve_paths_values_and_extra_arguments(
    tmp_path, monkeypatch, shell_tools, source
):
    env, log = shell_tools
    # Workspace paths and server names are literal strings, even with shell syntax.
    servers = [PAYLOAD] if source == "workspace" else ["server"]
    clients = ["client"]
    generator, _ = make_generator(
        tmp_path,
        monkeypatch,
        {"helm": {"path": "helm", "name": PAYLOAD, "startup_package_source": source}},
        servers,
        clients,
    )
    generator.helm_chart(force=True)
    for site in servers + clients:
        if source == "workspace":
            package = (
                generator._workspace_path / "project" / "prod_00" / f"{site}.tar.gz"
            )
            package.parent.mkdir(parents=True, exist_ok=True)
        else:
            package = tmp_path / f"{site} package {PAYLOAD}.tgz"
            env[f"DFM_WORKSPACE_{site.upper()}_INIT_PATH"] = str(package)
        package.touch()
    values = tmp_path / f"values {PAYLOAD}.yaml"
    values.touch()
    result = run_script(
        tmp_path / "deploy.sh",
        env,
        "-i",
        "-u",
        "-f",
        str(values),
        "--",
        "--set-string",
        "key=" + PAYLOAD,
    )
    assert result.returncode == 0, result.stderr
    calls = read_calls(log)
    assert calls[0]["args"] == ["uninstall", PAYLOAD, "--ignore-not-found"]
    assert calls[-1]["args"] == [
        "upgrade",
        "--install",
        PAYLOAD,
        PAYLOAD,
        "--values",
        str(values),
        "--set-string",
        "key=" + PAYLOAD,
    ]
    created = [
        call["args"]
        for call in calls
        if call["tool"] == "kubectl" and call["args"][0] == "create"
    ]
    assert len(created) == 2
    for site, args in zip(servers + clients, created):
        assert args[:4] == [
            "create",
            "secret",
            "generic",
            f"flare-workspace-{site}-init",
        ]
        assert args[4].startswith("--from-file=")
        assert Path(args[4].removeprefix("--from-file=")).is_file()
    assert_no_injection(tmp_path)


def test_helm_missing_values_file_argument_fails(tmp_path, monkeypatch, shell_tools):
    env, log = shell_tools
    generator, _ = make_generator(
        tmp_path,
        monkeypatch,
        {"helm": {"path": "helm", "name": "release", "startup_package_source": "env"}},
    )
    generator.helm_chart(force=True)
    result = run_script(tmp_path / "deploy.sh", env, "-i", "-f")
    assert result.returncode != 0
    assert "Missing file" in result.stderr
    assert read_calls(log) == []
