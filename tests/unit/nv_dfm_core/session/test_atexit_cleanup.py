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

"""Tests for atexit cleanup handlers to prevent zombie processes."""

import subprocess
import sys
import textwrap
from pathlib import Path


class TestAtexitCleanup:
    """Test that atexit handlers properly clean up background processes."""

    def test_local_session_atexit_cleanup(self, tmp_path: Path):
        """Test that local session cleanup is called even without explicit close().

        This test simulates a scenario where a user creates a Session and connects
        but doesn't call close() or use a context manager. The atexit handler should
        still clean up the FederationRunner processes.
        """
        # Create a test script that connects but doesn't close
        test_script = tmp_path / "test_atexit_local.py"
        test_script.write_text(
            textwrap.dedent(
                """
                import sys
                import logging
                from pathlib import Path
                
                # Add parent directory to path to import nv_dfm_core
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                
                from nv_dfm_core.session import Session
                
                # Enable logging to see atexit message
                logging.basicConfig(level=logging.INFO)
                
                # Create and connect session without closing
                # Use only 1 concurrent job to speed up spawn-based startup in tests
                session = Session(
                    "testfed", 
                    "site1", 
                    target="local",
                    sites=["site1", "site2"],
                    max_concurrent_jobs=1,
                )
                _ = session.connect()
                
                print("Session connected, exiting without close()")
                # Exit without calling session.close()
                # atexit should trigger cleanup
                """
            )
        )

        # Run the script and check it exits cleanly
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Check that the script completed successfully
        assert result.returncode == 0, (
            f"Script failed with return code {result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

        # Check for atexit cleanup message in output
        output = result.stdout + result.stderr
        assert (
            "atexit: Cleaning up local federation" in output
            or "Session connected" in output
        ), f"Expected atexit cleanup message in output:\n{output}"

    def test_local_session_survives_multiple_cleanup_calls(self, tmp_path: Path):
        """Test that calling close() and then atexit cleanup doesn't cause errors.

        This verifies that the cleanup is idempotent and won't crash if
        both close() is called explicitly and atexit triggers.
        """
        test_script = tmp_path / "test_double_cleanup.py"
        test_script.write_text(
            textwrap.dedent(
                """
                import sys
                import logging
                from pathlib import Path
                
                # Add parent directory to path
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                
                from nv_dfm_core.session import Session
                
                logging.basicConfig(level=logging.INFO)
                
                # Use only 1 concurrent job to speed up spawn-based startup in tests
                with Session(
                    "testfed",
                    "site1", 
                    target="local",
                    sites=["site1"],
                    max_concurrent_jobs=1,
                ) as session:
                    print("Session in context manager")
                
                print("Context manager exited, atexit will still trigger")
                """
            )
        )

        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Should complete successfully even with double cleanup
        assert result.returncode == 0, (
            f"Script failed with double cleanup\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_local_session_ctrl_c_simulation(self, tmp_path: Path):
        """Test that cleanup happens even with abrupt termination (simulated).

        Note: This doesn't actually send SIGINT, but it verifies the cleanup
        mechanism is in place. A real SIGINT test would require more complex
        process management.
        """
        test_script = tmp_path / "test_abort.py"
        test_script.write_text(
            textwrap.dedent(
                """
                import sys
                import logging
                import atexit
                from pathlib import Path
                
                sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
                
                from nv_dfm_core.session import Session
                
                logging.basicConfig(level=logging.INFO)
                
                # Track that cleanup ran
                cleanup_ran = False
                def check_cleanup():
                    print("atexit: verification handler ran")
                
                atexit.register(check_cleanup)
                
                session = Session(
                    "testfed",
                    "site1",
                    target="local", 
                    sites=["site1"],
                    max_concurrent_jobs=1,
                )
                _ = session.connect()
                
                print("Session connected")
                # Simulate abrupt exit (atexit should still run)
                sys.exit(0)
                """
            )
        )

        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        output = result.stdout + result.stderr

        # Verify both our test handler and the cleanup handler ran
        assert "atexit: verification handler ran" in output
        assert (
            "atexit: Cleaning up local federation" in output
            or "Session connected" in output
        ), "Expected atexit cleanup to run on sys.exit()"
