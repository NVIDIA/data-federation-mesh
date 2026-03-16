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

import pickle

import numpy as np
import pytest

from nv_dfm_core.api import PickledObject


# Mock HDF5 object reference class that mimics the unpicklable behavior
class MockHDF5ObjectReference:
    """Mock HDF5 object reference that is not picklable."""

    def __init__(self, ref_id=None):
        self.ref_id = ref_id

    def __repr__(self):
        return "<HDF5 object reference>"

    def __reduce__(self):
        # Make it unpicklable by raising an error
        raise pickle.PickleError("HDF5 object references cannot be pickled")


def test_dict_with_hdf5_references():
    """Test serialization/deserialization of a dictionary containing HDF5 object references.

    This test is designed to drive implementation of a solution for handling
    unpicklable objects (like HDF5 references) within picklable structures.
    """
    pytest.skip(
        "Future work: Implement handling of unpicklable objects (e.g., HDF5 references) "
        "within picklable structures. The current PickledObject implementation cannot "
        "serialize structures containing unpicklable objects like HDF5 object references."
    )

    # Create HDF5 object references (unpicklable objects)
    hdf5_refs = [MockHDF5ObjectReference(ref_id=i) for i in range(3)]

    # Minimal structure: dict with HDF5 references in a numpy object array
    original = {
        "cam": np.array(hdf5_refs, dtype=object),
    }

    # This test will fail with current implementation because HDF5 references are unpicklable
    obj = PickledObject(value=original)
    json_str = obj.model_dump_json()
    new_obj = PickledObject.model_validate_json(json_str)

    # Verify the structure is preserved
    assert "cam" in new_obj.value
    assert new_obj.value["cam"].dtype == object
    assert len(new_obj.value["cam"]) == len(original["cam"])
