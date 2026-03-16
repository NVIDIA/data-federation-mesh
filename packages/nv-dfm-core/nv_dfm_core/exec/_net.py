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

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nv_dfm_core.exec import TransitionTryActivateFunc


# NOTE: subclasses should also add the @dataclass decorator, it's not fully inherited really
@dataclass
class Net(ABC):
    """
    The modgen tool will generate modules that each contain exactly one
    ThisNet class which inherits from Net. The NetRunner retrieves the ThisNet
    class form the module, instantiates and executes it.
    The instance fields are the places
    and transitions are methods. The get_activation_functions method
    is used to get the activation functions for the transitions and is
    expected to be filled in by the code generator.
    NOTE: The generated ThisNet classes must have @dataclass annotations
    because the NetRunner uses dataclasses.asdict to get the places.
    """

    @abstractmethod
    def get_activation_functions(self) -> list[TransitionTryActivateFunc]:
        pass

    def assert_places_empty(self) -> None:
        """Assert that all places are empty."""
        for place in self.__dict__.values():
            from nv_dfm_core.exec._places import Place

            if isinstance(place, Place):
                place.assert_empty()
