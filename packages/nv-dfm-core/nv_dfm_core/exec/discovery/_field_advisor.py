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

from ._advised_values import AdvisedValue


# The decorator used in Adapters
def field_advisor(name, order=-1):
    """field_advisor(value: Any, context: BuilderEdge) are decorators that return
    an instance of AdvisedValue. If value is not Advise(), then the field_advisor
    is called during validation of a user-supplied value. The field_advisor can
    then either return an AdvisedValue and the framework will check if the user
    value works with it, or the field_advisor can directly return a AdvisedError
    if something is clearly wrong.
    The value is the value of the field attached to the field_advisor. The context.get()
    provides access to all previously handled fields
    """

    class FieldAdvisorConfig:
        """Params passed to the decorator are stored on the method as FieldAdvisorConfig"""

        field: str
        order: int

        def __init__(self, field, order):
            self.field = field
            # sort in order the advisors should get applied. Positive numbers come first, in order,
            # negative numbers come last (i.e. order=-1, the default,
            # is last, order=-2 are second to last)
            self.order = order if order >= 0 else 9999 + order

    def decorator(function):
        async def wrapper(*args, **kwargs):
            result = await function(*args, **kwargs)
            assert isinstance(result, AdvisedValue)
            return result

        wrapper.field_advisor_config = FieldAdvisorConfig(name, order)  # pyright: ignore[reportFunctionMemberAccess]
        return wrapper

    return decorator
