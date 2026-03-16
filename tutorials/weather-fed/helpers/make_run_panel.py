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

import ipywidgets as widgets
from ipywidgets import HBox, VBox


def make_run_panel(
    controls,  # list[Widget] shown before the Run button
    run_callback,  # fn(params_or_controls) -> result
    display_callback,  # fn(result, output) -> None
    *,
    values_fn=None,  # optional fn(controls) -> params dict
    run_button_description="Run",
    run_button_tooltip="Run",
    run_button_icon="play",
    running_description="Running...",
    running_icon="hourglass",
    button_style="primary",
    output_layout=None,
    container_width="1200px",
    gap="8px",
    auto_run=False,
):
    """
    Build a run panel with a button and output area, wiring in callbacks.

    Behavior:
      - Disables the button during run.
      - Clears the output before each run.
      - Calls run_callback with either params (if values_fn provided) or controls.
      - Passes the result to display_callback(result, output).
      - Restores the button to the configured idle state afterward.

    Returns:
      dict(ui=<VBox>, button=<Button>, output=<Output>)
    """
    output_layout = output_layout or {
        "border": "1px solid #ddd",
        "padding": "6px",
        "width": "1000px",
    }

    run_btn = widgets.Button(
        description=run_button_description,
        button_style=button_style,
        tooltip=run_button_tooltip,
        icon=run_button_icon,
    )

    output = widgets.Output(layout=output_layout)

    def on_run_click(b):
        with output:
            b.disabled = True
            b.description = running_description
            b.icon = running_icon
            try:
                output.clear_output()
                if values_fn is not None:
                    params = values_fn(controls)
                    result = run_callback(params)
                else:
                    result = run_callback(controls)
                display_callback(result, output)
            except Exception as e:
                print(f"❌ Error: {e!r}")
                raise
            finally:
                b.disabled = False
                b.description = run_button_description
                b.icon = run_button_icon

    run_btn.on_click(on_run_click)

    ctrl_box = HBox([*controls, run_btn], layout=widgets.Layout(gap=gap))
    ui = VBox([ctrl_box, output], layout=widgets.Layout(width=container_width))

    if auto_run:
        on_run_click(run_btn)

    return ui
