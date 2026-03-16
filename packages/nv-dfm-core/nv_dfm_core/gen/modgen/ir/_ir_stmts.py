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
from typing import Literal

from pydantic import ConfigDict, JsonValue
from typing_extensions import override

from nv_dfm_core.api import BooleanExpression
from nv_dfm_core.api._stop_token import StopToken
from nv_dfm_core.api.pydantic import PolymorphicBaseModel
from nv_dfm_core.exec._frame import FlowInfo

from ._bool_expr_to_string import BoolExprToString
from ._gen_context import GenContext
from ._stmt_ref import StmtRef


class IRStmt(PolymorphicBaseModel, ABC):
    model_config: ConfigDict = ConfigDict(extra="forbid", frozen=True)  # pyright: ignore[reportIncompatibleVariableOverride]

    @abstractmethod
    def emit_python(self, ctx: GenContext) -> None: ...


class Comment(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.Comment"] = (
        "nv_dfm_core.gen.modgen.ir.Comment"
    )
    comment: str

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(f"# {self.comment}")


class BodyStartMarker(IRStmt):
    """Body start is the signal for the outer try-except block:

    async def t1_fire(...):
        try: # <--- BodyStartMarker
            ... transition body ...
            try: # <--- SendStartMarker
                ... sends and epilog ...
            except Exception as e: # <--- SendEndMarker
                raise PanicException(error=e)
        except PanicException as e: # <--- BodyEndMarker
            raise e
        except Exception as e: # <--- BodyEndMarker
            await t1_signal_error(dfm_context, frame, e)
    """

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.BodyStartMarker"] = (
        "nv_dfm_core.gen.modgen.ir.BodyStartMarker"
    )

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit("try:")
        # emit a triple-quote in case the body is empty
        ctx.emit_indented('"""start body"""')
        ctx.enter_scope()


class BodyEndMarker(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.BodyEndMarker"] = (
        "nv_dfm_core.gen.modgen.ir.BodyEndMarker"
    )

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.exit_scope()
        #
        ctx.emit("except PanicError as e:")
        ctx.emit_indented("# Panic in body, raising")
        ctx.emit_indented("raise e")
        ctx.emit("except Exception as e:")
        ctx.emit_indented("# Exception in body, signalling error")
        ctx.emit_indented(
            f"{ctx.error_param_name()} = ErrorToken.from_exception(error=e)"
        )
        ctx.emit_indented(ctx.signal_error_function_call())


class SendStartMarker(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.SendStartMarker"] = (
        "nv_dfm_core.gen.modgen.ir.SendStartMarker"
    )

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit("")
        ctx.emit("# start epilog")
        ctx.emit("try:")
        # emit a triple-quote in case the epilog is empty
        ctx.emit_indented('"""start epilog"""')
        ctx.enter_scope()


class SendEndMarker(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.SendEndMarker"] = (
        "nv_dfm_core.gen.modgen.ir.SendEndMarker"
    )

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.exit_scope()

        ctx.emit("except Exception as e:")
        ctx.emit_indented("# Error in epilog, panicking")
        ctx.emit_indented("raise PanicError(error=e)")


class ReadPlaceStmt(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.ReadPlaceStmt"] = (
        "nv_dfm_core.gen.modgen.ir.ReadPlaceStmt"
    )
    stmt_id: str
    place: str

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(f"{self.stmt_id} = {ctx.data_param_name()}[{repr(self.place)}]")


class AdapterCallStmt(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.AdapterCallStmt"] = (
        "nv_dfm_core.gen.modgen.ir.AdapterCallStmt"
    )
    provider: str | None = None
    stmt_id: str
    has_users: bool
    adapter: str
    literal_params: dict[str, tuple[str, JsonValue]]
    stmt_params: dict[str, StmtRef]
    is_async: bool

    @override
    def emit_python(self, ctx: GenContext) -> None:
        # Build parameter string
        params: list[str] = []
        for k, v in self.literal_params.items():
            params.append(f"{k}=tagged_json_value_to_object({repr(v)})")
        for k, v in self.stmt_params.items():
            params.append(f"{k}={v.to_python()}")
        param_str = ", ".join(params)
        # assemble the call prototype some_package_Adapter(params, ...)
        proto = f"{self.adapter.replace('.', '_')}({param_str})"
        # assemble the overall call string
        stmt_id_str = f"{self.stmt_id} = " if self.has_users else ""
        awaitstr = "await " if self.is_async else ""
        providerstr = f"{self.provider}." if self.provider else ""
        ctx.emit(
            f"{stmt_id_str}{awaitstr}{ctx.site_param_name()}.{providerstr}call_{proto}"
        )


class AdapterDiscoveryStmt(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.AdapterDiscoveryStmt"] = (
        "nv_dfm_core.gen.modgen.ir.AdapterDiscoveryStmt"
    )
    provider: str | None = None
    nodeid: int | str
    adapter: str
    literal_params: dict[str, tuple[str, JsonValue]]
    is_async: bool

    @override
    def emit_python(self, ctx: GenContext) -> None:
        # Build parameter string
        params: list[str] = []
        for k, v in self.literal_params.items():
            params.append(f"{k}=tagged_json_value_to_object({repr(v)})")
        param_str = ", ".join(params)
        proto = f"{self.adapter.replace('.', '_')}({param_str})"
        awaitstr = "await " if self.is_async else ""
        provstr = f"{self.provider}." if self.provider else ""
        ctx.emit(
            f"{ctx.dfm_context_param_name()}.add_discovery({repr(self.nodeid)}, {awaitstr}{ctx.site_param_name()}.{provstr}discover_{proto})"
        )


class SendDiscoveryStmt(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.SendDiscoveryStmt"] = (
        "nv_dfm_core.gen.modgen.ir.SendDiscoveryStmt"
    )

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_discovery(frame={ctx.frame_param_name()})"
        )


class BoolValueStmt(IRStmt):
    """Evaluates a boolean expression."""

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.BoolValueStmt"] = (
        "nv_dfm_core.gen.modgen.ir.BoolValueStmt"
    )
    stmt_id: str
    cond: BooleanExpression

    @override
    def emit_python(self, ctx: GenContext) -> None:
        # collect info from the boolean expression
        visitor = BoolExprToString()
        visitor.visit_boolean_expression(self.cond)
        cond_str = visitor.get_string()
        assert cond_str is not None
        ctx.emit(f"{self.stmt_id} = {cond_str}")


class TokenSend(IRStmt):
    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.TokenSend"] = (
        "nv_dfm_core.gen.modgen.ir.TokenSend"
    )
    # site to send data to. If None, it's the homesite
    job: str | None
    site: str
    place: str
    node_id: int | str | None
    is_yield: bool
    kind: Literal["control", "data"]
    # SsaRef or literal value, stored as a tagged json value
    data: StmtRef | FlowInfo | StopToken | tuple[str, JsonValue]

    @override
    def emit_python(self, ctx: GenContext) -> None:
        if isinstance(self.data, StmtRef):
            data_str = self.data.to_python()
        elif isinstance(self.data, FlowInfo):
            # emit it as literal
            data_str = repr(self.data)
        elif isinstance(self.data, StopToken):
            data_str = "StopToken()"
        else:
            data_str = f"tagged_json_value_to_object({repr(self.data)})"

        # job may or may not be None
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job={repr(self.job)}, to_site={repr(self.site)}, to_place={repr(self.place)}, frame={ctx.frame_param_name()}, is_yield={self.is_yield}, data={data_str}, node_id={repr(self.node_id)})"
        )


class LoopStmt(IRStmt):
    """The loop statement is an exit statement. It sends an incremented frame to the loop head control.
    Something like: jump_to(loop_head, frame.with_loop_inc())"""

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.LoopStmt"] = (
        "nv_dfm_core.gen.modgen.ir.LoopStmt"
    )
    # the control place of the loop head
    site: str
    place: str

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.site)}, to_place={repr(self.place)}, frame={ctx.frame_param_name()}, is_yield=False, data={ctx.frame_param_name()}.with_loop_inc())"
        )


class BranchStmt(IRStmt):
    """The branch statement is an exit statement.
    It's something like:
    if __condition__:
        jump_to(taken, frame)
    else:
        jump_to(not_taken, frame)
    """

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.BranchStmt"] = (
        "nv_dfm_core.gen.modgen.ir.BranchStmt"
    )
    # if True, the condition is expected to be a FlowInfo object. Branch is taken for a jump-like flow, otherwise it's a forked flow
    is_branch_on_flow_info: bool = False
    # the stmt_id of the condition value
    condition: StmtRef
    # sites to branch to
    taken_site: str
    taken_place: str
    taken_branch_frame: Literal["keep", "inc", "push", "pop"] = "keep"
    not_taken_site: str
    not_taken_place: str
    not_taken_branch_frame: Literal["keep", "inc", "push", "pop"] = "keep"

    def _frame_modifier_str(self, hint: Literal["keep", "inc", "push", "pop"]) -> str:
        if hint == "keep":
            return ""
        elif hint == "inc":
            return ".with_loop_inc()"
        elif hint == "push":
            return ".with_pushed_scope()"
        elif hint == "pop":
            return ".with_popped_scope()"
        else:
            raise ValueError(f"Invalid hint: {hint}")

    @override
    def emit_python(self, ctx: GenContext) -> None:
        cond_str = self.condition.to_python()
        if self.is_branch_on_flow_info:
            exit_flow_value = cond_str
            ctx.emit(f"if {cond_str}.is_jump_like():")
        else:
            exit_flow_value = "FlowInfo()"
            ctx.emit(f"if {cond_str}:")
        taken_frame_param = f"{ctx.frame_param_name()}{self._frame_modifier_str(self.taken_branch_frame)}"
        ctx.emit_indented(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.taken_site)}, to_place={repr(self.taken_place)}, frame={taken_frame_param}, is_yield=False, data=FlowInfo())"
        )
        ctx.emit("else:")
        not_taken_frame_param = f"{ctx.frame_param_name()}{self._frame_modifier_str(self.not_taken_branch_frame)}"
        ctx.emit_indented(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.not_taken_site)}, to_place={repr(self.not_taken_place)}, frame={not_taken_frame_param}, is_yield=False, data={exit_flow_value})"
        )


class DrainIteratorStmt(IRStmt):
    """Something like:
    === DrainIteratorStmt ===
    try:
        data = next(next_node_input) # or await anext(next_node_input)
        has_next = True
    except StopIteration: # or StopAsyncIteration
        data = None
        has_next = False
    === DrainIteratorStmt ===

    Epilog: <for each receiver> send(frame, data) and branch on has_next
    """

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.DrainIteratorStmt"] = (
        "nv_dfm_core.gen.modgen.ir.DrainIteratorStmt"
    )
    data_stmt_id: str
    condition_stmt_id: str
    # the stmt_id of the condition value
    iterator: StmtRef
    is_async: bool

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit("try:")
        # data = next(iterator)
        if self.is_async:
            ctx.emit_indented(
                f"{self.data_stmt_id} = await anext({self.iterator.to_python()})"
            )
            ctx.emit_indented(f"{self.condition_stmt_id} = True")
            ctx.emit("except StopAsyncIteration:")
        else:
            ctx.emit_indented(
                f"{self.data_stmt_id} = next({self.iterator.to_python()})"
            )
            ctx.emit_indented(f"{self.condition_stmt_id} = True")
            ctx.emit("except StopIteration:")
        # Except block:
        ctx.emit_indented(f"{self.data_stmt_id} = None")
        ctx.emit_indented(f"{self.condition_stmt_id} = False")


class ForkFromIteratorStmt(IRStmt):
    """Something like:
    loop_frame = frame.push()
    loop_count = 0
    while has_next:
        try: # isolate application errors that may happen in next()
            data = next(it)
        except StopIteration:
            break
        except Exception as e:
            data = ErrorToken(error=e)
        <for each data receiver> send(loop_frame, data)
        <for each sync follower> send(loop_frame, FlowInfo())
        to fork target, send(loop_frame, FlowInfo())
    <for each sync follower> send(frame, expect(loop_count)) # NOT the loop frame
    send(frame, expect(loop_count)) to stop iteration
    """

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.IfStmt"] = (
        "nv_dfm_core.gen.modgen.ir.IfStmt"
    )
    data_stmt_id: str
    iterator: StmtRef
    is_async: bool
    data_receivers: list[tuple[str, str]]
    sync_receivers: list[tuple[str, str]]
    fork_site: str
    fork_place: str
    stop_iteration_site: str
    stop_iteration_place: str

    @override
    def emit_python(self, ctx: GenContext) -> None:
        loop_frame_var = ctx.fresh_var()
        loop_count_var = ctx.fresh_var()
        ctx.emit(f"{loop_frame_var} = {ctx.frame_param_name()}.with_pushed_scope()")
        ctx.emit(f"{loop_count_var} = 0")
        ctx.emit("while True:")
        ctx.enter_scope()
        ctx.emit("try:")
        if self.is_async:
            ctx.emit_indented(
                f"{self.data_stmt_id} = await anext({self.iterator.to_python()})"
            )
            ctx.emit("except StopAsyncIteration:")
            ctx.emit_indented("break")
        else:
            ctx.emit_indented(
                f"{self.data_stmt_id} = next({self.iterator.to_python()})"
            )
            ctx.emit("except StopIteration:")
            ctx.emit_indented("break")
        ctx.emit("except Exception as e:")
        ctx.emit_indented(f"{self.data_stmt_id} = ErrorToken(error=e)")

        # send data to all data receivers
        for data_receiver in self.data_receivers:
            ctx.emit(
                f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(data_receiver[0])}, to_place={repr(data_receiver[1])}, frame={loop_frame_var}, is_yield=False, data={self.data_stmt_id})"
            )

        # send FlowInfo to all sync followers
        for sync_receiver in self.sync_receivers:
            ctx.emit(
                f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(sync_receiver[0])}, to_place={repr(sync_receiver[1])}, frame={loop_frame_var}, is_yield=False, data=FlowInfo())"
            )

        # ping the fork target
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.fork_site)}, to_place={repr(self.fork_place)}, frame={loop_frame_var}, is_yield=False, data=FlowInfo())"
        )
        # end while loop
        ctx.exit_scope()
        # after the loop, send the loop count to all sync followers
        for sync_receiver in self.sync_receivers:
            ctx.emit(
                f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(sync_receiver[0])}, to_place={repr(sync_receiver[1])}, frame={ctx.frame_param_name()}, is_yield=False, data=False)"
            )

        # send the count to the end of the loop
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.stop_iteration_site)}, to_place={repr(self.stop_iteration_place)}, frame={ctx.frame_param_name()}, is_yield=False, data=FlowInfo(hint={loop_count_var}))"
        )


class EndForkStmt(IRStmt):
    """The end fork statement is an exit statement. It sends the frame to a CountingPlace and then ends.
    Something like: send(site, place, frame)."""

    dfm_class_name: Literal["nv_dfm_core.gen.modgen.ir.EndForkStmt"] = (
        "nv_dfm_core.gen.modgen.ir.EndForkStmt"
    )
    site: str
    place: str

    @override
    def emit_python(self, ctx: GenContext) -> None:
        ctx.emit(
            f"await {ctx.dfm_context_param_name()}.send_to_place(to_job=None, to_site={repr(self.site)}, to_place={repr(self.place)}, frame={ctx.frame_param_name()}, is_yield=False, data={ctx.frame_param_name()})"
        )
