"""s0uRc3 simplified terminal GUI (Textual).

Same engine as tui/app.py (AgentLoop, StreamSplitter, permission modal,
chat history) but built to actually be smooth on a fast local model. The
full TUI hops onto the UI thread and does a full rescan + full-widget
rebuild on *every token* (see tui/app.py's _on_token -> call_from_thread ->
render_stream -> ChatMessage.refresh_body -> Static.update, with a
query_one + scroll_end thrown in per token too). At even 20 tokens/sec
that's 20 full round-trips a second with no batching — the visible
stutter this module exists to remove.

The fix: tokens land in a plain list behind a lock from the worker
thread; a single Textual timer drains that list at a fixed ~20Hz and
does exactly one redraw per tick, however many tokens arrived since the
last one. StreamSplitter.feed() re-derives its view from the *whole*
accumulated buffer regardless of how many characters you hand it in one
call (see tui/stream.py — it just does `self.buf += token`), so feeding
it one batched chunk per tick is exactly equivalent to feeding it token
by token and only looking at the last result. Nothing about the
think/answer/JSON-fence parsing changes; only how often the UI looks at
it does.

Deliberately smaller in scope than tui/app.py: no in-app model picker or
config screens. Pick the model with `--model` (main.py) or the config
default; switch later with `/model <spec>`.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from core.llama_server import current_model_path, stop as stop_llama
from core.loop import AgentLoop
from core.model import choice_from_spec
from core.paths import AgentPaths
from core.permission import DENY
from core.reflection import reward_last_run
from tui.history import Message, load_history, save_history
from tui.runtime import create_loop, read_config
from tui.stream import StreamSplitter, StreamView

log = logging.getLogger("s0urc3.tui.simple")

REDRAW_HZ = 20  # throttle: one redraw per tick, no matter how many tokens arrived

HELP_TEXT = """\
Commands
  /help              this list
  /model [spec]      switch model (GGUF path or server id); no arg reloads default
  /dry-run           toggle deterministic policy (no LLM)
  /reward            reinforce the last run
  /clear             wipe chat history on this machine
  /quit              exit

Esc stops the current run. y/n/a answers a permission prompt.
"""


def finish_summary(result: dict[str, Any] | None) -> str:
    if not isinstance(result, dict):
        return ""
    finish = result.get("finish")
    if isinstance(finish, dict) and finish.get("summary"):
        return str(finish["summary"]).strip()
    err = result.get("error")
    if err and not result.get("ok"):
        return str(err)
    return ""


def format_args(args: Any) -> str:
    if not args:
        return "(none)"
    try:
        return json.dumps(args, indent=2, default=str)
    except TypeError:
        return str(args)


def run_simple_app(*, dry_run: bool = False, max_steps: int | None = None, model: str | None = None) -> int:
    try:
        app = build_simple_app(dry_run=dry_run, max_steps=max_steps, model=model)
    except ImportError:
        print("The terminal GUI needs textual. Install with:  pip install textual", flush=True)
        return 2
    app.run()
    return 0


def build_simple_app(*, dry_run: bool = False, max_steps: int | None = None, model: str | None = None):
    from rich.markup import escape
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.screen import ModalScreen
    from textual.widgets import Button, Input, RichLog, Static

    def render_line(who: str, color: str, thinking: str, answer: str) -> str:
        parts = [f"[bold {color}]{who}[/]"]
        thinking = (thinking or "").rstrip()
        answer = (answer or "").rstrip()
        if thinking or (who != "You" and not answer):
            label = "Thinking…" if who != "You" and not answer else "Thought"
            parts.append(f"[dim]{label}[/] [italic dim]{escape(thinking)}[/]" if thinking else f"[dim]{label}[/]")
        if answer:
            parts.append(escape(answer))
        return "\n".join(parts)

    class PermissionModal(ModalScreen[str]):
        BINDINGS = [
            Binding("y", "allow", "Allow", show=True),
            Binding("n", "deny", "Deny", show=True),
            Binding("a", "allow_run", "Allow for this run", show=True),
            Binding("escape", "deny", "Deny", show=False),
        ]

        def __init__(self, request: dict[str, Any]) -> None:
            super().__init__()
            self.request = request

        def compose(self) -> ComposeResult:
            req = self.request
            body = (
                f"[bold #F59E0B]Permission needed[/]\n\n"
                f"[dim]kind[/]  {req.get('kind') or 'use_tool'}\n"
                f"[dim]name[/]  {req.get('name') or '—'}\n"
            )
            if req.get("description"):
                body += f"[dim]what[/]  {req['description']}\n"
            body += f"\n[dim]args[/]\n{format_args(req.get('args'))}"
            with Vertical(id="perm-dialog"):
                yield Static(body, id="perm-body")
                with Horizontal(id="perm-buttons"):
                    yield Button("Allow  (y)", id="allow", variant="success")
                    yield Button("Deny  (n)", id="deny", variant="error")
                    yield Button("Allow for run  (a)", id="allow_run", variant="primary")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(str(event.button.id))

        def action_allow(self) -> None:
            self.dismiss("allow")

        def action_deny(self) -> None:
            self.dismiss("deny")

        def action_allow_run(self) -> None:
            self.dismiss("allow_run")

    class SimpleApp(App):
        TITLE = "s0uRc3"
        CSS = """
        Screen { background: #120e0a; color: #f5efe6; }
        #status { dock: top; height: 1; padding: 0 1; color: #c4b5a0; background: #1a1410; }
        #chat { height: 1fr; padding: 0 1; scrollbar-gutter: stable; }
        #live { height: auto; min-height: 0; padding: 0 1; color: #f5efe6; }
        #composer { dock: bottom; height: auto; padding: 0 1 1 1; }
        #prompt { width: 1fr; background: #1c1612; border: tall #F59E0B; }
        PermissionModal { align: center middle; }
        #perm-dialog {
            width: 72; height: auto; max-height: 28; padding: 1 2;
            background: #1c1612; border: thick #F59E0B;
        }
        #perm-buttons { height: auto; padding-top: 1; }
        #perm-buttons Button { margin-right: 1; }
        """
        BINDINGS = [Binding("escape", "cancel_run", "Stop")]

        def __init__(
            self,
            paths: AgentPaths,
            *,
            dry_run: bool = False,
            max_steps: int | None = None,
            model: str | None = None,
        ) -> None:
            super().__init__()
            self.paths = paths
            self.dry_run = dry_run
            self.max_steps = max_steps
            self.preset_model = (model or "").strip() or None
            self.loop: AgentLoop | None = None
            self.messages: list[Message] = []
            self._busy = False
            self._splitter = StreamSplitter()
            self._perm_done: threading.Event | None = None
            self._perm_box: dict[str, str] = {}
            self._tok_lock = threading.Lock()
            self._pending_tokens: list[str] = []
            self._streaming = False

        def compose(self) -> ComposeResult:
            yield Static("Starting…", id="status")
            yield RichLog(id="chat", wrap=True, markup=True, highlight=False, auto_scroll=True)
            yield Static("", id="live")
            with Horizontal(id="composer"):
                yield Input(placeholder="Ask s0uRc3  ·  /help for commands", id="prompt")

        def on_mount(self) -> None:
            # Cache every widget reference once instead of query_one() on every
            # token/turn — the other half of the throttling fix.
            self._chat = self.query_one("#chat", RichLog)
            self._live_widget = self.query_one("#live", Static)
            self._status_widget = self.query_one("#status", Static)
            self._prompt = self.query_one("#prompt", Input)

            self.messages = load_history(self.paths.db)
            if not self.messages:
                self.messages.append(
                    Message(
                        who="s0uRc3",
                        text="Hey — give me a goal and I'll loop perceive -> think -> act. I'll ask before any tool.",
                        is_agent=True,
                    )
                )
            for m in self.messages:
                self._write_settled(m)
            self._prompt.focus()

            # One timer, always running, cheap no-op when nothing streamed
            # since the last tick — this is what caps redraws to REDRAW_HZ
            # instead of one per token.
            self.set_interval(1.0 / REDRAW_HZ, self._drain_tokens)

            if self.dry_run:
                self.set_status("Dry-run  ·  no llama-server")
                self.run_worker(self._bootstrap, thread=True, exclusive=True, group="boot")
                return
            spec = self.preset_model or ((read_config(self.paths).get("llm") or {}).get("model"))
            self.set_status(f"Loading {spec}…" if spec else "Loading default model…")
            self.run_worker(lambda: self._load_model(spec), thread=True, exclusive=True, group="boot")

        def _write_settled(self, m: Message) -> None:
            color = "#F59E0B" if m.is_agent else "#93c5fd"
            self._chat.write(render_line(m.who, color, m.thinking, m.text))

        def set_status(self, text: str) -> None:
            self._status_widget.update(text)
            self.sub_title = text

        # -- model lifecycle ---------------------------------------------

        def _bootstrap(self) -> None:
            try:
                self.loop = create_loop(self.paths, on_token=self._on_token, on_permission=self._ask_permission)
                if not self.dry_run:
                    llm = (self.loop.cfg.get("llm") or {}) if self.loop else {}
                    model = Path(current_model_path() or str(llm.get("model") or "")).name or "idle"
                    self.call_from_thread(self.set_status, f"ready  ·  {model}")
            except Exception as exc:  # noqa: BLE001
                log.exception("bootstrap failed")
                self.call_from_thread(self.set_status, f"Startup error: {exc}")

        def _load_model(self, spec: str | None) -> None:
            from core.model import apply_choice

            try:
                if spec:
                    choice = choice_from_spec(spec, self.paths.root, read_config(self.paths))
                    if choice is None:
                        self.call_from_thread(self.set_status, f"No model matched {spec!r} — /model to retry")
                        self._bootstrap()
                        return
                    cfg = read_config(self.paths)
                    boot = apply_choice(self.paths, choice, cfg)
                    if not boot.get("ok"):
                        self.call_from_thread(self.set_status, f"Model load failed: {boot.get('error')}")
                self._bootstrap_or_reload()
            except Exception as exc:  # noqa: BLE001
                log.exception("model load failed")
                self.call_from_thread(self.set_status, f"Model load failed: {exc}")

        def _bootstrap_or_reload(self) -> None:
            self.loop = create_loop(
                self.paths, on_token=self._on_token, on_permission=self._ask_permission, previous=self.loop
            )
            llm = self.loop.cfg.get("llm") or {}
            model = Path(current_model_path() or str(llm.get("model") or "")).name or "idle"
            self.call_from_thread(self.set_status, f"ready  ·  {model}")

        # -- streaming (the part this module exists for) ------------------

        def _on_token(self, tok: str) -> None:
            """Called from the worker thread, once per token. No UI hop here —
            just append behind a lock. The timer on the UI thread decides when
            to actually look at this."""
            with self._tok_lock:
                self._pending_tokens.append(tok)

        def _drain_tokens(self) -> None:
            """Runs on the UI thread at a fixed rate. Batches everything that
            arrived since the last tick into one StreamSplitter.feed() call and
            one widget update — see the module docstring for why that's safe."""
            with self._tok_lock:
                if not self._pending_tokens:
                    return
                chunk = "".join(self._pending_tokens)
                self._pending_tokens.clear()
            view = self._splitter.feed(chunk)
            self._render_live(view)

        def _render_live(self, view: StreamView) -> None:
            self._live_widget.update(render_line("s0uRc3", "#F59E0B", view.thinking, view.answer))

        # -- permission -----------------------------------------------------

        def _ask_permission(self, request: dict[str, Any]) -> str:
            self._perm_box = {}
            self._perm_done = threading.Event()
            done = self._perm_done
            box = self._perm_box

            def on_done(decision: str | None) -> None:
                box["v"] = decision or DENY
                done.set()

            self.call_from_thread(lambda: self.push_screen(PermissionModal(request), on_done))
            if not done.wait(timeout=3600):
                return DENY
            return box.get("v") or DENY

        # -- turns ------------------------------------------------------

        def on_input_submitted(self, event: Input.Submitted) -> None:
            text = (event.value or "").strip()
            event.input.value = ""
            if not text:
                return
            if text.startswith("/"):
                self._command(text)
                return
            if self._busy:
                self.set_status("Already running — Esc to stop")
                return
            self._start_turn(text)

        def _start_turn(self, goal: str) -> None:
            self._busy = True
            user_msg = Message(who="You", text=goal, is_agent=False)
            self.messages.append(user_msg)
            self._write_settled(user_msg)
            self._splitter = StreamSplitter()
            self._streaming = True
            self._live_widget.update(render_line("s0uRc3", "#F59E0B", "", ""))
            self._prompt.disabled = True
            self.set_status("Dry-run…" if self.dry_run else "Thinking…")
            self.run_worker(lambda: self._run_turn(goal), thread=True, exclusive=True, group="run")

        def _run_turn(self, goal: str) -> None:
            result: dict[str, Any] = {"ok": False, "error": "run_failed"}
            try:
                if self.loop is None:
                    result = {"ok": False, "error": "Not ready — still starting."}
                else:
                    self.loop.on_token = self._on_token
                    self.loop.on_permission = self._ask_permission
                    try:
                        raw = self.loop.run(goal, dry_run=self.dry_run, max_steps=self.max_steps)
                        result = raw if isinstance(raw, dict) else {"ok": False, "error": str(raw)}
                    except Exception as exc:  # noqa: BLE001
                        log.exception("run failed")
                        result = {"ok": False, "error": str(exc)}
                view = self._splitter.finalize()
                summary = finish_summary(result)
                if summary:
                    view = StreamView(view.thinking, summary, False, "Thought")
                elif result.get("error") == "cancelled":
                    view = StreamView(view.thinking, view.answer or "Generation stopped.", False, "Thought")
                elif not result.get("ok") and result.get("error"):
                    view = StreamView(view.thinking, view.answer or str(result.get("error")), False, "Thought")
                self.call_from_thread(self._finish_turn, view, result)
            except Exception as exc:  # noqa: BLE001
                log.exception("turn worker crashed")
                self.call_from_thread(self._finish_turn, StreamView("", str(exc), False, "Thought"), {"ok": False, "error": str(exc)})

        def _finish_turn(self, view: StreamView, result: dict[str, Any]) -> None:
            self._streaming = False
            with self._tok_lock:
                self._pending_tokens.clear()
            final = Message(who="s0uRc3", text=view.answer or ("Done." if result.get("ok") else str(result.get("error") or "Failed")), thinking=view.thinking, is_agent=True)
            self.messages.append(final)
            self._write_settled(final)
            self._live_widget.update("")
            self._busy = False
            self._prompt.disabled = False
            self._prompt.focus()
            # File I/O off the UI thread — the full TUI does this synchronously
            # in the render path; a OneDrive-synced folder makes that worse.
            self.run_worker(self.save_chat, thread=True, group="save")
            llm = self.loop.cfg.get("llm") or {} if self.loop is not None else {}
            model = Path(current_model_path() or str(llm.get("model") or "")).name or "idle"
            steps = result.get("steps")
            suffix = f"  ·  {steps} steps" if steps is not None else ""
            if self.dry_run:
                self.set_status(f"Idle  ·  dry-run{suffix}")
            elif result.get("ok"):
                self.set_status(f"Idle  ·  {model}{suffix}")
            else:
                self.set_status(f"{result.get('error') or 'error'}{suffix}")

        def save_chat(self) -> None:
            try:
                save_history(self.paths.db, self.messages)
            except OSError as exc:
                log.warning("chat history save failed: %s", exc)

        # -- commands -----------------------------------------------------

        def _command(self, raw: str) -> None:
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/help", "/?"):
                self._chat.write(HELP_TEXT)
            elif cmd in ("/quit", "/exit", "/q"):
                self.exit()
                return
            elif cmd == "/clear":
                self.messages.clear()
                self._chat.clear()
                intro = Message(who="s0uRc3", text="Chat cleared. History wiped for this machine only.", is_agent=True)
                self.messages.append(intro)
                self._write_settled(intro)
                self.set_status("Chat cleared")
            elif cmd == "/dry-run":
                self.dry_run = not self.dry_run
                self.set_status("Dry-run ON" if self.dry_run else "Dry-run OFF")
            elif cmd == "/model":
                self.set_status(f"Loading {arg}…" if arg else "Reloading default model…")
                self.run_worker(lambda: self._load_model(arg or None), thread=True, exclusive=True, group="boot")
            elif cmd == "/reward":
                self._reward()
            else:
                self._chat.write(f"Unknown command {cmd}. /help for the list.")
            self.run_worker(self.save_chat, thread=True, group="save")

        def _reward(self) -> None:
            if self.loop is None:
                self.set_status("Not ready")
                return
            try:
                out = reward_last_run(self.loop.cfg, self.paths.db, note="simple tui /reward", memory=self.loop.memory)
                self._chat.write(str(out.get("message") or out))
                self.set_status("Reward saved" if out.get("ok") else "Reward failed")
            except Exception as exc:  # noqa: BLE001
                self._chat.write(f"Reward error: {exc}")

        # -- cancel ---------------------------------------------------------

        def action_cancel_run(self) -> None:
            if self.loop is not None:
                self.loop.cancel_requested = True
            if self._perm_done is not None:
                self._perm_box["v"] = DENY
                self._perm_done.set()
                try:
                    if isinstance(self.screen, ModalScreen):
                        self.pop_screen()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.workers.cancel_group("run")
            except Exception:  # noqa: BLE001
                pass
            self._busy = False
            self._prompt.disabled = False
            self._prompt.focus()
            if self._streaming:
                self.set_status("Stopped — type to continue")

        def stop_server(self) -> None:
            stop_llama()

    return SimpleApp(AgentPaths.discover(), dry_run=dry_run, max_steps=max_steps, model=model)
