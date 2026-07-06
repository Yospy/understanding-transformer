from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import NUMBER_FORMATS
from src.infer import AdditionChatModel


DEFAULT_CHECKPOINT = Path("runs/stage5/digit2-fixed-reversed-answer-weighted/checkpoint_final.pt")

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Addition Transformer Chat</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #171717;
      --muted: #66615a;
      --line: #d8d2c8;
      --paper: #f7f3ec;
      --panel: #fffdf8;
      --accent: #0f766e;
      --accent-dark: #0b4f4a;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      background:
        linear-gradient(90deg, rgba(23,23,23,.035) 1px, transparent 1px),
        linear-gradient(180deg, rgba(23,23,23,.035) 1px, transparent 1px),
        var(--paper);
      background-size: 24px 24px;
      color: var(--ink);
    }
    main {
      width: min(920px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 16px;
    }
    header {
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.1;
      font-weight: 800;
      letter-spacing: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 12px;
      text-align: right;
      line-height: 1.45;
    }
    #chat {
      overflow: auto;
      background: rgba(255,253,248,.74);
      border: 1px solid var(--line);
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
      min-height: 420px;
    }
    .msg {
      max-width: min(680px, 88%);
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 12px 14px;
      line-height: 1.45;
      font-size: 14px;
      white-space: pre-wrap;
    }
    .user {
      align-self: flex-end;
      border-color: #99bcb7;
      background: #eaf7f5;
    }
    .bot {
      align-self: flex-start;
    }
    .error {
      color: var(--bad);
      border-color: #f1b8b3;
      background: #fff5f4;
    }
    form {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
    }
    input, button {
      font: inherit;
      min-height: 46px;
      border: 1px solid var(--line);
    }
    input {
      padding: 0 14px;
      background: var(--panel);
      color: var(--ink);
    }
    button {
      padding: 0 18px;
      background: var(--accent);
      color: white;
      border-color: var(--accent-dark);
      cursor: pointer;
      font-weight: 800;
    }
    button:disabled {
      opacity: .58;
      cursor: wait;
    }
    @media (max-width: 620px) {
      header { align-items: start; flex-direction: column; }
      .meta { text-align: left; }
      form { grid-template-columns: 1fr; }
      .msg { max-width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Addition Transformer Chat</h1>
      <div class="meta">operands 0-99<br>trained arithmetic checkpoint</div>
    </header>
    <section id="chat" aria-live="polite">
      <div class="msg bot">Ask a 2-digit arithmetic question, for example: 80+26, 91-47, 12*8, or 80/7</div>
    </section>
    <form id="form">
      <input id="input" autocomplete="off" autofocus placeholder="37+48, 91-47, 12*8, 80/7">
      <button id="send" type="submit">Ask</button>
    </form>
  </main>
  <script>
    const chat = document.querySelector("#chat");
    const form = document.querySelector("#form");
    const input = document.querySelector("#input");
    const send = document.querySelector("#send");

    function addMessage(text, className) {
      const node = document.createElement("div");
      node.className = `msg ${className}`;
      node.textContent = text;
      chat.appendChild(node);
      chat.scrollTop = chat.scrollHeight;
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      addMessage(message, "user");
      input.value = "";
      send.disabled = true;
      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message})
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "request failed");
        addMessage(`${payload.expression} = ${payload.answer}\\nexpected: ${payload.expected}\\nmodel prompt: ${payload.prompt}\\nraw output: ${payload.raw_answer}`, "bot");
      } catch (error) {
        addMessage(error.message, "bot error");
      } finally {
        send.disabled = false;
        input.focus();
      }
    });
  </script>
</body>
</html>
"""


def prediction_payload(prediction: Any) -> dict[str, Any]:
    return {
        "expression": f"{prediction.a}{prediction.operation}{prediction.b}",
        "answer": prediction.answer,
        "expected": prediction.expected,
        "correct": prediction.correct,
        "prompt": prediction.prompt,
        "raw_answer": prediction.raw_answer,
        "generated": prediction.generated,
    }


def build_handler(chat_model: AdditionChatModel) -> type[BaseHTTPRequestHandler]:
    class ChatHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                message = str(payload.get("message", ""))
                prediction = chat_model.ask(message)
            except Exception as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            self.send_json(HTTPStatus.OK, prediction_payload(prediction))

    return ChatHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a trained addition-transformer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--digit-length", type=int, default=2)
    parser.add_argument("--number-format", choices=NUMBER_FORMATS, default="fixed_reversed")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--once", type=str, default=None, help="answer one expression and exit")
    parser.add_argument("--serve", action="store_true", help="start a local browser chat server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def run_repl(chat_model: AdditionChatModel) -> None:
    print("Addition Transformer Chat. Type q to quit.")
    while True:
        try:
            message = input("> ").strip()
        except EOFError:
            print()
            return
        if message.lower() in {"q", "quit", "exit"}:
            return
        if not message:
            continue
        try:
            prediction = chat_model.ask(message)
            print(f"{prediction.a}+{prediction.b} = {prediction.answer}")
            print(f"model prompt: {prediction.prompt} raw: {prediction.raw_answer}")
        except Exception as error:
            print(f"error: {error}")


def main() -> None:
    args = parse_args()
    chat_model = AdditionChatModel(
        checkpoint_path=args.checkpoint,
        digit_length=args.digit_length,
        number_format=args.number_format,
        device_preference=args.device,
    )

    if args.once is not None:
        prediction = chat_model.ask(args.once)
        print(f"{prediction.a}+{prediction.b} = {prediction.answer}")
        print(f"model_prompt={prediction.prompt} raw_answer={prediction.raw_answer} correct={prediction.correct}")
        return

    if args.serve:
        server = ThreadingHTTPServer((args.host, args.port), build_handler(chat_model))
        print(f"Serving Addition Transformer Chat at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
        finally:
            server.server_close()
        return

    run_repl(chat_model)


if __name__ == "__main__":
    main()
