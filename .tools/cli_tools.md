# CLI Tools Reference

**Status**: Production-Ready | **Version**: v1.0.0

---

## Invocation Pattern

All parallel workers use round-robin dispatch across configured `cli_tools`:

```
timeout --foreground --signal=KILL {configured_timeout_seconds} {tool_invocation} <shell-quoted-prompt>
```

- `configured_timeout_seconds`: orchestrator-supplied timeout such as `WORKER_TIMEOUT_SECONDS`; use a positive integer when enabled
- Round-robin: worker_i uses `cli_tools[i % len(cli_tools)]`

---

## Tools

| Tool | Provider | Model | Invocation |
|------|----------|-------|------------|
| `crush` | MiniMax | MiniMax-M2.7-highspeed | `crush run <prompt>` |
| `opencode` | Volce | doubao-seed-2.0-pro | `opencode run <prompt>` |
| `qwen` | Aliyun | qwen3-max | `qwen --yolo --prompt <prompt>` |
| `goose` | Baidu | deepseek-v3.2 | `goose run --quiet --text <prompt>` |
| `kimi` | Moonshot | kimi-k2.5 | `kimi --quiet --thinking --prompt <prompt>` |
| `claude` | Aliyun | glm-5 | `claude --dangerously-skip-permissions --print <prompt>` |

---

## Usage Notes

- All tools accept a shell-quoted prompt string as final argument
- `--quiet` / `--yolo` / `--dangerously-skip-permissions` suppress interactive prompts
- `--thinking` enables chain-of-thought reasoning (kimi)
- Round-robin dispatch minimizes LLM backend prejudice and reduces hallucination via multi-source consensus
