
## 🧩 Advanced Configuration

### Memory Settings (.env)
Control how Kira remembers things:
- `MEMORY_INCLUDE_EPISODIC=false` (Default: false) - Set to `true` to include past conversation logs. Keep 'false' for facts-only memory.
- `MEMORY_MAX_AGE_DAYS=7` - Memories older than this are ignored.
- `MEMORY_MAX_ITEMS=5` - Max number of memory facts injected per turn.
- `MEMORY_MAX_TOKENS=800` - Hard limit on memory token budget.

### Logging & Debugging
- Logs are now clean and sanitized by default (no printed prompts!).
- To see formatted prompts for debugging, set `DEBUG_PROMPT_PREVIEW=true` in your environment.
- Logs will show `[REDACTED]` for keys and system prompts.
