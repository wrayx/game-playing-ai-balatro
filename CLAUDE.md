# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Balatro-playing AI: screen capture → two YOLO11n detectors (entities + UI) → OCR enrichment → LLM reasoning (OpenRouter function calling) → mouse/keyboard automation against the live game window. Nothing is read from game memory or mods; everything is pixels.

`AGENTS.md` and `.cursorrules` carry a near-duplicate of this guidance for other tools — update them together with this file when conventions change.

## Commands

Pixi config lives in `pyproject.toml` under `[tool.pixi.*]` — there is **no `pixi.toml`**.

```bash
pixi install                         # create env from pyproject.toml + pixi.lock
pixi run dev                         # = python cli/run-ai-balatro.py (interactive detection demo)
pixi run test                        # = pytest tests/ -v
pixi run style                       # fmt + ruff-check + lint   <- what CI runs
pixi run quality                     # style + test
pixi run fmt / lint / ruff-check     # ruff format . / ruff check . / ruff check . --fix
pixi run start-benchmark             # OCR engine benchmark on sample dataset images
```

Single test / subset:

```bash
pixi run pytest tests/test_action_module.py -v
pixi run pytest tests/test_ocr_engines.py::test_ocr_engine_smoke -v
pixi run pytest tests/unit -v            # pure-logic tests, no models or game needed
pixi run pytest tests/integration -v     # agent framework; some tests skip without OPENROUTER_API_KEY
```

Anything not wrapped in a task: `pixi run python <script>`, or `pixi shell` first for CLIs (`yolo`, `huggingface-cli`).
Add deps with `pixi add --pypi <pkg>` (PyPI) or `pixi add <pkg>` (conda/system/CUDA); CUDA entries are per-platform (`--platform win-64 --platform linux-64`) because macOS uses MPS.

CI (`.github/workflows/ci.yml`) only runs `pixi run style` — the test job is commented out, so tests are on you locally.

## Before anything will run

`models/` and `data/datasets/` are **git submodules pointing at HuggingFace repos** and are empty on a fresh clone. Without them `MultiYOLODetector` logs "model not found" and silently returns zero detections.

```bash
git lfs install
git submodule update --init      # 3 model repos + 2 dataset repos
```

Runtime prerequisites beyond that: the Balatro game running and visible, macOS permissions (Screen Recording + Accessibility + Automation for the terminal/IDE — see README, restart the app after granting), and an LLM key for any reasoning path — `BALATRO_LLM_API_KEY` (or `ANTHROPIC_API_KEY`) for `AnthropicProvider`, `OPENROUTER_API_KEY` for `OpenRouterProvider`. The project variable is checked first deliberately, so the key need not sit in `ANTHROPIC_API_KEY`, which other Anthropic tooling on the same machine also reads. OCR backends (RapidOCR/PaddleOCR/EasyOCR) download their own weights on first use; Tesseract comes from pixi.

## Architecture

The pipeline is layered, and each layer is usable standalone — that matters because the lower layers work offline while the upper ones need the live game.

**Capture** — `core/screen_capture.py`. `mss` grabs frames; `_detect_balatro_window()` scores visible windows by title keyword and excludes IDE/browser windows (`config.yaml: screen_capture.excluded_apps`) so it doesn't lock onto your editor.

**Detection** — `core/multi_yolo_detector.py` wraps two `YOLODetector` instances:
- `entities` — cards, jokers, packs, tooltips (10 classes, `configs/v2-balatro-entities/dataset.yaml`)
- `ui` — buttons and numeric readouts (33 classes, `configs/v1-balatro-ui/dataset.yaml`)

Class names are read from the dataset submodules' `classes.txt`, not from the configs; the `configs/**/dataset.yaml` files exist for training and mirror those lists. `MultiYOLODetector` resolves model paths from the repo root via `utils/path_utils.resolve_path` and ignores `config/config.yaml`. `config.yaml` + `config/settings.py` only drive the older single-model path (`DetectionService` → `ui/demo_app.py`, what `pixi run dev` uses).

**State extraction** — `services/game_state_extraction.py` is the fast path that produces the dict the LLM sees. It batches: YOLO both models on one frame, OCR the dynamic UI boxes (`services/ui_text_service.py`, RapidOCR over `DYNAMIC_UI_CLASSES` — cash, hands/discards left, ante, round, chips, mult, target score), then optionally sweeps the hand with the mouse so each card's tooltip renders and can be OCR'd (`services/card_tooltip_service.py`, matched back to the card by geometry). Game phase is inferred from which buttons are visible.

UI readouts go through `RapidOCREngine.run_text_line`, **not** `run`. The crops are already localised by the UI detector, and RapidOCR's DB text detector is trained on words and lines — it finds nothing in a crop holding one large isolated glyph, which is exactly what Balatro's single-digit readouts are. Measured on a live frame the detector stage scored 2/10 fields, and skipping it scored 8/10; `_digits_only` closes the last two, since every `DYNAMIC_UI_CLASSES` entry is numeric and the recogniser decorates lone digits (`'-0'`, `'-*0-'`). Card descriptions are multi-word and still need `run`. The English recogniser is opt-in via `RapidOCREngine(rec_lang='en')` and is only used here — it reads pixel digits better but is worse on description crops, so do not make it the default.

**Reasoning** — `ai/` is a four-tier abstraction: `engines/` (transport) → `providers/` (`BaseProvider`/`LLMProvider`/`VLMProvider`, concrete: `providers/anthropic_provider.py` and `providers/openrouter.py`) → `agents/` (`BaseAgent` + `AgentOrchestrator`, concrete: `agents/balatro_agent.py`) with `memory/conversation.py` and `templates/prompt_template.py` alongside. The agent emits OpenAI-style function calls defined in `ai/actions/schemas.py::GAME_ACTIONS`. `create_llm_provider()` picks the provider whose credentials are present, preferring Anthropic, with `BALATRO_LLM_MODEL` overriding the model. The default is `claude-sonnet-5`: measured over real decisions through the production prompt it matched Opus on every case at ~2.5x lower cost, while Haiku played a no-pair hand holding three discards. `AnthropicProvider` translates GAME_ACTIONS to Anthropic tools (`parameters` → `input_schema`) and normalises the reply to the same `{'content', 'function_calls'}` shape OpenRouter returns, so the agent needs no branching. Under `strict` the schemas are deep-copied and sanitised: Anthropic's strict subset rejects numeric bounds, so `{'type': 'integer', 'minimum': 0}` is a 400 — the executor range-checks indices against the real hand anyway. Sampling parameters are never sent; current models reject `temperature`/`top_p`/`top_k`, and depth is steered with `output_config.effort` — which, with adaptive thinking, Haiku also rejects, so both are skipped for that family. The agent builds OpenAI-style history with the system prompt at `messages[0]`; the provider hoists every system turn into the top-level `system` parameter, since Sonnet rejects one inside `messages`. Thinking tokens count against `max_tokens`, which is why `_llm_query` asks for 8000 rather than the 1000 that truncated replies before the tool call arrived.

**Action** — `ai/actions/executor.py` (`ActionExecutor.process()`) dispatches those function calls to `card_action_engine.py` (hand detection, left-to-right index ordering, click sequences), `button_detector.py` (maps UI-model class names → button types via `button_class_map`), and `mouse_controller.py` (eased multi-step mouse movement — instant warps don't register in Balatro; plus window focus via AppleScript on macOS).

**Card classification is centralised in `core/entities.py`** and must stay that way. Every entities-model class name contains the substring `card` — including `joker_card` — so any local substring filter folds jokers, consumables, tooltips and the deck pile into the hand. Measured over 40 dataset frames before the fix: the LLM was shown 110 phantom hand cards, every hand was inflated (told 13 when holding 8), and all 29 joker detections were misfiled, leaving the jokers list permanently empty. The old executor also accepted jokers and consumables as playable, so a selected index could click one. Index misalignment between the prompt and executor was total whenever a tooltip was raised: verified live on a Small Blind hand, 8 of 8 hand cards were misindexed in all 105 captured frames. One visible tooltip fires two detections (`card_description` and `poker_card_description`) and both sort ahead of the leftmost card, shifting the hand by two. The LLM saw its pair of Aces at indices 1 and 3, and those executor indices were A♦ and Q♠ — Ace-high instead of a pair, one hand burned. This was the normal operating state, since `hover_before_action=True` raises tooltips before every decision. Static dataset frames show only 2 of 40 misaligned and are misleading here: training captures have no hover sweep running, so a tooltip never co-occurs with a hand in them. Measure this live, not on the dataset. The prompt builder (`_build_base_state`), the click path (`CardPositionDetector`) and the tooltip matcher all derive their sets from that module. Do not re-filter on class name locally; add to the taxonomy instead. `cli/diagnose-entity-classes.py` re-measures all of this.

Two ways to address cards coexist: the LLM-facing **index** API (`play_cards(indices=[0,1,2])`) and the internal **position array** (`[1,1,1,0]` play / `[-1,-1,0,0]` discard, mixed signs rejected). `CardAction.from_array` is the bridge. `src/ai_balatro/ai/actions/README.md` documents this module in depth (in Chinese).

**Side pipelines** — `ocr/` (engine wrappers + `llm_judge.py` + metrics, feeding the benchmark CLI and `docs/ai/findings/`) and `datasets/` (card-corner crop export, CNN rank/suit classifier, Label Studio import) with matching `cli/` scripts.

### Current gameplay coverage

The action layer implements play / discard / hover / click-button only. There is **no shop, purchase, sell, reroll, or blind-selection logic** — the UI model detects those buttons (`button_purchase`, `button_sell`, `button_store_reroll`, `button_cash_out`, `button_level_select`…) and `click_button` can reach some of them, but no agent reasoning drives them. Money (`ui_data_cash`) is OCR'd and dumped into the prompt as raw UI text; nothing spends it. Jokers are detected as a bare `joker_card` class with no name or effect text, so they reach the prompt as `Joker: joker_card (confidence: …)` and inform nothing. There is no concept of stake or deck anywhere in the codebase — run setup is manual and the agent never adapts to difficulty. Practically: the agent plays the hand phase of a blind and nothing else.

`ButtonDetector` only treats `button_*` classes (and the legacy aliases in `button_class_map`) as clickable. It used to accept any class containing a button word, so `ui_data_discards_left` — the discards-remaining counter — was offered as a discard button and lost to the real one by 0.011 confidence during a live discard; had it won, the action would have clicked a number and reported success.

`click_button` resolves a requested type through `BUTTON_CONFIG`, which maps each type to **exact** UI-model class names. It used to match keywords as substrings, which collided: `shop` hit `button_store_reroll` (a $5 reroll), `play` hit `button_main_menu_play` (starting a new run), and `discard` hit the `ui_data_discards_left` counter. `shop` is gone entirely — Balatro has no shop-entry button, the shop just appears. `ButtonDetector.button_class_map` is derived from the same table rather than hand-maintained; the old copy keyed the sort buttons differently from the config, so those calls always failed. `AGENT_BUTTON_TYPES` is the subset offered to the model: the shop and menu controls exist in the config so code can reach them, but are withheld until there is reasoning behind them.

`GameState` in `ai/llm/base.py` is vestigial — nothing constructs it.

## Conventions

**Imports.** The package is `src/ai_balatro/` (import root is `src/`, `ai_balatro_train/` sits beside it). There is no editable install — every entry point under `cli/` and `examples/` prepends `src/` to `sys.path` before importing, and `tests/__init__.py` does the same. Keep that shim when adding an entry point. Tests use both `from ai_balatro...` and `from src.ai_balatro...`; match the file you're editing.

**Style.** ruff, line length 88, **single quotes**, 4-space indent (`ruff.toml`). `E402` is ignored precisely because of the sys.path shims. Type hints on public functions; keep modules small.

**Language.** The codebase mixes English and Chinese in log messages, docstrings, and error strings (heavily so in `ai/actions/`). Follow the surrounding file rather than normalizing.

**Commits.** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `style:`).

## CV debugging

Save intermediate images — annotated frames, crops, tooltip matches — whenever touching detection or OCR; these bugs are visual and invisible in logs. `CardTooltipService` has `save_debug_images`, `ActionExecutor.execute_from_array(..., show_visualization=True)` previews the click plan before acting. Prototype in `notebooks/` (inline image display), then extract to modules and add tests.

## Design docs

`docs/ai/designs/YYYY-MM-DD-kebab-case-description.md`, YAML frontmatter with `title`, `date`, `coding_agents` (`authors` including human collaborators and Claude Code, `project`, `context`, `technologies`), `tags`. Append an EDIT changelog at the bottom for significant revisions. Benchmark write-ups go in `docs/ai/findings/`.
