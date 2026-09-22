"""Anthropic (Claude) LLM provider using the official SDK."""

import copy
import os
from typing import Any, Dict, List, Optional

import anthropic

from ..engines.api_provider_engine import APIProviderEngine
from ..llm.base import ProcessingResult
from .base import LLMProvider, ProviderConfig, ProviderType
from ...utils.logger import get_logger

logger = get_logger(__name__)

#: Environment variables searched for a key, in order. BALATRO_LLM_API_KEY comes
#: first so the project's key does not have to live in ANTHROPIC_API_KEY, which
#: other Anthropic tooling on the same machine also reads.
API_KEY_ENV_VARS = ('BALATRO_LLM_API_KEY', 'ANTHROPIC_API_KEY')

#: Sonnet rather than Opus: measured over real Balatro decisions through the
#: production prompt, Sonnet matched Opus on every case at roughly 2.5x lower
#: cost and lower latency. Haiku is cheaper still but played a no-pair hand
#: while holding three discards, wasting one of the four hands that decide a
#: run. Override with BALATRO_LLM_MODEL when a harder task warrants it.
DEFAULT_MODEL = 'claude-sonnet-5'

#: Model families that reject BOTH `thinking: {'type': 'adaptive'}` and
#: `output_config.effort` with a 400. Haiku still takes the older fixed-budget
#: thinking form, which this provider does not send. Verified against the API,
#: not inferred -- extend it when a 400 says so.
_NO_MODERN_CONTROLS_PREFIXES = ('claude-haiku',)

#: Current models reject temperature/top_p/top_k outright, so sampling controls
#: are never sent. Thinking depth is steered with output_config.effort instead.
DEFAULT_MAX_TOKENS = 16000


#: Numeric-range keywords that Anthropic's strict schema subset rejects. Dropping
#: them costs nothing here: the executor range-checks every index against the
#: actual hand before clicking, which is the only bound that reflects reality.
_STRICT_UNSUPPORTED_KEYWORDS = (
    'minimum',
    'maximum',
    'exclusiveMinimum',
    'exclusiveMaximum',
    'multipleOf',
)


def _strip_unsupported(schema: Any) -> Any:
    """Recursively drop schema keywords strict mode rejects."""
    if isinstance(schema, dict):
        return {
            key: _strip_unsupported(value)
            for key, value in schema.items()
            if key not in _STRICT_UNSUPPORTED_KEYWORDS
        }
    if isinstance(schema, list):
        return [_strip_unsupported(item) for item in schema]
    return schema


def _close_objects(schema: Any) -> Any:
    """Recursively require additionalProperties: false, as strict mode expects."""
    if isinstance(schema, dict):
        result = {key: _close_objects(value) for key, value in schema.items()}
        if result.get('type') == 'object':
            result.setdefault('additionalProperties', False)
        return result
    if isinstance(schema, list):
        return [_close_objects(item) for item in schema]
    return schema


def _to_anthropic_tool(function: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    """Convert one OpenAI-style function schema to an Anthropic tool.

    GAME_ACTIONS is written in the OpenAI shape ('parameters'); Anthropic calls
    the same thing 'input_schema'. Both are plain JSON Schema underneath, but
    strict mode accepts a narrower subset. Deep-copied because GAME_ACTIONS is a
    shared module constant the OpenRouter provider also reads.
    """
    schema = copy.deepcopy(
        function.get('parameters') or {'type': 'object', 'properties': {}}
    )

    if strict:
        schema = _close_objects(_strip_unsupported(schema))

    tool: Dict[str, Any] = {
        'name': function['name'],
        'description': function.get('description', ''),
        'input_schema': schema,
    }

    if strict:
        tool['strict'] = True

    return tool


class AnthropicProvider(LLMProvider):
    """Claude provider backed by the official anthropic SDK."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        effort: Optional[str] = 'high',
        thinking: bool = True,
        strict_tools: bool = True,
        **kwargs,
    ):
        """
        Initialize the Anthropic provider.

        Args:
            model_name: Claude model id, e.g. 'claude-opus-5'
            api_key: API key; read from BALATRO_LLM_API_KEY or ANTHROPIC_API_KEY
                when None
            timeout: Request timeout in seconds
            max_retries: Retries the SDK performs for 429/5xx/connection errors
            effort: output_config effort - low/medium/high/xhigh/max, or None
            thinking: Send adaptive thinking. Balatro decisions are strategic
                enough to benefit, and it is on by default for current models.
            strict_tools: Ask the API to guarantee schema-valid tool arguments,
                so a malformed index list cannot reach the executor
        """
        config = ProviderConfig(
            provider_type=ProviderType.LLM,
            model_name=model_name,
            api_key=api_key or _key_from_env(),
            base_url='https://api.anthropic.com',
            **kwargs,
        )

        engine = APIProviderEngine(
            name='Anthropic', timeout=timeout, max_retries=max_retries
        )

        super().__init__('Anthropic', config, engine)

        self.timeout = timeout
        self.max_retries = max_retries
        self.effort = effort
        self.thinking = thinking
        self.strict_tools = strict_tools
        self.client: Optional[anthropic.Anthropic] = None

        if not self.config.api_key:
            logger.error(
                'No API key found. Set one of: %s', ', '.join(API_KEY_ENV_VARS)
            )

    def initialize(self) -> bool:
        """Create the SDK client."""
        if not self.config.api_key:
            logger.error('Cannot initialize: no Anthropic API key provided')
            return False

        try:
            self.client = anthropic.Anthropic(
                api_key=self.config.api_key,
                timeout=float(self.timeout),
                max_retries=self.max_retries,
            )
            self.is_initialized = True
            logger.info(f'Anthropic provider initialized with {self.config.model_name}')
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f'Failed to initialize Anthropic provider: {e}')
            return False

    def generate_text(
        self, prompt: str, context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate a text response."""
        return self._create_message(prompt, None, context)

    def function_call(
        self, prompt: str, functions: List[Dict], context: Optional[Dict] = None
    ) -> ProcessingResult:
        """Generate a response that may contain tool calls."""
        tools = [_to_anthropic_tool(f, self.strict_tools) for f in functions]
        return self._create_message(prompt, tools, context)

    def _create_message(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]],
        context: Optional[Dict],
    ) -> ProcessingResult:
        """Issue one Messages API request and normalise the response."""
        if not self.is_initialized or self.client is None:
            return ProcessingResult(
                success=False, data=None, errors=['Provider not initialized']
            )

        context = context or {}

        # Anthropic takes the system prompt in a top-level parameter, and Sonnet
        # rejects a system message inside `messages` outright. The agent builds
        # OpenAI-style history with the system prompt at messages[0], so hoist
        # every system turn out of the history and into `system`.
        messages: List[Dict[str, Any]] = []
        system_parts: List[str] = []

        for message in context.get('history') or []:
            content = message.get('content')
            if message.get('role') == 'system':
                if isinstance(content, str) and content.strip():
                    system_parts.append(content)
            else:
                messages.append(message)

        messages.append({'role': 'user', 'content': prompt})

        request: Dict[str, Any] = {
            'model': self.config.model_name,
            'max_tokens': context.get('max_tokens')
            or getattr(self.config, 'max_tokens', None)
            or DEFAULT_MAX_TOKENS,
            'messages': messages,
        }

        system_message = context.get('system_message')
        if system_message:
            system_parts.append(system_message)

        if system_parts:
            request['system'] = '\n\n'.join(system_parts)

        modern_controls = self._supports_modern_controls()

        if self.thinking and modern_controls:
            request['thinking'] = {'type': 'adaptive'}

        if self.effort and modern_controls:
            request['output_config'] = {'effort': self.effort}

        if tools:
            request['tools'] = tools

        try:
            response = self.client.messages.create(**request)
        except anthropic.NotFoundError as e:
            return self._failure(f'Unknown model {self.config.model_name!r}: {e}')
        except anthropic.AuthenticationError as e:
            return self._failure(f'Authentication failed: {e}')
        except anthropic.RateLimitError as e:
            return self._failure(f'Rate limited: {e}')
        except anthropic.APIStatusError as e:
            return self._failure(f'API error {e.status_code}: {e}')
        except anthropic.APIConnectionError as e:
            return self._failure(f'Connection error: {e}')

        return self._normalise(response)

    def _supports_modern_controls(self) -> bool:
        """Whether this model accepts adaptive thinking and the effort knob."""
        model = (self.config.model_name or '').lower()
        return not model.startswith(_NO_MODERN_CONTROLS_PREFIXES)

    @staticmethod
    def _failure(message: str) -> ProcessingResult:
        logger.error(message)
        return ProcessingResult(success=False, data=None, errors=[message])

    def _normalise(self, response: Any) -> ProcessingResult:
        """Shape an SDK response like the OpenRouter provider's result.

        Consumers read data['content'] and data['function_calls']; keeping the
        two providers interchangeable means the agent needs no branching.
        """
        text_parts: List[str] = []
        function_calls: List[Dict[str, Any]] = []

        for block in response.content:
            if block.type == 'text':
                text_parts.append(block.text)
            elif block.type == 'tool_use':
                function_calls.append(
                    {
                        'id': block.id,
                        'name': block.name,
                        # The SDK has already parsed this into a dict; never
                        # string-match the serialised form.
                        'arguments': dict(block.input),
                    }
                )

        if response.stop_reason == 'refusal':
            details = getattr(response, 'stop_details', None)
            return self._failure(
                f'Model declined the request '
                f'(category: {getattr(details, "category", "unknown")})'
            )

        usage = getattr(response, 'usage', None)

        return ProcessingResult(
            success=True,
            data={
                'content': '\n'.join(text_parts).strip(),
                'function_calls': function_calls,
                'model': getattr(response, 'model', self.config.model_name),
                'usage': {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0),
                }
                if usage
                else {},
                'finish_reason': response.stop_reason or '',
            },
        )

    def get_available_models(self) -> List[str]:
        """List models the account can use, newest first."""
        if not self.is_initialized or self.client is None:
            return [DEFAULT_MODEL]
        try:
            return [model.id for model in self.client.models.list()]
        except Exception as e:  # noqa: BLE001
            logger.warning(f'Could not list models: {e}')
            return [DEFAULT_MODEL]

    def shutdown(self) -> None:
        """Release the client."""
        self.client = None
        self.is_initialized = False


def _key_from_env() -> Optional[str]:
    """First API key found in the supported environment variables."""
    for name in API_KEY_ENV_VARS:
        value = os.getenv(name)
        if value:
            return value
    return None
