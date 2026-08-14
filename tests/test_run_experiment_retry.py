from unittest.mock import Mock

import pytest

from evaluation.runners import run_experiment
from intent_construction.intent_extraction.core.llm_utils import (
    LLMAccountingError,
    LLMBudgetExceeded,
    LLMIncompleteResponse,
)


@pytest.mark.parametrize(
    "error_type",
    [LLMAccountingError, LLMBudgetExceeded, LLMIncompleteResponse],
)
def test_accounting_errors_are_not_retried_or_delayed(monkeypatch, error_type):
    api_call = Mock(side_effect=error_type("stop immediately"))
    sleep = Mock()
    monkeypatch.setattr(run_experiment, "generate_multi_turn", api_call)
    monkeypatch.setattr(run_experiment.time, "sleep", sleep)

    with pytest.raises(error_type, match="stop immediately"):
        run_experiment.call_with_retry(
            messages=[{"role": "user", "content": "test"}],
            model="offline-test-model",
            max_retries=15,
        )

    api_call.assert_called_once()
    sleep.assert_not_called()
