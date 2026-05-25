"""Framework-specific exceptions."""


class HarnessError(Exception):
    """Base exception for all harness failures."""


class ConfigurationError(HarnessError):
    """Raised when configuration is invalid or incomplete."""


class BudgetExceededError(HarnessError):
    """Raised when execution exceeds a configured budget."""


class PermissionDeniedError(HarnessError):
    """Raised when an action is blocked by governance policy."""


class ToolExecutionError(HarnessError):
    """Raised when a tool fails during execution."""


class SkillLoadError(HarnessError):
    """Raised when a skill package cannot be loaded."""

