"""A single access-check request used by the hidden verifier matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccessRequest:
    user_id: str
    action: str
    resource_id: str
    context: tuple[tuple[str, str], ...] = ()

    def context_dict(self) -> dict[str, str]:
        return dict(self.context)

    def as_visible(self) -> dict[str, object]:
        return {
            "user": self.user_id,
            "action": self.action,
            "resource": self.resource_id,
            "context": self.context_dict(),
        }
