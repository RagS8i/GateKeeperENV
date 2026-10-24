"""OpenAI function-tool schemas exposed to the model."""

from __future__ import annotations

from typing import Any

_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_requirements",
            "description": "Return the natural-language access requirements for this task.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_policies",
            "description": "List a page of the current access-policy statements, including statement ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "1-based page number.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Inspect a user, group, resource, tag or action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["user", "group", "resource", "tag", "action"],
                    },
                    "id": {
                        "type": "string",
                        "description": "Identifier, with or without its selector prefix.",
                    },
                },
                "required": ["kind", "id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate",
            "description": "Evaluate one access request against the current policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {
                        "type": "string",
                        "description": "User id, optionally prefixed with user:.",
                    },
                    "action": {"type": "string"},
                    "resource": {
                        "type": "string",
                        "description": "Resource id, optionally prefixed with resource:.",
                    },
                    "context": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Request context, for example {network, mfa}.",
                    },
                },
                "required": ["principal", "action", "resource"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "put_policy",
            "description": "Add a new statement or replace an existing statement with the same id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "object",
                        "description": "A complete statement object.",
                        "properties": {
                            "id": {"type": "string"},
                            "effect": {"type": "string", "enum": ["allow", "deny"]},
                            "principals": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "actions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "resources": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "condition": {
                                "type": ["object", "null"],
                                "additionalProperties": {"type": "string"},
                            },
                            "exceptions": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": [
                            "id",
                            "effect",
                            "principals",
                            "actions",
                            "resources",
                        ],
                    }
                },
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_policy",
            "description": "Delete one statement by id.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo",
            "description": "Undo the most recent policy edit.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff",
            "description": "Show changes from the initial broken policy.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_examples",
            "description": "Check the current policy against the visible examples.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Declare the policy repaired and finish the episode.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_SPECS: tuple[dict[str, Any], ...] = tuple(_TOOLS)
TOOLS = TOOL_SPECS
TOOL_NAMES: frozenset[str] = frozenset(
    spec["function"]["name"] for spec in TOOL_SPECS
)
SPECS: dict[str, dict[str, Any]] = {
    spec["function"]["name"]: spec["function"]["parameters"]
    for spec in TOOL_SPECS
}
