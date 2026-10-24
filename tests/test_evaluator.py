from __future__ import annotations

import pytest
from pydantic import ValidationError

from gatekeeper.evaluator import Decision, Evaluator
from gatekeeper.models import Group, Policy, Resource, Statement, User, World


def make_statement(
    statement_id: str,
    effect: str,
    *,
    principals: list[str],
    actions: list[str],
    resources: list[str],
    condition: dict[str, str] | None = None,
    exceptions: list[str] | None = None,
) -> Statement:
    return Statement(
        id=statement_id,
        effect=effect,
        principals=principals,
        actions=actions,
        resources=resources,
        condition=condition,
        exceptions=exceptions or [],
    )


def make_policy(*statements: Statement) -> Policy:
    return Policy(statements=list(statements))


def make_world() -> World:
    return World(
        users={user_id: User(id=user_id) for user_id in ("u1", "u2", "u3", "u4")},
        groups={
            "eng": Group(id="eng", members=["user:u1", "user:u2"]),
            "leads": Group(id="leads", members=["user:u1"]),
            "sales": Group(id="sales", members=["user:u3"]),
            "staff": Group(id="staff", members=["group:eng", "group:sales"]),
            "everyone": Group(id="everyone", members=["group:staff", "user:u4"]),
        },
        resources={
            "r1": Resource(id="r1", tags=["finance", "secret"]),
            "r2": Resource(id="r2", tags=["eng", "public"]),
            "r3": Resource(id="r3", tags=["hr", "secret"]),
            "r4": Resource(id="r4", tags=["finance", "public"]),
        },
    )


@pytest.fixture()
def evaluator() -> Evaluator:
    return Evaluator(make_world())


def test_default_deny(evaluator: Evaluator) -> None:
    policy = make_policy()
    assert evaluator.decide(policy, "u1", "read", "r1") == Decision("deny", None)


def test_exact_user_and_resource_allow(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["user:u1"],
            actions=["read"],
            resources=["resource:r1"],
        )
    )
    assert evaluator.decide(policy, "u1", "read", "r1") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u2", "read", "r1") == Decision("deny", None)
    assert evaluator.decide(policy, "u1", "read", "r2") == Decision("deny", None)
    assert evaluator.decide(policy, "u1", "write", "r1") == Decision("deny", None)


def test_group_and_nested_group_membership(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:eng"],
            actions=["read"],
            resources=["resource:r2"],
        )
    )
    assert evaluator.decide(policy, "u1", "read", "r2") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u2", "read", "r2") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u3", "read", "r2") == Decision("deny", None)

    nested = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:everyone"],
            actions=["write"],
            resources=["resource:r3"],
        )
    )
    for user_id in ("u1", "u2", "u3", "u4"):
        assert evaluator.decide(nested, user_id, "write", "r3") == Decision(
            "allow", "s1"
        )


def test_effective_groups_are_transitive(evaluator: Evaluator) -> None:
    assert evaluator.effective_groups("u2") == frozenset(
        {"eng", "staff", "everyone"}
    )
    assert evaluator.effective_groups("u3") == frozenset(
        {"sales", "staff", "everyone"}
    )
    assert evaluator.effective_groups("u4") == frozenset({"everyone"})


def test_explicit_deny_beats_allow(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:everyone"],
            actions=["read"],
            resources=["tag:public"],
        ),
        make_statement(
            "s2",
            "deny",
            principals=["user:u2"],
            actions=["read"],
            resources=["resource:r2"],
        ),
    )
    assert evaluator.decide(policy, "u2", "read", "r2") == Decision("deny", "s2")
    assert evaluator.decide(policy, "u1", "read", "r2") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u2", "read", "r4") == Decision("allow", "s1")


def test_first_deny_by_statement_id_is_deciding(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s2",
            "deny",
            principals=["user:u2"],
            actions=["delete"],
            resources=["resource:r1"],
        ),
        make_statement(
            "s1",
            "deny",
            principals=["group:everyone"],
            actions=["delete"],
            resources=["tag:secret"],
        ),
    )
    assert evaluator.decide(policy, "u2", "delete", "r1") == Decision("deny", "s1")


def test_conditions_all_must_hold(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:eng"],
            actions=["read"],
            resources=["resource:r2"],
            condition={"network": "corp", "mfa": "true"},
        )
    )
    assert evaluator.decide(
        policy, "u1", "read", "r2", {"network": "corp", "mfa": "true"}
    ) == Decision("allow", "s1")
    assert evaluator.decide(policy, "u1", "read", "r2", {"network": "corp"}) == (
        Decision("deny", None)
    )
    assert evaluator.decide(
        policy, "u1", "read", "r2", {"network": "corp", "mfa": "false"}
    ) == Decision("deny", None)
    assert evaluator.decide(
        policy, "u1", "read", "r2", {"network": "vpn", "mfa": "true"}
    ) == Decision("deny", None)


def test_empty_condition_always_holds(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["user:u4"],
            actions=["read"],
            resources=["resource:r1"],
            condition={},
        )
    )
    assert evaluator.decide(policy, "u4", "read", "r1") == Decision("allow", "s1")


def test_user_exception_skips_a_deny(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:everyone"],
            actions=["delete"],
            resources=["tag:secret"],
        ),
        make_statement(
            "s2",
            "deny",
            principals=["group:everyone"],
            actions=["delete"],
            resources=["tag:secret"],
            exceptions=["user:u1"],
        ),
    )
    assert evaluator.decide(policy, "u1", "delete", "r1") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u2", "delete", "r1") == Decision("deny", "s2")
    assert evaluator.decide(policy, "u4", "delete", "r3") == Decision("deny", "s2")


def test_group_exception_skips_a_deny(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:everyone"],
            actions=["delete"],
            resources=["tag:secret"],
        ),
        make_statement(
            "s2",
            "deny",
            principals=["group:everyone"],
            actions=["delete"],
            resources=["tag:secret"],
            exceptions=["group:eng"],
        ),
    )
    assert evaluator.decide(policy, "u1", "delete", "r1") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u2", "delete", "r1") == Decision("allow", "s1")
    assert evaluator.decide(policy, "u3", "delete", "r1") == Decision("deny", "s2")
    assert evaluator.decide(policy, "u4", "delete", "r1") == Decision("deny", "s2")


def test_wildcard_principal_action_and_resource(evaluator: Evaluator) -> None:
    allow_all = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["*"],
            actions=["*"],
            resources=["*"],
        )
    )
    for user_id in ("u1", "u2", "u3", "u4"):
        for resource_id in ("r1", "r2", "r3", "r4"):
            assert evaluator.decide(allow_all, user_id, "share", resource_id) == (
                Decision("allow", "s1")
            )

    tagged = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["user:u3"],
            actions=["share"],
            resources=["tag:public"],
        )
    )
    assert evaluator.decide(tagged, "u3", "share", "r2") == Decision("allow", "s1")
    assert evaluator.decide(tagged, "u3", "share", "r4") == Decision("allow", "s1")
    assert evaluator.decide(tagged, "u3", "share", "r1") == Decision("deny", None)


def test_policy_order_does_not_change_decisions(evaluator: Evaluator) -> None:
    allow = make_statement(
        "s1",
        "allow",
        principals=["group:everyone"],
        actions=["read"],
        resources=["resource:r2"],
    )
    deny = make_statement(
        "s2",
        "deny",
        principals=["user:u3"],
        actions=["read"],
        resources=["resource:r2"],
    )
    forward = evaluator.compile(make_policy(allow, deny))
    reverse = evaluator.compile(make_policy(deny, allow))
    for user_id in ("u1", "u2", "u3", "u4"):
        assert evaluator.decide(forward, user_id, "read", "r2") == evaluator.decide(
            reverse, user_id, "read", "r2"
        )


def test_policy_and_compiled_policy_agree(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["user:u1"],
            actions=["read"],
            resources=["resource:r1"],
        )
    )
    compiled = evaluator.compile(policy)
    assert evaluator.decide(policy, "u1", "read", "r1") == evaluator.decide(
        compiled, "u1", "read", "r1"
    )


def test_unknown_request_components_raise(evaluator: Evaluator) -> None:
    policy = make_policy()
    with pytest.raises(KeyError, match="unknown user"):
        evaluator.decide(policy, "nobody", "read", "r1")
    with pytest.raises(KeyError, match="unknown resource"):
        evaluator.decide(policy, "u1", "read", "missing")
    with pytest.raises(ValueError, match="unknown action"):
        evaluator.decide(policy, "u1", "admin", "r1")


def test_duplicate_statement_ids_are_rejected() -> None:
    statement = make_statement(
        "s1",
        "allow",
        principals=["user:u1"],
        actions=["read"],
        resources=["resource:r1"],
    )
    with pytest.raises(ValidationError, match="statement ids must be unique"):
        Policy(statements=[statement, statement])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("principals", ["u1"]),
        ("principals", ["role:eng"]),
        ("resources", ["r1"]),
        ("resources", ["dept:finance"]),
        ("actions", ["read", "admin"]),
    ],
)
def test_invalid_selectors_are_rejected(field: str, value: list[str]) -> None:
    kwargs: dict[str, object] = {
        "principals": ["user:u1"],
        "actions": ["read"],
        "resources": ["resource:r1"],
    }
    kwargs[field] = value
    with pytest.raises(ValidationError):
        Statement(id="s1", effect="allow", **kwargs)


def test_invalid_condition_is_rejected() -> None:
    with pytest.raises(ValidationError, match="condition keys must not be empty"):
        Statement(
            id="s1",
            effect="allow",
            principals=["user:u1"],
            actions=["read"],
            resources=["resource:r1"],
            condition={"": "corp"},
        )


def test_group_membership_wildcard_is_rejected() -> None:
    with pytest.raises(ValidationError, match="wildcard"):
        Group(id="eng", members=["*"])


def test_dangling_group_members_are_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown user"):
        World(
            users={"u1": User(id="u1")},
            groups={"eng": Group(id="eng", members=["user:u2"])},
            resources={"r1": Resource(id="r1")},
        )
    with pytest.raises(ValidationError, match="unknown group"):
        World(
            users={"u1": User(id="u1")},
            groups={"eng": Group(id="eng", members=["group:sales"])},
            resources={"r1": Resource(id="r1")},
        )


def test_empty_policy_is_deny_all(evaluator: Evaluator) -> None:
    policy = make_policy()
    compiled = evaluator.compile(policy)
    assert compiled.statement_count == 0
    assert all(
        evaluator.decide(compiled, user_id, action, resource_id) == Decision("deny", None)
        for user_id in ("u1", "u2", "u3", "u4")
        for action in ("read", "write", "delete", "share")
        for resource_id in ("r1", "r2", "r3", "r4")
    )


def test_extra_context_fields_do_not_break_conditions(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["user:u1"],
            actions=["read"],
            resources=["resource:r1"],
            condition={"network": "corp"},
        )
    )
    assert evaluator.decide(
        policy, "u1", "read", "r1", {"network": "corp", "mfa": "false"}
    ) == Decision("allow", "s1")


def test_unknown_referenced_objects_never_match(evaluator: Evaluator) -> None:
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:missing"],
            actions=["read"],
            resources=["resource:missing"],
        )
    )
    assert evaluator.decide(policy, "u1", "read", "r1") == Decision("deny", None)


def test_group_membership_cycles_are_safe() -> None:
    world = World(
        users={"u1": User(id="u1")},
        groups={
            "a": Group(id="a", members=["user:u1", "group:b"]),
            "b": Group(id="b", members=["group:a"]),
        },
        resources={"r1": Resource(id="r1")},
    )
    evaluator = Evaluator(world)
    assert evaluator.effective_groups("u1") == frozenset({"a", "b"})
    policy = make_policy(
        make_statement(
            "s1",
            "allow",
            principals=["group:b"],
            actions=["read"],
            resources=["resource:r1"],
        )
    )
    assert evaluator.decide(policy, "u1", "read", "r1") == Decision("allow", "s1")


def test_unknown_world_action_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown world action"):
        World(
            users={"u1": User(id="u1")},
            resources={"r1": Resource(id="r1")},
            actions=("read", "admin"),
        )
