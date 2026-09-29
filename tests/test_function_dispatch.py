from cattrs import BaseConverter
from cattrs.dispatch import (
    DispatchCache,
    FunctionDispatch,
    HookRegistry,
    MultiStrategyDispatch,
    select_hook,
)


def test_function_dispatch():
    dispatch = FunctionDispatch(BaseConverter())

    assert dispatch.dispatch(float) is None

    test_func = object()

    dispatch.register(lambda cls: issubclass(cls, float), test_func)

    assert dispatch.dispatch(float) == test_func


def test_function_clears_cache_after_function_added():
    dispatch = FunctionDispatch(BaseConverter())

    class Foo:
        pass

    Foo()

    class Bar(Foo):
        pass

    Bar()

    dispatch.register(lambda cls: issubclass(cls, Foo), "foo")
    assert dispatch.dispatch(Bar) == "foo"
    dispatch.register(lambda cls: issubclass(cls, Bar), "bar")
    assert dispatch.dispatch(Bar) == "bar"


def test_function_dispatch_exception():
    """Function dispatch gracefully handles exceptions in predicates."""
    dispatch = FunctionDispatch(BaseConverter())

    def raising_predicate(cls):
        raise ValueError("This predicate raises an error")

    dispatch.register(lambda cls: issubclass(cls, float), "float")
    dispatch.register(raising_predicate, "error")

    assert dispatch.dispatch(float) == "float"


class Foo:
    pass


class Bar(Foo):
    pass


class Baz:
    pass


def test_hook_registry_three_entry_kinds():
    """The registry exposes class, predicate and factory entries separately."""
    registry = HookRegistry()

    registry.register_class(Foo, "cls-hook")
    registry.register_class(Bar, "direct-hook", direct=True)
    registry.register_predicate(lambda t: t is Foo, "pred-hook")
    registry.register_factory(lambda t: t is Bar, lambda t: "generated")
    registry.register_factory(
        lambda t: t is Baz, lambda t, c: "generated-with-conv", takes_converter=True
    )

    class_hooks = registry.class_hooks
    assert class_hooks[Foo] == "cls-hook"
    assert class_hooks[Bar] == "direct-hook"

    assert [hook for _, hook in registry.predicate_hooks] == ["pred-hook"]

    factories = registry.factory_hooks
    assert [takes_conv for _, _, takes_conv in factories] == [True, False]


def test_registry_entries_are_newest_first():
    """Later predicate/factory registrations take precedence over earlier ones."""
    registry = HookRegistry()

    registry.register_predicate(lambda t: True, "first")
    registry.register_predicate(lambda t: True, "second")

    assert [hook for _, hook in registry.predicate_hooks] == ["second", "first"]


def test_select_hook_without_converter():
    """Strategy selection works without a converter for plain entries."""
    registry = HookRegistry()
    registry.register_class(Foo, "cls-hook")
    registry.register_factory(lambda t: t is Baz, lambda t: "generated")
    registry.register_predicate(lambda t: t is Baz, "pred-hook")

    assert select_hook(Foo, registry, lambda t: "fallback") == "cls-hook"
    # The predicate was registered after the factory, so it wins.
    assert select_hook(Baz, registry, lambda t: "fallback") == "pred-hook"
    assert select_hook(int, registry, lambda t: "fallback") == "fallback"


def test_select_hook_strategy_priority():
    """The strategy order is: singledispatch, direct, function dispatch, fallback."""
    registry = HookRegistry()
    registry.register_factory(lambda t: t is Foo, lambda t: "generated")
    registry.register_class(Foo, "direct", direct=True)

    # A direct class hook beats a factory hook.
    assert select_hook(Foo, registry, lambda t: "fallback") == "direct"

    registry.register_class(Foo, "singledispatch")

    # A singledispatch class hook beats a direct one.
    assert select_hook(Foo, registry, lambda t: "fallback") == "singledispatch"


def test_select_hook_class_hook_beats_factory_regardless_of_order():
    """Pin the cross-strategy priority: class hooks always beat factories.

    Registration order only matters within a strategy, not across strategies;
    this is the rule parent/child converter copies rely on.
    """
    registry = HookRegistry()
    registry.register_class(Foo, "cls-hook")
    registry.register_factory(lambda t: t is Foo, lambda t: "generated")

    assert select_hook(Foo, registry, lambda t: "fallback") == "cls-hook"


def test_select_hook_extended_factory_uses_converter():
    """Factories registered with a converter receive it at selection time."""
    registry = HookRegistry()
    registry.register_factory(
        lambda t: t is Foo, lambda t, c: (t, c), takes_converter=True
    )

    converter = BaseConverter()
    assert select_hook(Foo, registry, lambda t: "fallback", converter) == (
        Foo,
        converter,
    )


def test_select_hook_tolerates_raising_predicates():
    """A raising predicate is skipped during selection."""
    registry = HookRegistry()

    def raising_predicate(cls):
        raise ValueError("boom")

    registry.register_predicate(lambda t: t is Foo, "pred-hook")
    registry.register_predicate(raising_predicate, "unreachable")

    assert select_hook(Foo, registry, lambda t: "fallback") == "pred-hook"


def test_dispatch_cache_hits_and_clear():
    """Cache hits are observable, and `clear` forces reselection."""
    calls = []

    def select(typ):
        calls.append(typ)
        return f"hook-{len(calls)}"

    cache = DispatchCache(select)

    assert cache.dispatch(Foo) == "hook-1"
    assert cache.dispatch(Foo) == "hook-1"

    info = cache.cache_info()
    assert (info.hits, info.misses) == (1, 1)

    cache.clear()

    assert cache.cache_info().hits == 0
    assert cache.dispatch(Foo) == "hook-2"


def _make_dispatch():
    return MultiStrategyDispatch(lambda t: "fallback", BaseConverter())


def test_multi_strategy_dispatch_register_invalidates_cache():
    """Registering any kind of hook invalidates previously cached dispatches."""
    dispatch = _make_dispatch()

    assert dispatch.dispatch(Foo) == "fallback"
    assert dispatch.dispatch.cache_info().misses == 1

    dispatch.register_cls_list([(Foo, "cls-hook")])
    assert dispatch.dispatch(Foo) == "cls-hook"

    dispatch.register_func_list([(lambda t: t is Baz, "pred-hook")])
    # Class hooks still win over predicate hooks.
    assert dispatch.dispatch(Foo) == "cls-hook"
    assert dispatch.dispatch(Baz) == "pred-hook"

    dispatch.register_func_list([(lambda t: t is Baz, lambda t: "generated", True)])
    # The newer factory registration beats the older predicate hook.
    assert dispatch.dispatch(Baz) == "generated"


def test_multi_strategy_dispatch_clear_cache():
    """`clear_cache` clears both the direct dispatch and the LRU cache."""
    dispatch = _make_dispatch()

    dispatch.register_cls_list([(Foo, "direct-hook")], direct=True)
    assert dispatch.dispatch(Foo) == "direct-hook"
    assert dispatch.dispatch.cache_info().misses == 1

    dispatch.clear_cache()

    assert dispatch.dispatch.cache_info().misses == 0
    assert dispatch.dispatch(Foo) == "fallback"


def test_multi_strategy_dispatch_register_cls_clears_direct():
    """Registering a singledispatch class hook clears generated direct hooks."""
    dispatch = _make_dispatch()

    dispatch.register_cls_list([(Foo, "direct-hook")], direct=True)
    assert dispatch.dispatch(Foo) == "direct-hook"

    dispatch.register_cls_list([(Bar, "cls-hook")])

    assert dispatch.dispatch(Foo) == "fallback"
    assert dispatch.dispatch(Bar) == "cls-hook"


def test_multi_strategy_dispatch_copy_to():
    """`copy_to` copies custom registrations, but not direct hooks or caches."""
    source = _make_dispatch()
    source.register_func_list([(lambda t: t is Foo, "pred-hook")])
    source.register_cls_list([(Bar, "cls-hook")])
    source.register_cls_list([(Baz, "direct-hook")], direct=True)
    source.dispatch(Foo)

    target = _make_dispatch()
    source.copy_to(target, skip=0)

    # Singledispatch class hooks are copied; direct hooks are not
    # (they are regenerated on demand).
    assert target.dispatch(Bar) == "cls-hook"
    assert target.dispatch(Baz) == "fallback"
    # `skip=0` copies no predicate/factory pairs (a `[:-0]` slice is empty).
    assert target.get_num_fns() == 0
    assert target.dispatch(Foo) == "fallback"


def test_multi_strategy_dispatch_copy_to_skip():
    """`copy_to` copies custom pairs, skipping the oldest (default) ones."""
    source = _make_dispatch()
    source.register_func_list([(lambda t: t is Foo, "default-ish")])
    num_defaults = source.get_num_fns()
    source.register_func_list([(lambda t: t is Bar, "custom")])

    target = _make_dispatch()
    source.copy_to(target, skip=num_defaults)

    assert target.dispatch(Bar) == "custom"
    assert target.dispatch(Foo) == "fallback"
