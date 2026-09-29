from __future__ import annotations

from functools import lru_cache, singledispatch
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar

from attrs import Factory, define

from ._compat import TypeAlias
from .fns import Predicate

if TYPE_CHECKING:
    from .converters import BaseConverter

TargetType: TypeAlias = Any
UnstructuredValue: TypeAlias = Any
StructuredValue: TypeAlias = Any

StructureHook: TypeAlias = Callable[[UnstructuredValue, TargetType], StructuredValue]
UnstructureHook: TypeAlias = Callable[[StructuredValue], UnstructuredValue]

Hook = TypeVar("Hook", StructureHook, UnstructureHook)
HookFactory: TypeAlias = Callable[[TargetType], Hook]


@define
class _DispatchNotFound:
    """A dummy object to help signify a dispatch not found."""


@define
class HookRegistry:
    """The registration stage of hook resolution.

    Stores the three kinds of hook registrations, independently of any
    converter:

    * class hooks, either exact (_direct_) or MRO-based (_singledispatch_)
    * predicate hooks, dispatched on a predicate returning true
    * factory hooks, predicates producing hooks via a factory (optionally
      taking a converter)

    Newer predicate and factory registrations take precedence over older ones.

    .. versionadded:: 26.3.0
    """

    _direct_dispatch: dict[TargetType, Callable[..., Any]] = Factory(dict)
    _single_dispatch: Any = Factory(lambda: singledispatch(_DispatchNotFound))
    _handler_pairs: list[tuple[Predicate, Callable[..., Any], bool, bool]] = Factory(
        list
    )

    def register_class(
        self, cls: Any, hook: Callable[..., Any], *, direct: bool = False
    ) -> None:
        """Register a hook for a class, either exactly or through the MRO."""
        if direct:
            self._direct_dispatch[cls] = hook
        else:
            self._single_dispatch.register(cls, hook)

    def register_predicate(
        self, predicate: Predicate, hook: Callable[..., Any]
    ) -> None:
        """Register a hook to be used when the predicate matches."""
        self._handler_pairs.insert(0, (predicate, hook, False, False))

    def register_factory(
        self,
        predicate: Predicate,
        factory: Callable[..., Any],
        *,
        takes_converter: bool = False,
    ) -> None:
        """Register a hook factory to be used when the predicate matches."""
        self._handler_pairs.insert(0, (predicate, factory, True, takes_converter))

    @property
    def class_hooks(self) -> dict[TargetType, Callable[..., Any]]:
        """The registered class hooks, both singledispatch and direct."""
        return {
            **{
                cls: hook
                for cls, hook in self._single_dispatch.registry.items()
                if hook is not _DispatchNotFound
            },
            **self._direct_dispatch,
        }

    @property
    def predicate_hooks(self) -> list[tuple[Predicate, Callable[..., Any]]]:
        """The registered predicate hooks, most recent first."""
        return [
            (predicate, hook)
            for predicate, hook, is_generator, _ in self._handler_pairs
            if not is_generator
        ]

    @property
    def factory_hooks(self) -> list[tuple[Predicate, Callable[..., Any], bool]]:
        """The registered hook factories, most recent first."""
        return [
            (predicate, factory, takes_converter)
            for predicate, factory, is_generator, takes_converter in self._handler_pairs
            if is_generator
        ]

    def clear_direct(self) -> None:
        """Clear the direct (exact) class hooks."""
        self._direct_dispatch.clear()

    def copy_to(self, other: HookRegistry, skip: int = 0) -> None:
        """Copy registrations into another registry.

        The `skip` oldest handler pairs (usually the converter defaults) are
        not copied. Direct class hooks are not copied either; they are a
        product of hook generation and are regenerated on demand.
        """
        other._handler_pairs = self._handler_pairs[:-skip] + other._handler_pairs
        for cls, fn in self._single_dispatch.registry.items():
            other._single_dispatch.register(cls, fn)


def _dispatch_function(
    handler_pairs: list[tuple[Predicate, Callable[..., Any], bool, bool]],
    typ: Any,
    converter: BaseConverter | None,
) -> Callable[..., Any] | None:
    """Return the hook of the first matching handler pair, if any."""
    for can_handle, handler, is_generator, takes_converter in handler_pairs:
        # can handle could raise an exception here
        # such as issubclass being called on an instance.
        # it's easier to just ignore that case.
        try:
            ch = can_handle(typ)
        except Exception:  # noqa: S112
            continue
        if ch:
            if is_generator:
                if takes_converter:
                    return handler(typ, converter)
                return handler(typ)

            return handler
    return None


def select_hook(
    typ: TargetType,
    registry: HookRegistry,
    fallback_factory: HookFactory,
    converter: BaseConverter | None = None,
) -> Any:
    """The strategy selection stage of hook resolution.

    The priority order is:

    1. singledispatch class hooks
    2. direct (exact) class hooks
    3. predicate and factory hooks, most recently registered first
    4. the fallback factory

    No caching happens at this stage.

    :param converter: Only needed to resolve factories registered
        as taking a converter.
    """
    try:
        dispatch = registry._single_dispatch.dispatch(typ)
        if dispatch is not _DispatchNotFound:
            return dispatch
    except Exception:  # noqa: S110
        pass

    direct_dispatch = registry._direct_dispatch.get(typ)
    if direct_dispatch is not None:
        return direct_dispatch

    res = _dispatch_function(registry._handler_pairs, typ, converter)
    return res if res is not None else fallback_factory(typ)


@define(init=False)
class DispatchCache:
    """The caching stage of hook resolution.

    Wraps a selection callable in an unbounded LRU cache, offering explicit
    invalidation and introspection of cache hits.

    .. versionadded:: 26.3.0
    """

    dispatch: Callable[[TargetType], Any]

    def __init__(self, select: Callable[[TargetType], Any]) -> None:
        self.dispatch = lru_cache(maxsize=None)(select)

    def clear(self) -> None:
        """Invalidate all cached hooks."""
        self.dispatch.cache_clear()

    def cache_info(self) -> Any:
        """Return the cache statistics (hits and misses) of the cache."""
        return self.dispatch.cache_info()


@define
class FunctionDispatch:
    """
    FunctionDispatch is similar to functools.singledispatch, but
    instead dispatches based on functions that take the type of the
    first argument in the method, and return True or False.

    objects that help determine dispatch should be instantiated objects.

    :param converter: A converter to be used for factories that require converters.

    ..  versionchanged:: 24.1.0
        Support for factories that require converters, hence this requires a
        converter when creating.
    """

    _converter: BaseConverter
    _handler_pairs: list[tuple[Predicate, Callable[[Any, Any], Any], bool, bool]] = (
        Factory(list)
    )

    def register(
        self,
        predicate: Predicate,
        func: Callable[..., Any],
        is_generator=False,
        takes_converter=False,
    ) -> None:
        self._handler_pairs.insert(0, (predicate, func, is_generator, takes_converter))

    def dispatch(self, typ: Any) -> Callable[..., Any] | None:
        """
        Return the appropriate handler for the object passed.
        """
        return _dispatch_function(self._handler_pairs, typ, self._converter)

    def get_num_fns(self) -> int:
        return len(self._handler_pairs)

    def copy_to(self, other: FunctionDispatch, skip: int = 0) -> None:
        other._handler_pairs = self._handler_pairs[:-skip] + other._handler_pairs


@define(init=False)
class MultiStrategyDispatch(Generic[Hook]):
    """
    MultiStrategyDispatch uses a combination of exact-match dispatch,
    singledispatch, and FunctionDispatch.

    :param fallback_factory: A hook factory to be called when a hook cannot be
        produced.
    :param converter: A converter to be used for factories that require converters.

    .. versionchanged:: 23.2.0
        Fallbacks are now factories.
    .. versionchanged:: 24.1.0
        Support for factories that require converters, hence this requires a
        converter when creating.
    """

    _fallback_factory: HookFactory[Hook]
    _converter: BaseConverter
    _registry: HookRegistry
    _cache: DispatchCache
    dispatch: Callable[[TargetType, BaseConverter], Hook]

    def __init__(
        self, fallback_factory: HookFactory[Hook], converter: BaseConverter
    ) -> None:
        self._fallback_factory = fallback_factory
        self._converter = converter
        self._registry = HookRegistry()
        self._cache = DispatchCache(
            lambda typ: select_hook(typ, self._registry, fallback_factory, converter)
        )
        self.dispatch = self._cache.dispatch

    @property
    def _direct_dispatch(self) -> dict[TargetType, Hook]:
        return self._registry._direct_dispatch

    @property
    def _single_dispatch(self) -> Any:
        return self._registry._single_dispatch

    def dispatch_without_caching(self, typ: TargetType) -> Hook:
        """Dispatch on the type but without caching the result."""
        return select_hook(typ, self._registry, self._fallback_factory, self._converter)

    def register_cls_list(self, cls_and_handler, direct: bool = False) -> None:
        """Register a class to direct or singledispatch."""
        for cls, handler in cls_and_handler:
            self._registry.register_class(cls, handler, direct=direct)
            if not direct:
                self._registry.clear_direct()
        self._cache.clear()

    def register_func_list(
        self,
        pred_and_handler: list[
            tuple[Predicate, Any]
            | tuple[Predicate, Any, bool]
            | tuple[Predicate, Callable[[Any, BaseConverter], Any], Literal["extended"]]
        ],
    ):
        """
        Register a predicate function to determine if the handler
        should be used for the type.

        :param pred_and_handler: The list of predicates and their associated
            handlers. If a handler is registered in `extended` mode, it's a
            factory that requires a converter.
        """
        for tup in pred_and_handler:
            if len(tup) == 2:
                func, handler = tup
                self._registry.register_predicate(func, handler)
            else:
                func, handler, is_gen = tup
                if is_gen == "extended":
                    self._registry.register_factory(func, handler, takes_converter=True)
                else:
                    self._registry.register_factory(func, handler)
        self._registry.clear_direct()
        self._cache.clear()

    def clear_direct(self) -> None:
        """Clear the direct dispatch."""
        self._registry.clear_direct()

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._registry.clear_direct()
        self._cache.clear()

    def get_num_fns(self) -> int:
        return len(self._registry._handler_pairs)

    def copy_to(self, other: MultiStrategyDispatch, skip: int = 0) -> None:
        self._registry.copy_to(other._registry, skip=skip)
        other.clear_cache()
