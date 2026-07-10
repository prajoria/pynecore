"""E0.1 split verification: compiler_errors.py owns compiler+runtime errors;
errors.py owns provider-side errors; both paths still work for one release."""


def test_compiler_errors_module_exports_move_list() -> None:
    from openbb_pine import compiler_errors
    expected = {
        "PineError", "Diagnostic", "PineCompileError", "PineSyntaxError",
        "PineTypeError", "PineUnsupportedBuiltinError",
        "PineUnsupportedFeatureError", "PineCodegenError",
        "PineInternalCompilerError", "PineCacheError",
        "PineRuntimeError", "PineStrategyNotYetImplementedError",
        "PineSecurityError", "PineExecTimeoutError",
        "PineDataResolverError", "PineSecurityContextNotFoundError",
    }
    missing = expected - set(dir(compiler_errors))
    assert not missing, f"compiler_errors missing: {missing}"


def test_errors_module_still_exports_provider_errors() -> None:
    from openbb_pine import errors
    for name in (
        "PineProviderError", "PineFMPRequiredError",
        "PineFMPUnreachableError", "PineDataValidationError",
    ):
        assert hasattr(errors, name), f"errors missing STAY symbol: {name}"


def test_errors_module_still_reexports_compiler_symbols_for_one_release() -> None:
    # Old callers `from openbb_pine.errors import PineSyntaxError` must keep working.
    from openbb_pine import errors, compiler_errors
    assert errors.PineSyntaxError is compiler_errors.PineSyntaxError
    assert errors.PineDataResolverError is compiler_errors.PineDataResolverError
    assert errors.PineSecurityContextNotFoundError is compiler_errors.PineSecurityContextNotFoundError


def test_provider_errors_are_NOT_in_compiler_errors() -> None:
    from openbb_pine import compiler_errors
    for provider_only in ("PineFMPRequiredError", "PineFMPUnreachableError", "PineDataValidationError"):
        assert not hasattr(compiler_errors, provider_only), (
            f"{provider_only} is provider-side, must NOT leak into compiler_errors"
        )


def test_wildcard_import_from_errors_only_yields_stay_symbols() -> None:
    """PR #351 review: ``from openbb_pine.errors import *`` must yield ONLY
    the STAY-list (provider) symbols. MOVE-list symbols remain importable
    via explicit ``from openbb_pine.errors import PineSyntaxError`` for the
    one-release shim window, but wildcard callers get nudged toward the
    new canonical home (``openbb_pine.compiler_errors``)."""
    from openbb_pine import errors
    stay = {
        "PineProviderError", "PineFMPRequiredError",
        "PineFMPUnreachableError", "PineDataValidationError",
    }
    move = {
        "PineError", "Diagnostic", "PineCompileError", "PineSyntaxError",
        "PineTypeError", "PineUnsupportedBuiltinError",
        "PineUnsupportedFeatureError", "PineCodegenError",
        "PineInternalCompilerError", "PineCacheError",
        "PineRuntimeError", "PineStrategyNotYetImplementedError",
        "PineSecurityError", "PineExecTimeoutError",
        "PineDataResolverError", "PineSecurityContextNotFoundError",
    }
    exported = set(errors.__all__)
    assert exported == stay, (
        f"errors.__all__ must expose ONLY STAY symbols; got extras: "
        f"{exported - stay}, missing: {stay - exported}"
    )
    leaked = exported & move
    assert not leaked, f"MOVE symbols leaked into wildcard export: {leaked}"

    # Explicit-attribute access still works for MOVE symbols during the shim window.
    assert errors.PineSyntaxError is not None
    assert errors.Diagnostic is not None
