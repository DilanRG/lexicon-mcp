"""Schema-2 pack planning."""

from __future__ import annotations

import pytest

from lexicon_mcp.pipeline.packs import (
    MIB,
    LanguageSize,
    PlannedPack,
    plan_capability_packs,
    plan_lexical_packs,
)

# 160 compressed bytes per term is the calibrated estimator, so 5 MiB lands at
# ~32,768 terms.
THRESHOLD_TERMS = 5 * MIB // 160


def sizes(**languages: int) -> list[LanguageSize]:
    return [LanguageSize(language, terms) for language, terms in languages.items()]


def test_heavy_languages_get_their_own_pack() -> None:
    packs = plan_lexical_packs(sizes(en=1_985_802, fr=1_548_392))

    assert [pack.id for pack in packs] == ["lexical-en", "lexical-fr"]
    assert all(not pack.bundled for pack in packs)
    assert packs[0].languages == ("en",)


def test_languages_below_the_threshold_are_bundled() -> None:
    packs = plan_lexical_packs(sizes(en=1_985_802, cy=1_000, gv=800, mt=600))

    assert [pack.id for pack in packs] == ["lexical-en", "lexical-bundle-001"]
    assert packs[1].languages == ("cy", "gv", "mt")
    assert packs[1].bundled is True


def test_bundles_close_once_they_reach_the_target() -> None:
    # bytes_per_term=1 makes terms and estimated bytes the same number, so the
    # boundary is exact: 4 MiB each is below the 5 MiB individual threshold, and
    # ten of them reach the 40 MiB bundle target precisely.
    packs = plan_lexical_packs(
        [LanguageSize(f"l{index:02d}", 4 * MIB) for index in range(21)],
        bytes_per_term=1,
    )

    assert [pack.id for pack in packs] == [
        "lexical-bundle-001",
        "lexical-bundle-002",
        "lexical-bundle-003",
    ]
    assert [len(pack.languages) for pack in packs] == [10, 10, 1]


def test_the_long_tail_needs_no_special_case() -> None:
    """4,508 tiny languages collapse into the final bundle on their own."""

    packs = plan_lexical_packs(
        [LanguageSize("en", 1_985_802)]
        + [LanguageSize(f"t{index:04d}", 12) for index in range(4508)]
    )

    assert [pack.id for pack in packs] == ["lexical-en", "lexical-bundle-001"]
    assert len(packs[1].languages) == 4508


def test_a_language_exactly_on_the_threshold_gets_its_own_pack() -> None:
    packs = plan_lexical_packs(sizes(aa=THRESHOLD_TERMS, bb=THRESHOLD_TERMS - 1))

    assert [pack.id for pack in packs] == ["lexical-aa", "lexical-bundle-001"]


def test_planning_is_independent_of_input_order() -> None:
    forward = plan_lexical_packs(sizes(en=900_000, fr=800_000, cy=10, gv=20))
    reverse = plan_lexical_packs(sizes(gv=20, cy=10, fr=800_000, en=900_000))

    assert forward == reverse


def test_every_language_lands_in_exactly_one_pack() -> None:
    requested = sizes(en=900_000, fr=800_000, de=40_000, cy=900, gv=20, mt=5)

    packs = plan_lexical_packs(requested)

    placed = [language for pack in packs for language in pack.languages]
    assert sorted(placed) == sorted(item.language for item in requested)
    assert len(placed) == len(set(placed))


def test_duplicate_and_negative_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate language"):
        plan_lexical_packs([LanguageSize("en", 10), LanguageSize("en", 20)])

    with pytest.raises(ValueError, match="negative term count"):
        plan_lexical_packs([LanguageSize("en", -1)])

    with pytest.raises(ValueError, match="must be positive"):
        plan_lexical_packs(sizes(en=10), bundle_target=0)


def test_empty_input_plans_no_packs() -> None:
    assert plan_lexical_packs([]) == ()


def test_narrow_capabilities_get_one_pack_per_language() -> None:
    packs = plan_capability_packs("semantic", ["fr", "en", "en"])

    assert packs == (
        PlannedPack("semantic-en", "semantic", ("en",), 0),
        PlannedPack("semantic-fr", "semantic", ("fr",), 0),
    )
