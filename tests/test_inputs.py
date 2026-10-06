"""The Energy substep's inputs are unchanged by the input builder, and work
without the graphical interface."""

from pathlib import Path

import pytest

from . import input_harness as harness

INPUTS = Path(__file__).parent / "data" / "inputs"


@pytest.fixture
def db():
    from molsystem import SystemDB

    db = SystemDB(filename="file:vasp_inputs?mode=memory&cache=shared")
    yield db
    db.close()


@pytest.mark.parametrize("case", sorted(harness.CASES) + sorted(harness.OPTIMIZATION))
def test_inputs_are_byte_identical(case, db, tmp_path):
    """INCAR, POTCAR, KPOINTS and POSCAR as vasp_step wrote them before the
    builder was factored out (tests/data/inputs)."""
    files = harness.generate(case, db, tmp_path)
    for name, text in files.items():
        expected = (INPUTS / case / name).read_text()
        assert text == expected, f"{case}/{name} changed"


def test_no_potentials_chosen_uses_the_defaults(db, tmp_path):
    """Headless, the "potentials" parameter is empty: the set's defaults are
    used (this raised a KeyError before)."""
    import vasp_step

    system, configuration = harness.lif(db)
    energy = vasp_step.Energy()
    energy._id = (1, 1)
    pass
    energy.get_system_configuration = lambda *a, **k: (system, configuration)
    from types import SimpleNamespace

    energy.parent = SimpleNamespace(
        potential_metadata=harness.potential_metadata(), get_value=lambda v: v
    )
    P = energy.parameters.current_values_to_dict(context={})
    assert P["potentials"] == {}
    text = energy.get_POTCAR(P)
    assert "PAW_PBE F" in text and "PAW_PBE Li_sv" in text
    assert text.index("PAW_PBE F") < text.index("PAW_PBE Li_sv")


def test_hard_variant():
    from vasp_step.potentials import potentials_for

    hard = potentials_for("potpaw_PBE.64", ["O", "H", "Li"], variant="hard")
    assert hard == {"O": "O_h", "H": "H_h", "Li": "Li_sv"}
    chosen = potentials_for(
        "potpaw_PBE.64", ["O", "H"], chosen={"O": "O_s"}, variant="hard"
    )
    assert chosen == {"O": "O_s", "H": "H_h"}
    with pytest.raises(ValueError, match="No potential for Xx"):
        potentials_for("potpaw_PBE.64", ["Xx"])
