import sys


def test_pytest_anda():
    assert True


def test_estoy_en_el_venv():
    # sys.prefix apunta a la carpeta del entorno activo.
    # Si corrés pytest con el Python global, este test falla.
    assert sys.prefix != sys.base_prefix
