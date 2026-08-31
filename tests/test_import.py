def test_import_package():
    import hale_core

    assert hale_core.__version__ == "0.1.0"
