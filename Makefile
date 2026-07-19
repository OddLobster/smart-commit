.PHONY: install reinstall uninstall

install:
	uv tool install .

reinstall:
	uv tool install --reinstall .

uninstall:
	uv tool uninstall smart-commit
