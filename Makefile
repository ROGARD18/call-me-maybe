VENV_NAME := .venv
SRCDIR := src

install:
	@uv sync
	
run:
	@uv run python -m $(SRCDIR)

debug:
	@uv run python -m pdb -m $(SRCDIR)

clean:
	@rm -rf data/output __pycache__ .mypy_cache
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	@echo "  ______   __"
	@echo " /      \ /  |                                                _"
	@echo "/000000  |00 |  ______    ______   _______                   //"
	@echo "00 |  00/ 00 | /      \  /      \ /       \                 //"
	@echo "00 |      00 |/000000  | 000000  |0000000  |               //"
	@echo "00 |   __ 00 |00    00 | /    00 |00 |  00 |              //"
	@echo "00 \__/  |00 |00000000/ /0000000 |00 |  00 |             //"
	@echo "00    00/ 00 |00       |00    00 |00 |  00 |            //"
	@echo " 000000/  00/  0000000/  0000000/ 00/   00/            //"
	@echo "                                              ________//_______"
	@echo "                                             |################|"
	@echo "                                              ################ "
	@echo ""
	@echo "                                         . : .  *    ."
	@echo "                                        . : *. * . : * . ."
	@echo "                                       :: *   ..  : . *"
	@echo "                                      * . *:   *   . * .  ."

lint:
	@flake8 .
	@mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@flake8 .
	@mypy . --strict

.PHONY: install run debug clean lint lint-strict
