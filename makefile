# ======================================================
# Cel: `wykonać python -m build`
# Uwaga:
# 1. README.md:
#  - wymaga aktualizacji - wstawienia aktualnej nazwy pliku `.whl`, ale plik nie
#    został jeszcze zbudowany - przed build
#  - dla działania pomocy skryptu i app.py nie musi być aktualny dlatego jest
#    dodawany do pakietu
#  - natomiast plik używany w repo na githubie jest aktualizowany po `build`
#
#  Steps:
#  1. Kopiuje dokumentację do pakietu `acc` - docs i README.md
#  2. Kopiuje dane przykładowe `example/data` do `acc`
#  3. Wykonuje `build` - nowy plik `whl`
#  4. Aktualizuje README.md w głównym katalogu projektu
#  5. Clean: usuwa docs i example/data
#  =====================================================

build: docs_cp data_cp python_build copy_wheel README.md docs_rm data_rm
	@echo "build wykonany"

docs_cp:
	cp README.md ./acc/README.md; \
	mkdir -p ./acc/docs; \
	cp -r ./docs/* ./acc/docs/

docs_rm:
	rm ./acc/README.md; \
	rm -rf ./acc/docs/

data_cp:
	mkdir -p ./acc/example/data; \
	cp -r ./example/data/* ./acc/example/data/

data_rm:
	rm -rf ./acc/example/

python_build:
	python -m build

copy_wheel:
	latest_whl=$$(ls -t dist/*.whl | head -n 1); \
	cp $$latest_whl wheels/

.PHONY: README.md
README.md:
	latest_whl=$$(ls -t dist/*.whl | head -n 1); \
	latest_name=$$(basename $$latest_whl); \
	sed "s|{{latest.whl}}|$$latest_name|g" readme_source > README.md

