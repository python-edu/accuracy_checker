readme_cp:
	cp README.md ./acc/README.md

readme_rm:
	rm ./acc/README.md

docs_cp: readme_cp
	mkdir -p ./acc/docs; \
	cp -r ./docs/* ./acc/docs/

python_build:
	python -m build

copy_wheel:
	latest_whl=$$(ls -t dist/*.whl | head -n 1); \
	cp $$latest_whl -t wheels/

.PHONY: README.md
README.md: readme_source
	latest_whl=$$(ls -t dist/*.whl | head -n 1); \
	latest_name=$$(basename $$latest_whl); \
	sed "s|{{latest.whl}}|$$latest_name|g" readme_source > README.md

build: python_build copy_wheel README.md docs_cp
	@echo "build wykonany"

	
