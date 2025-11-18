readme_cp:
	cp README.md ./acc/README.md

readme_rm:
	rm ./acc/README.md

docs_cp:
	mkdir -p ./acc/docs
	cp -r ./docs/* ./acc/docs/

python_build:
	python -m build

copy_wheel:
	latest="$$(ls -t dist/*.whl | head -n 1)"; \
	cp "$$latest" wheels/; \
	cp "$$latest" wheels/accuracy-latest-0.0.0-py3-none-any.whl

build: readme_cp docs_cp python_build copy_wheel
	@echo "build wykonany"
