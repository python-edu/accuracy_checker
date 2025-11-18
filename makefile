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
	cp "$$(ls -t dist/*.whl | head -n 1)" wheels/

build: readme_cp docs_cp python_build copy_wheel
	@echo "build wykonany"
