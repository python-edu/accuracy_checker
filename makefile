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

latest_name := $(notdir $(shell ls -t dist/*.whl | head -n 1))

update_readme:
	@echo "latest_name: $(latest_name)"
	@cp readme_source README.md
	@sed -i "s|{{latest.whl}}|$(latest_name)|" \
	README.md

build: update_readme readme_cp docs_cp python_build copy_wheel
	@echo "build wykonany"

	
