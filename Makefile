test-uv:
	uv run pytest .
	make clean
test:
	pytest .
	make clean

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	