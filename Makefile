.PHONY: build build-asan test test-live mutation bench clean
build:
	cmake -S mock_server -B mock_server/build -DCMAKE_BUILD_TYPE=Release && cmake --build mock_server/build -j
build-asan:
	cmake -S mock_server -B mock_server/build-asan -DENABLE_SANITIZERS=ON && cmake --build mock_server/build-asan -j
test: build
	cd api_tests && python -m pytest
test-live:
	cd api_tests && IQLAB_TARGET=cerebras python -m pytest
mutation: build
	python mock_server/mutation/run_mutations.py
clean:
	rm -rf mock_server/build mock_server/build-asan api_tests/reports results
