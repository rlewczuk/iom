# Installing dependencies (Ubuntu)

```
sudo apt install build-essential cmake doctest-dev doxygen clangd lldb-20
```

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Run

```sh
./build/iom
```

# Test

```
cmake --build build --target iom_tests
ctest --test-dir build --output-on-failure
```
