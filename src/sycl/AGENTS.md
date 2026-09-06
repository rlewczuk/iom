# SYCL troubleshooting

If tests report `eligible_device_count() == 0` while `sycl-ls` lists GPUs, the command likely ran in a non-interactive shell without the oneAPI environment. Run through an environment-loaded login/interactive shell (or source the oneAPI setup /opt/intel/oneapi/setvars.sh before executing), then confirm with `sycl-ls`; do not diagnose the backend from the uninitialized shell result.
