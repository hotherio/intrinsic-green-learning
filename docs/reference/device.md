# Device

::: igl.get_device
    options:
      show_root_heading: true
      show_source: false

## Execution branches

Training selects one backend per fit from the module's device. The backend
owns everything that differs between devices: the readout solve, the Green
kernel path, the optimizer construction, snapshots for early stopping, the
matmul precision context, and the single host transfer per epoch. See
[Device branches](../concepts.md#device-branches-cpu-mps-cuda) for the table.

::: igl.device.select_backend
    options:
      show_root_heading: true
      show_source: false

::: igl.device.Backend
    options:
      show_root_heading: true
      show_source: false
      members: false

::: igl.device.CpuBackend
    options:
      show_root_heading: true
      show_source: false
      members: false

::: igl.device.MpsBackend
    options:
      show_root_heading: true
      show_source: false
      members: false

::: igl.device.CudaBackend
    options:
      show_root_heading: true
      show_source: false
      members: false
