# hbmlens

Read HBM failures like an inspection image.

hbmlens injects faults into a virtual high-bandwidth memory, runs memory test
patterns on that virtual device or on a real GPU (CUDA), keeps an exact fail log,
and analyzes where and why cells fail.

Status: early development (v0.1-dev). See [docs/roadmap.md](docs/roadmap.md).

## Why hbmlens

Measured, not claimed ([details](docs/vs-free-tools.md)):

- **Better tests**: the hbmlens suite detects 100% of nine classic fault models with
  64 memory operations per word; a cuda_memtest-style suite needs 289 and still misses
  a class of coupling faults ([coverage table](docs/coverage.md)).
- **Every failure kept**: exact counts and full records, not the last 10.
- **Fast**: 609-828 GB/s test throughput on a laptop RTX 5080.
- **Failures are read**: DQ lane / row / column / bank / cell signatures, fail
  bitmaps and a 3D view of the device.

```bash
pip install -e ".[cuda]"
hbmlens demo                      # inject faults, test, analyze, write a 3D viewer
hbmlens run --backend cuda        # test real GPU memory
hbmlens coverage                  # which pattern catches which fault
```

```python
from hbmlens.mapping import get_mapper
from hbmlens.faults import demo_faults
from hbmlens.backends.virtual import VirtualBackend
from hbmlens.patterns.library import get_pattern

mapper = get_mapper("interleaved", "tiny")
log = VirtualBackend("tiny", demo_faults(mapper)).run(get_pattern("march-c-minus"))
print(log.meta.total_fails, "failing reads")
```

GPU backend: `pip install -e ".[cuda]"`, then `hbmlens.backends.cuda.CudaBackend`.

Everything is based on public information. hbmlens is not affiliated with any
memory or GPU vendor, and injected (synthetic) faults are always labeled as such.

## 한국어 요약

가상 HBM에 불량을 주입하고, 메모리 테스트 패턴을 가상 장치와 실제 GPU(CUDA)에서
똑같이 돌린 뒤, 불량을 전량 기록해 위치와 원인을 판독하는 오픈소스 도구입니다.
공개 자료만 사용하며 특정 회사와 무관합니다.

License: Apache-2.0
