# hbmlens

Read HBM failures like an inspection image.

hbmlens injects faults into a virtual high-bandwidth memory, runs memory test
patterns on that virtual device or on a real GPU (CUDA), keeps an exact fail log,
and analyzes where and why cells fail.

Status: early development (v0.1-dev). See [docs/roadmap.md](docs/roadmap.md).

## Why hbmlens

Measured, not claimed ([details](docs/vs-free-tools.md)):

- **Test science you can check**: faults are fault primitives `<S/F/R>` from the memory
  test literature (static, dynamic, and DRAM-specific partial / dirty / soft / transient
  faults), and `hbmlens fp-coverage` proves exhaustively which ones a march test detects.
  It reproduces the published results it is built on (all 64 values of the dynamic-fault
  coverage table of VTS 2002, the first-detection table of DATE 2006, the completeness of
  March SS, RAW1 and RAW) and found that hammered writes let plain write-destructive
  faults escape March H1C and T1C, and March H2C with an odd hammer count, with fixes
  proven complete ([fault-primitives.md](docs/fault-primitives.md)).
- **Better tests**: across 17 fault models from the memory test literature, the
  hbmlens suite (March SS + intra-word + retention) detects 100% with 51 memory
  operations per word; a cuda_memtest-style suite uses 289 and averages 82.5%, with
  0% on deceptive read destructive faults ([coverage tables](docs/coverage.md)).
- **Every access tests DRAM**: repeated accesses to a word inside one GPU pass are
  served from cache; hbmlens runs those elements op-major so each op reaches DRAM.
- **Every failure kept**: exact counts and full records, not the last 10.
- **Fast**: a whole suite takes 0.30 s of memory passes per 4 GiB on a laptop
  RTX 5080 (730-775 GB/s), plus two 64 s retention pauses.
- **Failures are read**: DQ lane / row / column / bank / cell signatures, fail
  bitmaps and a 3D view of the device.

```bash
pip install -e ".[cuda]"
hbmlens demo                      # inject faults, test, analyze, write a 3D viewer
hbmlens run --backend cuda --geometry medium --pattern march-ss   # test 4 GiB of GPU memory
hbmlens coverage                  # which pattern catches which fault
hbmlens fp-coverage               # prove march tests against fault primitives
```

```python
from hbmlens.fp import STATIC_SINGLE, STATIC_TWO
from hbmlens.fpsim import evaluate
from hbmlens.patterns.march import parse_march

test = parse_march("{⇕(w0); ⇑(r0,w1); ⇑(r1,w0); ⇓(r0,w1); ⇓(r1,w0); ⇕(r0)}", "march-c-minus")
missed = [r.fp.name for r in evaluate(test, STATIC_SINGLE + STATIC_TWO) if not r.detected]
```

Design and quality bar: [docs/design.md](docs/design.md).

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
