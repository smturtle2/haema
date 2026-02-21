# HAEMA

[English](README.md) | [한국어](README.ko.md)

HAEMA는 ChromaDB 기반 에이전트 메모리 프레임워크입니다.

세 가지 메모리 모드를 하나의 API로 다룹니다.

- `core memory`: 핵심 고정 정보 (`get_core`)
- `latest memory`: 최신 메모리 조회 (`get_latest`)
- `long-term memory`: 의미 기반 검색 (`search`)

쓰기 API는 `add(contents)` 하나만 사용하며, 호출 시 모든 메모리 레이어가 자동 갱신됩니다.

## 설치

```bash
pip install haema
```

개발 환경:

```bash
pip install -e ".[dev]"
```

## 빠른 시작

```python
from haema import Memory

m = Memory(
    path="./haema_store",       # 저장 루트 디렉토리
    output_dimensionality=1536,
    embedding_client=...,       # EmbeddingClient 구현체
    llm_client=...,             # LLMClient 구현체
)

m.add([
    "사용자는 간결하고 실행 가능한 답변을 선호한다.",
    "사용자는 ChromaDB 기반으로 HAEMA를 개발 중이다.",
])

print(m.get_core())
print(m.get_latest(begin=1, count=5))
print(m.search("사용자 선호", 3))
```

실제 Google GenAI 연동 예제:

- `examples/google_genai_example.py`

## 핵심 API

- `Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=5, merge_distance_cutoff=0.25)`
- `get_core() -> str`
- `get_latest(begin: int, count: int) -> list[str]`
- `search(content: str, n: int) -> list[str]`
- `add(contents: list[str]) -> None`

## 클라이언트 인터페이스

- `EmbeddingClient.embed(texts, output_dimensionality) -> np.ndarray`
  - 반환: 2D `numpy.ndarray`, `float32`, shape `(len(texts), output_dimensionality)`
- `LLMClient.generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`
  - 반환: `response_model`로 파싱 가능한 dict

## 저장 구조

`path="./haema_store"`일 때:

- 장기 메모리 DB: `./haema_store/db`
- 코어 메모리: `./haema_store/core.md`

메타데이터:

- `timestamp` (UTC ISO8601)
- `timestamp_ms` (Unix epoch milliseconds)

## 문서

- `docs/index.md`
- `docs/usage.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/release.md`

기본 문서는 영어(`README.md`)입니다.
