# HAEMA

[English](README.md) | [한국어](README.ko.md)

HAEMA는 ChromaDB 기반 에이전트 메모리 프레임워크입니다.

하나의 쓰기 API(`add(contents)`)로 3개 메모리 레이어를 자동 갱신합니다.

- `core memory`: 장기 핵심 정보 (`get_core`)
- `latest memory`: 최신순 조회 (`get_latest`)
- `long-term memory`: 의미 기반 검색 (`search`)

## 현재 설계 핵심

- `add(contents)`는 호출당 1회 N:M 재구성을 수행합니다.
- 임베딩 인터페이스는 query/document로 분리됩니다.
  - `embed_query(...)`
  - `embed_document(...)`
- no-related 전용 경로를 제거하고 단일 재구성 경로로 통합했습니다.
- 재구성 스키마:
  - `memories: list[str]`
  - `coverage: "complete" | "incomplete"`

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
    path="./haema_store",
    output_dimensionality=1536,
    embedding_client=...,   # EmbeddingClient 구현체
    llm_client=...,         # LLMClient 구현체
    merge_top_k=3,
    merge_distance_cutoff=0.25,
)

m.add([
    "사용자는 간결하고 실행 가능한 답변을 선호한다.",
    "사용자는 ChromaDB 기반으로 HAEMA를 개발 중이다.",
])

print(m.get_core())
print(m.get_latest(begin=1, count=5))
print(m.search("사용자 선호", 3))
```

Google GenAI 예제:

- `examples/google_genai_example.py`

## 공개 API

- `Memory(path, output_dimensionality, embedding_client, llm_client, merge_top_k=3, merge_distance_cutoff=0.25)`
- `get_core() -> str`
- `get_latest(begin: int, count: int) -> list[str]`
- `search(content: str, n: int) -> list[str]`
- `add(contents: str | list[str]) -> None`

## 클라이언트 인터페이스

### `EmbeddingClient`

- `embed_query(texts, output_dimensionality) -> np.ndarray`
- `embed_document(texts, output_dimensionality) -> np.ndarray`

반환 형식:

- 2D `numpy.ndarray`
- dtype `float32`
- shape `(len(texts), output_dimensionality)`

### `LLMClient`

- `generate_structured(system_prompt, user_prompt, response_model) -> dict[str, Any]`

전달된 Pydantic 모델로 파싱 가능한 dict를 반환해야 합니다.

## 재구성 스키마

```python
class MemoryReconstructionResponse(BaseModel):
    memories: list[str]
    coverage: Literal["complete", "incomplete"]
```

`memories`가 비었거나 `coverage == "incomplete"`이면 1회 보강 재시도를 수행합니다.  
그래도 실패하면 정규화된 `contents`를 안전 폴백으로 저장합니다.

## 프롬프트 계약 (레이어 책임 분리)

HAEMA는 3단계 프롬프트를 독립적으로 사용하며, 각 단계의 출력 책임이 다릅니다.

- pre-memory split:
  - 입력: 단일 raw add 문자열
  - 출력 스키마: `PreMemorySplitResponse(contents)`
  - 책임: 사실 단위 분해만 수행 (core 정책 판단 금지)
- reconstruction:
  - 입력: 관련 기억 + 신규 contents
  - 출력 스키마: `MemoryReconstructionResponse(memories, coverage)`
  - 책임: long-term 기억 재구성만 수행
- core update:
  - 입력: 현재 core + 재구성된 신규 기억
  - 출력 스키마: `CoreUpdateResponse(should_update, core_markdown)`
  - 책임: 보수적인 core 갱신 판단만 수행

프롬프트 입력 경계는 다음과 같은 태그로 표시합니다.

- `<raw_input> ... </raw_input>`
- `<related_memories> ... </related_memories>`
- `<new_contents> ... </new_contents>`
- `<current_core_markdown> ... </current_core_markdown>`
- `<candidate_new_memories> ... </candidate_new_memories>`

이 태그는 파싱/런타임 제어용이 아니라, 모델이 입력 경계를 정확히 인지하도록 돕는 표기입니다.

## Core 메모리 정책

core에는 장기적이고 영향도 높고 신뢰도 높은 정보만 남겨야 합니다.
프롬프트 정책상 다음 기준을 모두 만족해야 core 후보가 됩니다.

1. 장기성 (여러 세션에서 재사용 가능)
2. 행동 영향도 (향후 응답/결정에 실질적 변화 유발)
3. 신뢰도 (근거가 분명한 고신뢰 정보)

또한 core 프롬프트는 다음을 요구합니다.

- `SOUL/TOOLS/RULE/USER` 중 정확히 한 섹션으로 라우팅
- 세션성/임시성/로그성 저신호 정보 제외
- 전체를 고신호로 압축하고 soft budget 기준 약 8개 bullet 내 유지

## 저장 구조

`path="./haema_store"` 기준:

- 장기 메모리 벡터 DB: `./haema_store/db`
- 코어 메모리: `./haema_store/core.md`
- 최신 인덱스 DB: `./haema_store/latest.sqlite3`

메타데이터:

- `timestamp` (UTC ISO8601)
- `timestamp_ms` (Unix epoch milliseconds)

## add() 동작

1. 입력 문자열 정규화
   - `contents`가 단일 `str`이면 구조화 LLM 출력으로 pre-memory 항목 여러 개로 먼저 확장
2. 전체 contents를 query 임베딩
3. 각 query별 top-k 조회 후 cutoff 이하만 채택
4. 관련 기억을 ID 기준으로 union
5. 관련 기억 + 전체 신규 contents로 재구성 1회 수행
6. 결과를 document 임베딩 후 upsert
7. upsert 성공 후 기존 related IDs delete
8. add 호출당 core 업데이트 1회

## Breaking Changes

이전 버전 대비:

1. `EmbeddingClient.embed(...)` 제거
2. `NoRelatedMemoryResponse` 제거
3. `MemorySynthesisResponse(update: list[str])` 제거 후 `MemoryReconstructionResponse` 도입
4. `merge_top_k` 기본값 `5 -> 3`

## 문서

- `docs/index.md`
- `docs/usage.md`
- `docs/api.md`
- `docs/architecture.md`
- `docs/release.md`
