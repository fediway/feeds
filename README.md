# feeds

A generic recommendation and ranking engine for building algorithmic feeds.

## Install

```
pip install fediway-feeds
```

## Core Concepts

- **Feed** — Abstract base class. Subclass it, define your sources and processing logic.
- **Pipeline** — Fluent builder for chaining steps: source → rank → sample → paginate.
- **Source** — Where candidates come from (database, Redis, API, etc.).
- **Ranker** — Scores candidates using features.
- **Heuristic** — Adjusts scores post-ranking (e.g. diversity).
- **Candidate / CandidateList** — The items flowing through the pipeline.

## Usage

### Feed (subclass approach)

```python
from feeds import Feed, Source, CandidateList

class MyFeed(Feed):
    entity = "post_id"

    def sources(self):
        return {
            "main": [(my_source, 100)],
            "_fallback": [(fallback_source, 50)],
        }

    async def process(self, candidates: CandidateList) -> CandidateList:
        candidates = self.unique(candidates)
        candidates = await self.rank(candidates, my_ranker)
        return self.sample(candidates, n=20)

feed = MyFeed()
results = await feed.execute(limit=10)
```

### Pipeline (builder approach)

```python
from feeds import Pipeline

results = await (
    Pipeline()
    .select("post_id")
    .source(my_source, n=100)
    .rank(my_ranker)
    .sample(20)
    .paginate(limit=10)
    .execute()
)
```
