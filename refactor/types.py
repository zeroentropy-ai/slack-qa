# Slack QA types:

from pydantic import BaseModel
from typing import Any, overload
from abc import ABC
from utils import map_parallel

class ErrorReport(BaseModel):
    message: str

class Document(BaseModel):
    id: str
    content: str
    metadata: dict[str, Any]

# A question asked by user
class UserQuery(BaseModel):
    id: str
    query: str
    target_document_id: str # Document.id*

# Output of a model. A model here predicts slack api calls for a UserQuery
class SearchStrings(BaseModel):
    query_id: str # Query.id
    search_strings: list[str]
    generated_by: str # model name, e.g. gpt-5-mini, llama, llama-finetuned-oct19

# result of executing a slack-like search api call
class SearchResult(BaseModel):
    query_id: str
    search_string: str
    results: list[Document] | ErrorReport
    source: str # "local" | "slack"

class ResultsAccumulator(BaseModel):
    query: UserQuery
    search_strings: SearchStrings
    search_results: list[SearchResult] | None # as many as there are search strings

class KeywordGenModel(BaseModel, ABC):
    model_name: str
    async def generate_search_strings(self, _s: str) -> list[str] | ErrorReport:
        return ErrorReport(message="Not implemented")

@overload
async def run_model(m: KeywordGenModel, q: str) -> list[str] | ErrorReport: ...
@overload
async def run_model(m: KeywordGenModel, q: UserQuery) -> SearchStrings | ErrorReport: ...
@overload
async def run_model(m: KeywordGenModel, q: ResultsAccumulator) -> ResultsAccumulator | ErrorReport: ...

async def run_model(m: KeywordGenModel, q: str | UserQuery | ResultsAccumulator):
    if isinstance(q, UserQuery):
        result = await run_model(m, q.query)
        if isinstance(result, ErrorReport):
            return result
        return SearchStrings(query_id=q.id, search_strings=result, generated_by=m.model_name)
    elif isinstance(q, ResultsAccumulator):
        result = await run_model(m, q.query)
        if isinstance(result, ErrorReport):
            return result
        else:
            return ResultsAccumulator(query=q.query, search_strings=result, search_results=q.search_results)
    else: # str
        return await m.generate_search_strings(q)


class SearchAPI(BaseModel, ABC):
    source_name: str

    async def run_search(self, _s: str) -> list[Document] | ErrorReport:
        return ErrorReport(message="Not implemented")

    async def parallel_search(self, searches: list[str]) -> list[list[Document] | ErrorReport]:
        return await map_parallel(self.run_search, searches)

@overload
async def run_search(m: SearchAPI, q: str) -> list[Document] | ErrorReport: ...
@overload
async def run_search(m: SearchAPI, q: SearchStrings) -> list[SearchResult]: ...

async def run_search(m: SearchAPI, q: str | SearchStrings):
    if isinstance(q, str):
        return await m.run_search(q)
    else:
        results = await m.parallel_search(q.search_strings)

        return [SearchResult(query_id=q.query_id, search_string=s, results=rs, source=m.source_name) for (s, rs) in zip(q.search_strings, results)]
