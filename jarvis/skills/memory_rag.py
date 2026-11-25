"""Long-term semantic vector memory and local document RAG skills."""

from __future__ import annotations

from jarvis.core.safety import RiskLevel
from jarvis.skills.registry import SkillContext, SkillRegistry, SkillResult


def register_memory_skills(registry: SkillRegistry) -> None:
    @registry.register(
        name="remember_fact",
        description="Store a personal preference, fact, or piece of information in J.A.R.V.I.S.'s permanent semantic vector memory.",
        category="memory",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The information to remember permanently."}
            },
            "required": ["fact"],
        },
        examples=["remember that my favorite IDE is VS Code", "remember my meeting pin is 4821"],
    )
    def remember_fact(ctx: SkillContext, fact: str) -> SkillResult:
        cleaned = fact.strip()
        if not cleaned:
            return SkillResult(ok=False, message="Please tell me what you would like me to remember.")
        item = ctx.vector_store.add_memory(content=cleaned, namespace="fact", source="voice_memory")
        return SkillResult(
            ok=True,
            message=f"Stored in long-term semantic memory: '{cleaned}'.",
            data=item,
        )

    @registry.register(
        name="recall_memory",
        description="Semantically search J.A.R.V.I.S.'s long-term vector memory and indexed local knowledge base.",
        category="memory",
        risk=RiskLevel.SAFE,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or keywords to search in memory / RAG."},
                "namespace": {
                    "type": "string",
                    "enum": ["all", "fact", "rag"],
                    "description": "Search user facts, indexed knowledge docs (rag), or both (all).",
                },
                "top_k": {"type": "integer", "description": "Number of matches to retrieve (1-8)."},
            },
            "required": ["query"],
        },
        examples=["what do you remember about my IDE", "search knowledge base for wake words"],
    )
    def recall_memory(
        ctx: SkillContext,
        query: str,
        namespace: str = "all",
        top_k: int = 4,
    ) -> SkillResult:
        ns = None if namespace == "all" else namespace
        matches = ctx.vector_store.search(query=query, top_k=max(1, min(8, int(top_k))), namespace=ns)
        if not matches:
            return SkillResult(
                ok=True,
                message=f"I couldn't find any stored memories or knowledge matching '{query}'.",
                data={"matches": []},
            )
        top = matches[0]
        snippets = " | ".join(f"[{m['source']}] {m['content'][:220]}" for m in matches[:3])
        return SkillResult(
            ok=True,
            message=f"Top memory match ({top['source']}, relevance {top['score']:.2f}): {top['content'][:360]}",
            data={"matches": matches, "combined_context": snippets},
        )

    @registry.register(
        name="index_knowledge_base",
        description="Re-index all markdown and text documents in the local `knowledge/` folder for RAG retrieval.",
        category="memory",
        risk=RiskLevel.SAFE,
        examples=["index my knowledge base", "refresh rag documents"],
    )
    def index_knowledge_base(ctx: SkillContext) -> SkillResult:
        summary = ctx.vector_store.ingest_directory(ctx.config.knowledge_dir)
        return SkillResult(
            ok=True,
            message=(
                f"Indexed {summary['files_indexed']} document(s) into {summary['total_chunks']} "
                f"semantic vector chunk(s) from {ctx.config.knowledge_dir.name}/."
            ),
            data=summary,
        )
