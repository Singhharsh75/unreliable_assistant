"""Gate 10: W3C TraceContext spans on every hop, exported over OTLP (design.md §1.4 span names).

Code calls `span(...)` on the OpenTelemetry API, which is a no-op until `setup_tracing` installs the SDK.
With OTEL_EXPORTER_OTLP_ENDPOINT set, spans go to the OTel Collector (Tempo for traces, spanmetrics for
Prometheus; see observability/otel-collector.yaml). Every request's spans share the trace id from its `traceparent`.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

# hook.pre and cache.lookup are reserved: the pre-hooks (Gate 13) and the semantic cache (Gate 17) are not built yet.
SPANS = ("hook.pre", "cache.lookup", "triage.classify", "mcp.client", "mcp.rag.bm25", "mcp.rag.biencoder",
         "mcp.rag.rrf", "mcp.rag.rerank", "agent.query", "agent.react.step", "llm.generate", "verify.nli",
         "hook.post.ragas")

tracer = trace.get_tracer("unreliable_assistant")
_propagator = TraceContextTextMapPropagator()


def parent_context(traceparent: str | None) -> otel_context.Context | None:
    """The active span when there is one (in-process call), else the remote parent named by `traceparent`."""
    if trace.get_current_span().get_span_context().is_valid:
        return otel_context.get_current()
    return _propagator.extract({"traceparent": traceparent}) if traceparent else None


def remote_context(traceparent: str | None) -> otel_context.Context | None:
    """Parent from `traceparent` alone, ignoring whatever span is current (background work for another request)."""
    return _propagator.extract({"traceparent": traceparent}, context=otel_context.Context()) if traceparent else None


def current_traceparent(fallback: str) -> str:
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    tp = carrier.get("traceparent")
    # keep only the sampled bit: newer SDKs also set the W3C level-2 random bit (flags 03), which CallContext rejects
    return f"{tp[:-2]}{int(tp[-2:], 16) & 1:02x}" if tp else fallback


@contextmanager
def span(name: str, *, context: otel_context.Context | None = None, **attrs: Any) -> Iterator[trace.Span]:
    """A span that records the exception and an `error.code` attribute when the body raises."""
    attributes = {k: v for k, v in attrs.items() if v is not None}
    with tracer.start_as_current_span(name, context=context, attributes=attributes,
                                      record_exception=False, set_status_on_exception=False) as s:
        try:
            yield s
        except BaseException as e:
            s.set_attribute("error.code", getattr(e, "name", type(e).__name__))
            s.set_status(Status(StatusCode.ERROR, str(e)[:200]))
            raise


def setup_tracing(*processors: Any, service_name: str = "unreliable-assistant") -> Any:
    """Install the SDK provider once per process. Extra span processors (tests, WindowedMetrics) are added to it."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        trace.set_tracer_provider(provider)
    for p in processors:
        provider.add_span_processor(p)
    return provider
