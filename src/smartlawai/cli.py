"""smartlaw ingest / smartlaw ask -- CLI entrypoint over Pipeline, so a request
can flow through every stage end-to-end from a terminal (PLAN.md M0's literal
done-when criterion)."""
from __future__ import annotations

import json

import click

from smartlawai import config
from smartlawai.adapters.factory import get_backend
from smartlawai.pipeline import Pipeline
from smartlawai.scope import Scope


@click.group()
def cli() -> None:
    """SmartLawAI command-line interface."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--doc-type", default="JUDGMENT")
@click.option("--owner-id", default=config.DEFAULT_OWNER_ID)
def ingest(path: str, doc_type: str, owner_id: str) -> None:
    be = get_backend()
    doc_id, trace = Pipeline(be).ingest(path, doc_type=doc_type, source="cli",
                                        owner_id=owner_id)
    click.echo(f"doc_id: {doc_id}")
    click.echo(json.dumps(trace.to_dict(), indent=2, default=str))


@cli.command()
@click.option("--doc-id", "doc_ids", required=True, multiple=True,
             help="Repeatable: --doc-id X --doc-id Y to scope to several documents.")
@click.option("--owner-id", default=config.DEFAULT_OWNER_ID)
@click.option("--question", required=True)
@click.option("--json", "as_json", is_flag=True, default=False)
def ask(doc_ids: tuple[str, ...], owner_id: str, question: str, as_json: bool) -> None:
    be = get_backend()
    scope = Scope(doc_ids=doc_ids, owner_id=owner_id)
    result = Pipeline(be).ask(question, scope)
    payload = {"answer": result.answer, "decision": result.decision,
               "trace": result.trace.to_dict()}
    if as_json:
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo(f"Decision: {result.decision}")
        click.echo(f"Answer: {result.answer}")
        click.echo(json.dumps(payload["trace"], indent=2, default=str))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
