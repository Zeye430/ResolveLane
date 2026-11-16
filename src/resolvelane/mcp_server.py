"""The help center as an MCP server, so other assistants and tools can search it.

Only the public help center is exposed: no cases, orders or customer data, so the server
needs no customer identity. Search covers the article versions in effect when it is called,
as in the reply agent, and each passage carries its source ID for citation.

Run over stdio, for example from an MCP client's configuration::

    python -m resolvelane.mcp_server
"""

from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from resolvelane.knowledge import HelpCenter
from resolvelane.recognition import load_local_embedder
from resolvelane.simulator.help_center import ARTICLES


class HelpCenterPassage(BaseModel):
    source_id: str
    article: str
    section: str
    text: str


class HelpCenterResults(BaseModel):
    passages: list[HelpCenterPassage]
    search: str = Field(description="hybrid (keywords and embeddings) or keyword")


def build_server(help_center: HelpCenter) -> MCPServer:
    server = MCPServer("ResolveLane help center", version="0.1.0",
                       instructions="Search the simulated marketplace's help center. Answers should cite source_id.")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                                             openWorldHint=False))
    def search_help_center(query: str = Field(min_length=3, max_length=200)) -> HelpCenterResults:
        """Search the help center articles in effect now. Returns no passages when nothing matches."""
        result = help_center.search(query, datetime.now(timezone.utc))
        return HelpCenterResults(passages=[HelpCenterPassage(source_id=passage.source_id,
                                                             article=passage.citation.title,
                                                             section=passage.heading, text=passage.text)
                                           for passage in result.passages], search=result.mode)

    return server


if __name__ == "__main__":
    build_server(HelpCenter(ARTICLES, load_local_embedder())).run(transport="stdio")
