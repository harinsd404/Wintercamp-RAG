from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mcp import Client

app = FastAPI()

mcp: Client | None = None


class RepoRequest(BaseModel):
    owner: str
    repo: str
    branch: str = "main"


@app.on_event("startup")
async def startup():
    global mcp
    mcp = Client(
        transport="stdio",
        command="npx",
        args=["@modelcontextprotocol/server-github"],
        env={
            "GITHUB_TOKEN": "YOUR_GITHUB_TOKEN"
        }
    )
    await mcp.__aenter__()


@app.on_event("shutdown")
async def shutdown():
    global mcp
    if mcp:
        await mcp.__aexit__(None, None, None)


@app.post("/github/tree")
async def fetch_repo_tree(body: RepoRequest):
    if not mcp:
        raise HTTPException(status_code=500, detail="MCP not ready")

    result = await mcp.call_tool(
        "github.get_repository_tree",
        {
            "owner": body.owner,
            "repo": body.repo,
            "branch": body.branch
        }
    )
    return result