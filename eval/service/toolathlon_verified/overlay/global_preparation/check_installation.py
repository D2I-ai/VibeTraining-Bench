from utils.mcp.tool_servers import MCPServerManager, call_tool_with_retry
import asyncio
import os
import json
import yaml
from utils.general.helper import print_color
from configs.global_configs import global_configs

async def main():

    to_check_servers = []

    for server_file_path in os.listdir("configs/mcp_servers"):
        with open(os.path.join("configs/mcp_servers", server_file_path), "r") as f:
            server_config = yaml.safe_load(f)
        server_name = server_config["name"]
        # print(server_name)
        to_check_servers.append(server_name)
    
    
    # create a ./dumps/mcp_servers_check directory
    os.makedirs("dumps_for_mcp_check/mcp_servers_check", exist_ok=True)

    xx_MCPServerManager = MCPServerManager(agent_workspace="./dumps_for_mcp_check/mcp_servers_check") # a pseudo server manager
    
    for server_name in to_check_servers:
        # huggingface IS verified now (a real hf_ token is configured); only the
        # notion servers stay skipped (their page-id token is a per-task
        # KEEP_IT_ASIS placeholder, so a global check cannot pass by design).
        if server_name in ("notion", "notion_official"):
            # we do not verify this server
            continue
        server_x = xx_MCPServerManager.servers[server_name]
        
        max_retries = 4
        # Per-server wall-clock cap. Without it a server whose stdio process
        # dies mid-handshake (e.g. k8s here: its kubeconfig_path is a
        # KEEP_IT_ASIS placeholder that only exists inside a task-specific dir,
        # so mcp-server-kubernetes crashes on startup) makes `async with` block
        # until client_session_timeout_seconds (600s) x max_retries, wedging the
        # whole run on the FIRST server. wait_for turns that hang into a normal
        # retryable error so the sweep always progresses.
        per_server_timeout = float(os.environ.get("MCP_CHECK_PER_SERVER_TIMEOUT", "90"))

        async def _list_once():
            async with server_x as _srv:
                return await _srv.list_tools()

        for attempt in range(1, max_retries + 1):
            try:
                print_color(f"Server {server_name} is checking (attempt {attempt}) ... ", "yellow")
                tools = await asyncio.wait_for(_list_once(), timeout=per_server_timeout)
                print_color(f"Server {server_name} is checked", "green")
                break
            except Exception as e:
                if attempt < max_retries:
                    print_color(f"Server {server_name} check failed (attempt {attempt}): {e}", "red")
                    await asyncio.sleep(0.8)
                    continue
                else:
                    print_color(f"Server {server_name} is checked with error after {max_retries} attempts: {e}", "red")
                    break
        
    
    

    # print_color("All servers are checked", "green")
if __name__ == "__main__":
    asyncio.run(main())
