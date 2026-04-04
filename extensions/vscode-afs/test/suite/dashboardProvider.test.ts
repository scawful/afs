import * as assert from "node:assert";
import { describe, it } from "node:test";
import { AfsDashboardProvider } from "../../src/views/dashboardProvider";
import { MockTransport } from "./mockTransport";

describe("AfsDashboardProvider", () => {
  it("renders session hint commands and actions when transport session info is available", () => {
    const transport = new MockTransport();
    transport.sessionInfo = {
      sessionId: "sess-vscode",
      payloadFile: "/tmp/session_client_vscode.json",
      contextPath: "/tmp/workspace/.context",
      promptJson: "/tmp/session_system_prompt_vscode.json",
      promptText: "/tmp/session_system_prompt_vscode.txt",
      workspace: "/tmp/workspace",
      cliHints: {
        workspacePath: "/tmp/workspace",
        queryShortcut: "afs query <text> --path /tmp/workspace",
        queryCanonical: "afs context query <text> --path /tmp/workspace",
        indexRebuild: "afs index rebuild --path /tmp/workspace",
        notes: ["Indexed retrieval may be stale."],
      },
    };

    const provider = new AfsDashboardProvider(
      transport,
      {
        appendLine() {},
        dispose() {},
      } as never,
    );

    const html = (provider as any).buildHtml(
      true,
      transport.capabilities(),
      null,
      null,
      null,
      null,
      transport.getSessionInfo(),
    ) as string;

    assert.match(html, /Session Hints/);
    assert.match(html, /afs query &lt;text&gt; --path \/tmp\/workspace/);
    assert.match(html, /afs context query &lt;text&gt; --path \/tmp\/workspace/);
    assert.match(html, /afs index rebuild --path \/tmp\/workspace/);
    assert.match(html, /Query More/);
    assert.match(html, /Copy Query Command/);
    assert.match(html, /Copy Rebuild Command/);
    assert.match(html, /Indexed retrieval may be stale\./);
  });
});
