#!/usr/bin/env node
// MCP stdio server exposing one tool, select_findings, which shows the user a
// checkbox list via MCP elicitation and returns the ids they picked.
//
// Zero dependencies: speaks newline-delimited JSON-RPC 2.0 on stdin/stdout
// directly, so it runs from a repo checkout with no install step. stdout
// carries protocol traffic only; diagnostics go to stderr.

import { createInterface } from "node:readline";

const SERVER_NAME = "review-picker";
const SERVER_VERSION = "0.1.0";
const FALLBACK_PROTOCOL = "2025-06-18";
const KNOWN_PROTOCOLS = new Set([
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
]);

// Titles longer than this are truncated so the dialog stays readable.
const MAX_LABEL = 200;

let clientSupportsElicitation = false;
let nextRequestId = 1;
const pendingRequests = new Map();

function send(message)
{
    process.stdout.write(JSON.stringify(message) + "\n");
}

function sendResult(id, result)
{
    send({ jsonrpc: "2.0", id, result });
}

function sendError(id, code, message)
{
    send({ jsonrpc: "2.0", id, error: { code, message } });
}

function log(message)
{
    process.stderr.write(`[${SERVER_NAME}] ${message}\n`);
}

// Sends a server->client request and resolves with the client's result.
function request(method, params)
{
    return new Promise((resolve, reject) => {
        const id = `srv-${nextRequestId++}`;
        pendingRequests.set(id, { resolve, reject });
        send({ jsonrpc: "2.0", id, method, params });
    });
}

const TOOL = {
    name: "select_findings",
    description:
        "Ask the user to choose items from a list, shown as a checkbox dialog. " +
        "Pass every candidate item with a stable id and a one-line label; the " +
        "result names the ids the user selected. Use this before acting on a " +
        "set of items the user should filter, such as applying a subset of " +
        "code-review findings. Returns action 'accept' with the chosen ids, " +
        "'decline' or 'cancel' if the user dismissed the dialog, or " +
        "'unsupported' when this client cannot show one.",
    inputSchema: {
        type: "object",
        properties: {
            message: {
                type: "string",
                description: "Prompt shown above the checkboxes, e.g. 'Which findings should I fix?'",
            },
            title: {
                type: "string",
                description: "Short heading for the checkbox group.",
            },
            items: {
                type: "array",
                description: "The selectable items, in the order they should appear.",
                items: {
                    type: "object",
                    properties: {
                        id: {
                            type: "string",
                            description: "Stable identifier returned when this item is selected.",
                        },
                        label: {
                            type: "string",
                            description: "One-line description shown next to the checkbox.",
                        },
                    },
                    required: ["id", "label"],
                },
            },
            preselect: {
                type: "array",
                description: "Ids checked by default when the dialog opens.",
                items: { type: "string" },
            },
        },
        required: ["items"],
    },
};

function normalizeItems(rawItems)
{
    const seen = new Set();
    const items = [];
    for(const raw of rawItems)
    {
        if(raw === null || typeof raw !== "object")
            continue;
        const id = String(raw.id ?? "").trim();
        const label = String(raw.label ?? raw.id ?? "").trim();
        if(id === "" || label === "")
            continue;
        if(seen.has(id))
        {
            log(`dropping duplicate item id ${id}`);
            continue;
        }
        seen.add(id);
        items.push({
            id,
            label: label.length > MAX_LABEL ? `${label.slice(0, MAX_LABEL - 1)}…` : label,
        });
    }
    return items;
}

async function selectFindings(args)
{
    const items = normalizeItems(Array.isArray(args?.items) ? args.items : []);
    if(items.length === 0)
        return { action: "accept", selected: [], note: "no items to choose from" };

    if(not(clientSupportsElicitation))
    {
        return {
            action: "unsupported",
            selected: [],
            note: "client did not advertise the elicitation capability",
        };
    }

    const ids = new Set(items.map(item => item.id));
    const preselect = (Array.isArray(args?.preselect) ? args.preselect : [])
        .map(String)
        .filter(id => ids.has(id));

    const selectedSchema = {
        type: "array",
        title: String(args?.title ?? "Select items"),
        description: "Check every item to act on; leave a box clear to skip it.",
        minItems: 0,
        maxItems: items.length,
        items: { anyOf: items.map(item => ({ const: item.id, title: item.label })) },
    };
    if(preselect.length > 0)
        selectedSchema.default = preselect;

    let response;
    try
    {
        response = await request("elicitation/create", {
            mode: "form",
            message: String(args?.message ?? "Select the items to act on."),
            requestedSchema: {
                type: "object",
                properties: { selected: selectedSchema },
            },
        });
    }
    catch(error)
    {
        log(`elicitation failed: ${error?.message ?? error}`);
        return {
            action: "unsupported",
            selected: [],
            note: `elicitation request failed: ${error?.message ?? error}`,
        };
    }

    const action = response?.action ?? "cancel";
    if(action !== "accept")
        return { action, selected: [] };

    const raw = response?.content?.selected;
    const selected = (Array.isArray(raw) ? raw : []).map(String).filter(id => ids.has(id));
    return { action: "accept", selected };
}

// `not` keeps the negation readable next to the async calls above.
function not(value)
{
    return !value;
}

async function handleRequest(message)
{
    const { id, method, params } = message;
    switch(method)
    {
    case "initialize":
    {
        const requested = params?.protocolVersion;
        const protocolVersion =
            typeof requested === "string" && KNOWN_PROTOCOLS.has(requested)
                ? requested
                : FALLBACK_PROTOCOL;
        clientSupportsElicitation = Boolean(params?.capabilities?.elicitation);
        if(not(clientSupportsElicitation))
            log("client did not advertise elicitation; select_findings will report 'unsupported'");
        sendResult(id, {
            protocolVersion,
            capabilities: { tools: {} },
            serverInfo: { name: SERVER_NAME, version: SERVER_VERSION },
        });
        return;
    }
    case "ping":
        sendResult(id, {});
        return;
    case "tools/list":
        sendResult(id, { tools: [TOOL] });
        return;
    case "tools/call":
    {
        if(params?.name !== TOOL.name)
        {
            sendError(id, -32602, `Unknown tool: ${params?.name}`);
            return;
        }
        try
        {
            const result = await selectFindings(params?.arguments ?? {});
            sendResult(id, {
                content: [{ type: "text", text: JSON.stringify(result) }],
            });
        }
        catch(error)
        {
            sendError(id, -32603, `select_findings failed: ${error?.message ?? error}`);
        }
        return;
    }
    default:
        sendError(id, -32601, `Method not found: ${method}`);
    }
}

function handleMessage(message)
{
    // A response to an elicitation request we sent.
    if(message.id !== undefined && message.method === undefined)
    {
        const pending = pendingRequests.get(message.id);
        if(pending === undefined)
            return;
        pendingRequests.delete(message.id);
        if(message.error)
            pending.reject(new Error(message.error.message ?? "client returned an error"));
        else
            pending.resolve(message.result);
        return;
    }

    // A notification needs no reply.
    if(message.id === undefined)
        return;

    handleRequest(message).catch(error => {
        log(`unhandled error: ${error?.stack ?? error}`);
        sendError(message.id, -32603, String(error?.message ?? error));
    });
}

const lines = createInterface({ input: process.stdin });
lines.on("line", line => {
    const text = line.trim();
    if(text === "")
        return;
    let message;
    try
    {
        message = JSON.parse(text);
    }
    catch(error)
    {
        log(`ignoring unparseable line: ${error?.message ?? error}`);
        return;
    }
    handleMessage(message);
});
lines.on("close", () => process.exit(0));
