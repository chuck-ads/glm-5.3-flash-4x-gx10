"""End-to-end checks of the fail-closed parser on real GLM tool-call text, fed
once whole (a non-streaming response) and once token by token (streaming).

The model writes arguments as <arg_key>/<arg_value> pairs, which the engine
turns into JSON; validating the raw text refused every streamed call. Needs the
checkpoint's tokenizer at /model and the plugin at /tmp/plugin.py.
"""
import json

from transformers import AutoTokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager as M

M.import_tool_parser("/tmp/plugin.py")
Parser = M.get_tool_parser("glm47_failclosed")
tok = AutoTokenizer.from_pretrained("/model", trust_remote_code=True)
tools = [
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
        "command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}}}}},
]
req = ChatCompletionRequest(model="glm53", messages=[{"role": "user", "content": "x"}], tools=tools)
R = "_rejected_by_server"
CASES = [  # (label, model output, [(name, argument keys or R)])
    ("good", "</think>Listing.<tool_call>bash<arg_key>command</arg_key><arg_value>ls /tmp</arg_value></tool_call>",
     [("bash", ["command"])]),
    ("two args", "</think><tool_call>bash<arg_key>command</arg_key><arg_value>sleep 1</arg_value>"
     "<arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call>", [("bash", ["command", "timeout"])]),
    ("key not in schema", "</think><tool_call>bash<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>",
     [("bash", R)]),
    ("good then bad", "</think><tool_call>read_file<arg_key>path</arg_key><arg_value>/etc/hosts</arg_value>"
     "</tool_call><tool_call>bash<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>",
     [("read_file", ["path"]), ("bash", R)]),
    ("markup after name", "</think><tool_call>bash</arg_key><arg_key>command</arg_key><arg_value>pwd"
     "</arg_value></tool_call>", [("bash", ["command"])]),
    ("tool not offered", "</think><tool_call>curl<arg_key>url</arg_key><arg_value>x</arg_value></tool_call>",
     [("curl", R)]),
]


def whole(text):
    info = Parser(tok, req.tools).extract_tool_calls(text, req)
    return [(c.function.name, c.function.arguments) for c in info.tool_calls]


def streamed(text):
    p = Parser(tok, req.tools)
    ids = tok.encode(text, add_special_tokens=False)
    names, args, prev_text = {}, {}, ""
    for i in range(len(ids)):
        cur = tok.decode(ids[: i + 1])
        d = p.extract_tool_calls_streaming(prev_text, cur, cur[len(prev_text):], ids[:i], ids[: i + 1], [ids[i]], req)
        for tc in (d.tool_calls or []) if d else []:
            if tc.function and tc.function.name:
                names[tc.index] = tc.function.name
            if tc.function and tc.function.arguments:
                args[tc.index] = args.get(tc.index, "") + tc.function.arguments
        prev_text = cur
    return [(names.get(k), args.get(k, "")) for k in sorted(set(names) | set(args))]


def shape(calls):
    out = []
    for name, arguments in calls:
        parsed = json.loads(arguments or "{}")
        out.append((name, R if R in parsed else sorted(parsed)))
    return out


for label, text, want in CASES:
    for mode, run in (("whole", whole), ("streamed", streamed)):
        got = shape(run(text))
        print(f"  {label:<18} {mode:<9} {'ok' if got == want else 'WANTED ' + str(want) + ' GOT'} {got}")
