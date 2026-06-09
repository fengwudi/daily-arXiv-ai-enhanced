import os
import json
import sys
import re
import time
from typing import List, Dict

import requests
import dotenv
import argparse
from tqdm import tqdm
from anthropic import Anthropic
from json_repair import repair_json

if os.path.exists('.env'):
    dotenv.load_dotenv()

template = open("template.txt", "r").read()
system_prompt = open("system.txt", "r").read()

# Anthropic 配置（从 test_anthropic_sdk.py 迁移）
ANTHROPIC_BASE_URL = os.environ.get(
    "ANTHROPIC_BASE_URL", "https://idealab.alibaba-inc.com/api/anthropic"
)
ANTHROPIC_AUTH_TOKEN = os.environ.get(
    "ANTHROPIC_AUTH_TOKEN", "03bc7ddb1b58b382eb44c1bfe3cdd822"
)
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")

# 模拟 claude-code CLI 客户端
CLI_HEADERS = {
    "User-Agent": "claude-cli/1.0.60 (external, cli)",
    "anthropic-beta": "claude-code-20250219",
    "x-app": "cli",
}

# JSON 输出格式说明
JSON_SCHEMA_INSTRUCTION = """
You MUST respond with a valid JSON object (no markdown, no code fences) with exactly these 5 fields:
{
  "tldr": "a concise TL;DR summary",
  "motivation": "the motivation of this paper",
  "method": "the method proposed",
  "result": "key results",
  "conclusion": "conclusion"
}
"""


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="(unused, kept for compatibility)")
    return parser.parse_args()


def build_client() -> Anthropic:
    """构造走 Bearer 鉴权 + CLI 伪装头的 Anthropic 客户端"""
    # 防止 SDK 自动读取 ANTHROPIC_API_KEY
    os.environ.pop("ANTHROPIC_API_KEY", None)
    return Anthropic(
        base_url=ANTHROPIC_BASE_URL,
        auth_token=ANTHROPIC_AUTH_TOKEN,
        default_headers=CLI_HEADERS,
        timeout=120.0,
        max_retries=2,
    )


def call_ai(client: Anthropic, content: str, language: str) -> Dict:
    """调用 Anthropic API 分析论文摘要，返回结构化 JSON"""
    full_system = system_prompt.replace("{language}", language) + "\n" + JSON_SCHEMA_INSTRUCTION
    user_message = template.replace("{content}", content)

    resp = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        system=full_system,
        messages=[{"role": "user", "content": user_message}],
    )

    # 提取文本
    text = "".join(
        block.text for block in resp.content
        if getattr(block, "type", None) == "text"
    )

    # 清理：去掉 <think>...</think> 标签（思考型模型可能带）
    text = text.strip()
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # 清理 markdown code fence
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 使用 json_repair 修复常见问题（未转义引号、换行等）
    try:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    # 兜底：用正则提取第一个 {...} JSON 对象
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            repaired = repair_json(match.group(0), return_objects=True)
            if isinstance(repaired, dict):
                return repaired

    raise ValueError(f"Cannot extract JSON from response: {text[:200]}")


def check_github_code(content: str) -> Dict:
    """提取并验证 GitHub 链接"""
    code_info = {}

    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)

    if match:
        owner, repo = match.groups()
        repo = repo.rstrip(".git").rstrip(".,)")
        full_url = f"https://github.com/{owner}/{repo}"
        code_info["code_url"] = full_url

        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"

        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            pass
        return code_info

    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)
    if match_io:
        url = match_io.group(0).rstrip(".,)")
        code_info["code_url"] = url

    return code_info


def process_single_item(client: Anthropic, item: Dict, language: str) -> Dict:
    """处理单篇论文"""
    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    try:
        ai_result = call_ai(client, item['summary'], language)
        item['AI'] = ai_result
    except json.JSONDecodeError as e:
        print(f"JSON parse error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    except Exception as e:
        print(f"Error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields

    # 确保所有字段都存在
    for field in default_ai_fields:
        if field not in item.get('AI', {}):
            item.setdefault('AI', {})[field] = default_ai_fields[field]

    return item


def process_all_items(data: List[Dict], language: str) -> List[Dict]:
    """串行处理所有论文，带速率限制和重试"""
    client = build_client()
    print(f'🔗 Anthropic API: {ANTHROPIC_BASE_URL}', file=sys.stderr)
    print(f'🤖 Model: {ANTHROPIC_MODEL}', file=sys.stderr)
    print(f'📊 论文总数: {len(data)} 篇', file=sys.stderr)

    # 速率限制配置
    rate_limit = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
    interval = 60.0 / rate_limit
    max_retries = 3

    print(f'⏱️  速率限制: {rate_limit} 次/分钟 (间隔 {interval:.1f}s)', file=sys.stderr)

    processed_data = []
    for item in tqdm(data, desc="Processing papers"):
        success = False

        for attempt in range(max_retries):
            try:
                result = process_single_item(client, item, language)
                processed_data.append(result)
                success = True
                break
            except Exception as e:
                error_msg = str(e)
                if 'rate' in error_msg.lower() or 'IRC-001' in error_msg or '超过' in error_msg:
                    wait_time = 65
                    print(f'\n⏳ 速率限制，等待 {wait_time}s (attempt {attempt+1}/{max_retries})...', file=sys.stderr)
                    time.sleep(wait_time)
                elif '429' in error_msg:
                    wait_time = 30
                    print(f'\n⏳ 429 Too Many Requests，等待 {wait_time}s...', file=sys.stderr)
                    time.sleep(wait_time)
                else:
                    print(f"Fatal error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
                    break

        if not success:
            item['AI'] = {
                "tldr": "Processing failed",
                "motivation": "Processing failed",
                "method": "Processing failed",
                "result": "Processing failed",
                "conclusion": "Processing failed"
            }
            processed_data.append(item)

        # 速率控制
        time.sleep(interval)

    return processed_data


def main():
    args = parse_args()
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print(f'Open: {args.data} ({len(data)} unique papers)', file=sys.stderr)

    # 处理所有论文
    processed_data = process_all_items(data, language)

    # 保存结果
    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

    print(f'✅ 完成！输出: {target_file}', file=sys.stderr)


if __name__ == "__main__":
    main()
