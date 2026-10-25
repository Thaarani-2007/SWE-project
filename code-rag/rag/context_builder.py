from typing import Any, Dict, List


def build_rag_context(
    documents: List[Dict[str, Any]],
    max_characters: int = 12000,
) -> str:
    """Assembles structured markdown context with line numbers, metadata,

    and relationship tags within a strict character budget.

    Format:
    ### File: <file_path> (Lines <start>-<end>) [Symbol: <symbol_name>]
    # Relationship: <Direct Match | Caller of X | Callee of X | Data-Flow>
    <code_body_with_line_numbers>
    """
    if not documents:
        return "No relevant code context found."

    context_sections: List[str] = []
    current_length = 0

    for doc in documents:
        meta = doc.get("meta") or doc.get("metadata") or {}
        text = doc.get("text", "")
        if not text:
            continue

        file_path = meta.get("file") or meta.get("file_path") or "unknown.py"
        lines_val = meta.get("lines")

        start_line = 1
        end_line = 1
        if isinstance(lines_val, str) and "-" in lines_val:
            parts = lines_val.split("-")
            try:
                start_line = int(parts[0])
                end_line = int(parts[1])
            except ValueError:
                pass
            lines_str = lines_val
        elif isinstance(lines_val, (list, tuple)) and len(lines_val) >= 2:
            start_line = int(lines_val[0])
            end_line = int(lines_val[1])
            lines_str = f"{start_line}-{end_line}"
        else:
            start_line = int(meta.get("start_line", 1))
            end_line = int(meta.get("end_line", start_line + len(text.splitlines()) - 1))
            lines_str = f"{start_line}-{end_line}"

        symbol_name = meta.get("name") or meta.get("symbol_id") or "snippet"
        # If symbol_id has ::, pick the leaf name
        if "::" in symbol_name:
            leaf = symbol_name.split("::")[-1]
            if leaf:
                symbol_name = leaf

        relationship = meta.get("relationship", "Direct Match")

        # Format code with line numbers
        lines = text.splitlines()
        numbered_lines = []
        for idx, line in enumerate(lines):
            line_no = start_line + idx
            numbered_lines.append(f"{line_no:4d} | {line}")
        numbered_code = "\n".join(numbered_lines)

        header = f"### File: {file_path} (Lines {lines_str}) [Symbol: {symbol_name}]\n# Relationship: {relationship}"
        section = f"{header}\n```python\n{numbered_code}\n```\n"

        if current_length >= max_characters:
            break

        if current_length + len(section) > max_characters:
            remaining = max_characters - current_length
            if remaining > 80:
                context_sections.append(section[:remaining] + "\n... [truncated]\n```\n")
            break

        context_sections.append(section)
        current_length += len(section)

    return "\n".join(context_sections)
