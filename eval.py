import json
import re

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

dataset = load_json('/Users/zeta/Desktop/ByteBell/mimir/dataset.json')
q_mcp = load_json('/Users/zeta/Desktop/mimir/q_mcp.json')
q_no_mcp = load_json('/Users/zeta/Desktop/mimir/q.json')

# Normalize dataset keys
gt_dict = {}
for q in dataset['questions']:
    num = int(q['id'][1:])
    gt_dict[f"q{num}"] = q

def evaluate_correctness(answer_obj, gt_obj):
    answer_str = json.dumps(answer_obj).lower()
    gt = gt_obj['ground_truth']
    if isinstance(gt, list):
        hits = 0
        total = len(gt)
        missed = []
        for item in gt:
            file_part = item.split(':')[0].lower()
            if file_part in answer_str or item.lower() in answer_str:
                hits += 1
            else:
                missed.append(item)
        score = hits / total if total > 0 else 0
        return score, missed
    elif isinstance(gt, dict):
        return 1.0, []
    return 0.0, []

def extract_explanation(answer_obj):
    if isinstance(answer_obj, str):
        return answer_obj.replace('\n', ' ')
    if isinstance(answer_obj, dict):
        for key in ['root_cause', 'diagnosis', 'summary']:
            if key in answer_obj:
                val = answer_obj[key]
                if isinstance(val, str):
                    return val.replace('\n', ' ')
                elif isinstance(val, list):
                    return " ".join(val).replace('\n', ' ')
        return json.dumps(answer_obj)
    return "N/A"

def generate_report(name, pred_data, gt_data):
    lines = [f"# Report: {name} vs Dataset", ""]
    total_score = 0
    total_tools = 0
    total_questions = len(pred_data)
    
    for q_id, pred in pred_data.items():
        if q_id not in gt_data:
            continue
        
        tool_calls = len(pred.get('tool_calls', []))
        total_tools += tool_calls
        
        score, missed = evaluate_correctness(pred.get('answer', {}), gt_data[q_id])
        total_score += score
        
        explanation = extract_explanation(pred.get('answer', {}))
        
        lines.append(f"## {q_id}")
        lines.append(f"- **Correctness Score**: {score:.2f}")
        if missed:
            lines.append(f"- **Missed Ground Truth**: {', '.join(missed)}")
        lines.append(f"- **Tool Calls**: {tool_calls}")
        lines.append(f"- **Root Cause / Summary**:\n  > {explanation}")
        lines.append("")
        
    avg_score = total_score / total_questions if total_questions > 0 else 0
    lines.insert(2, f"**Average Correctness**: {avg_score:.2f}")
    lines.insert(3, f"**Total Tool Calls**: {total_tools}")
    lines.insert(4, "")
    
    return "\n".join(lines)

def generate_comparison_report(q_data, mcp_data, gt_data):
    lines = ["# Report: q.json vs q_mcp.json", ""]
    
    total_q_tools = 0
    total_mcp_tools = 0
    total_q_score = 0
    total_mcp_score = 0
    
    for q_id in q_data.keys():
        q_tools = len(q_data[q_id].get('tool_calls', []))
        mcp_tools = len(mcp_data.get(q_id, {}).get('tool_calls', []))
        total_q_tools += q_tools
        total_mcp_tools += mcp_tools
        
        q_score, _ = evaluate_correctness(q_data[q_id].get('answer', {}), gt_data[q_id])
        mcp_score, _ = evaluate_correctness(mcp_data.get(q_id, {}).get('answer', {}), gt_data[q_id])
        
        total_q_score += q_score
        total_mcp_score += mcp_score
        
        q_explanation = extract_explanation(q_data[q_id].get('answer', {}))
        mcp_explanation = extract_explanation(mcp_data.get(q_id, {}).get('answer', {}))
        
        lines.append(f"## {q_id}")
        lines.append(f"### q.json")
        lines.append(f"- **Tool Calls**: {q_tools}, **Score**: {q_score:.2f}")
        lines.append(f"- **Root Cause / Summary**:\n  > {q_explanation}\n")
        
        lines.append(f"### q_mcp.json")
        lines.append(f"- **Tool Calls**: {mcp_tools}, **Score**: {mcp_score:.2f}")
        lines.append(f"- **Root Cause / Summary**:\n  > {mcp_explanation}\n")
        lines.append("---")
        lines.append("")
        
    total_questions = len(q_data)
    avg_q_score = total_q_score / total_questions if total_questions > 0 else 0
    avg_mcp_score = total_mcp_score / total_questions if total_questions > 0 else 0
    
    lines.insert(2, f"**Total q.json Tool Calls**: {total_q_tools}")
    lines.insert(3, f"**Total q_mcp.json Tool Calls**: {total_mcp_tools}")
    lines.insert(4, f"**Average q.json Score**: {avg_q_score:.2f}")
    lines.insert(5, f"**Average q_mcp.json Score**: {avg_mcp_score:.2f}")
    lines.insert(6, "")
    
    return "\n".join(lines)

report1 = generate_report("q_mcp.json", q_mcp, gt_dict)
report2 = generate_report("q.json", q_no_mcp, gt_dict)
report3 = generate_comparison_report(q_no_mcp, q_mcp, gt_dict)

with open('/Users/zeta/Desktop/mimir/report_q_mcp_vs_dataset.md', 'w') as f:
    f.write(report1)

with open('/Users/zeta/Desktop/mimir/report_q_vs_dataset.md', 'w') as f:
    f.write(report2)

with open('/Users/zeta/Desktop/mimir/report_q_vs_q_mcp.md', 'w') as f:
    f.write(report3)

print("Reports updated to include root cause!")
