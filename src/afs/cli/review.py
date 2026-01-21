import os
import shutil


def register_parsers(subparsers):
    review_parser = subparsers.add_parser("review", help="Review and manage agent-generated documents.")
    review_subparsers = review_parser.add_subparsers(dest="review_command")

    # list
    list_parser = review_subparsers.add_parser("list", help="List all pending reviews.")
    list_parser.set_defaults(func=handle_list)

    # approve
    approve_parser = review_subparsers.add_parser("approve", help="Approve a document and move it to project memory.")
    approve_parser.add_argument("project", help="The name of the project.")
    approve_parser.add_argument("filename", help="The filename in the review queue.")
    approve_parser.set_defaults(func=handle_approve)

    # reject
    reject_parser = review_subparsers.add_parser("reject", help="Reject a document.")
    reject_parser.add_argument("project", help="The name of the project.")
    reject_parser.add_argument("filename", help="The filename in the review queue.")
    reject_parser.add_argument("--reason", "-r", help="Reason for rejection.")
    reject_parser.set_defaults(func=handle_reject)

def handle_list(args):
    context_root = get_base_context_root()
    review_root = os.path.join(context_root, "review")

    if not os.path.exists(review_root):
        print("No review queue found.")
        return 0

    categories = ["plans", "walkthroughs", "automated_reports"]
    for cat in categories:
        cat_path = os.path.join(review_root, cat)
        if not os.path.exists(cat_path):
            continue

        files = [f for f in os.listdir(cat_path) if not f.startswith(".")]
        if files:
            print(f"\n[{cat.upper()}]")
            for f in files:
                print(f"  - {f}")

    return 0

def handle_approve(args):
    context_root = get_base_context_root()
    review_root = os.path.join(context_root, "review")
    project_root = os.path.join(context_root, "projects", args.project)

    if not os.path.exists(project_root):
        print(f"Error: Project '{args.project}' not found in {context_root}/projects/")
        return 1

    # Find the file in the review queue
    found_path = None
    category = None
    for cat in ["plans", "walkthroughs", "automated_reports"]:
        potential_path = os.path.join(review_root, cat, args.filename)
        if os.path.exists(potential_path):
            found_path = potential_path
            category = cat
            break

    if not found_path:
        print(f"Error: File '{args.filename}' not found in review queue.")
        return 1

    # Determine destination
    if category == "plans":
        dest_dir = os.path.join(project_root, "memory")
    else:
        dest_dir = os.path.join(project_root, "history")

    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, args.filename)

    shutil.move(found_path, dest_path)
    print(f"Approved and moved to {dest_path}")

    # Proactive: Update task.md if it exists
    task_file = os.path.join(project_root, "task.md")
    if os.path.exists(task_file):
        with open(task_file, "a") as f:
            f.write(f"\n- [x] Approved {category[:-1]} {args.filename}")
        print(f"Updated {task_file}")

    return 0

def handle_reject(args):
    context_root = get_base_context_root()
    review_root = os.path.join(context_root, "review")

    # Simplified reject: move to a 'rejected' subfolder in history of the project
    # or just delete and notify. For now, let's just move it to project history with a suffix.

    project_root = os.path.join(context_root, "projects", args.project)
    if not os.path.exists(project_root):
        print(f"Error: Project '{args.project}' not found.")
        return 1

    found_path = None
    for cat in ["plans", "walkthroughs", "automated_reports"]:
        potential_path = os.path.join(review_root, cat, args.filename)
        if os.path.exists(potential_path):
            found_path = potential_path
            break

    if not found_path:
        print(f"Error: File '{args.filename}' not found.")
        return 1

    history_dir = os.path.join(project_root, "history")
    os.makedirs(history_dir, exist_ok=True)
    dest_path = os.path.join(history_dir, f"REJECTED_{args.filename}")

    shutil.move(found_path, dest_path)
    print(f"Rejected and moved to {dest_path}")

    if args.reason:
        # Write reason to a companion file or scratchpad
        reason_file = dest_path + ".reason"
        with open(reason_file, "w") as f:
            f.write(args.reason)
        print(f"Rejection reason saved to {reason_file}")

    return 0

def get_base_context_root():
    # Helper to get ~/.context even if current working dir isn't a project
    home = os.path.expanduser("~")
    return os.path.join(home, ".context")
